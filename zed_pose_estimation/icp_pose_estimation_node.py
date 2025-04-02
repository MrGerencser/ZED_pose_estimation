#!/usr/bin/env python3

import os
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation



class ICPPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('icp_pose_estimation_node')

        # Declare parameters
        self.declare_parameter('transformed_cloud_topic', '/perception/transformed_fused_point_cloud')
        self.declare_parameter('full_cloud_topic', '/perception/full_fused_point_cloud')
        self.declare_parameter('model_path', '/home/chris/franka_ros2_ws/src/zed_pose_estimation/models/objects/cone with planar surface.ply')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('voxel_size', 0.005)
        self.declare_parameter('icp_distance_threshold', 0.03)
        self.declare_parameter('visualize', True)

        # Get parameters
        self.transformed_topic = self.get_parameter('transformed_cloud_topic').get_parameter_value().string_value
        self.full_topic = self.get_parameter('full_cloud_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.icp_distance_threshold = self.get_parameter('icp_distance_threshold').get_parameter_value().double_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        
        # Store the most recent point clouds
        self.latest_transformed_cloud = None
        self.latest_full_cloud = None
        self.transform_frame_id = None
        self.full_frame_id = None
        
        # Load reference model (same as before)
        try:
            # Your existing model loading code
            original_model = o3d.io.read_point_cloud(model_path)
            if not original_model.has_points():
                self.get_logger().error(f"Model file {model_path} loaded but contains no points!")
                return

            self.get_logger().info(f"Loaded model with {len(original_model.points)} points from {model_path}")

            scale_factor = 0.001  # Convert mm to meters
            scaled_points = np.asarray(original_model.points) * scale_factor

            self.reference_model = o3d.geometry.PointCloud()
            self.reference_model.points = o3d.utility.Vector3dVector(scaled_points)
            # downsample
            self.reference_model = self.reference_model.voxel_down_sample(voxel_size=self.voxel_size)
            self.processed_ref_points = np.asarray(self.reference_model.points).copy()
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

        # Publisher setup (same as before)
        self.pose_publisher = self.create_publisher(PoseStamped, '/perception/object_pose', 10)
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        # QoS setup for both subscriptions
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscriptions to both point cloud topics
        self.create_subscription(
            PointCloud2,
            self.transformed_topic,
            self.transformed_cloud_callback,
            qos_profile
        )
        
        self.create_subscription(
            PointCloud2,
            self.full_topic,
            self.full_cloud_callback,
            qos_profile
        )
        
        # After creating subscriptions, add:
        self.get_logger().info(f"Subscribing to transformed cloud topic: {self.transformed_topic}")
        self.get_logger().info(f"Subscribing to full cloud topic: {self.full_topic}")

        # Create timer for processing both clouds (runs every 1 second)
        self.create_timer(1.0, self.process_point_clouds)
        
        self.get_logger().info('ICP Pose Estimation Node initialized')
    
    def transformed_cloud_callback(self, msg):
        """Process incoming segmented point cloud messages."""
        try:
            points = np.array([list(p)[:3] for p in pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )], dtype=np.float64)

            if len(points) == 0:
                self.get_logger().warn("Received empty transformed point cloud")
                return

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            
            # Store the processed cloud and frame_id
            self.latest_transformed_cloud = self.preprocess_point_cloud(cloud)
            self.transform_frame_id = msg.header.frame_id
            self.get_logger().debug("Received transformed point cloud")
            
        except Exception as e:
            self.get_logger().error(f"Error processing transformed point cloud: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
    def full_cloud_callback(self, msg):
        """Process incoming full point cloud messages."""
        try:
            points = np.array([list(p)[:3] for p in pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )], dtype=np.float64)

            if len(points) == 0:
                self.get_logger().warn("Received empty full point cloud")
                return

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            
            # Store the processed cloud and frame_id
            self.latest_full_cloud = self.preprocess_point_cloud(cloud)
            self.full_frame_id = msg.header.frame_id
            self.get_logger().debug("Received full point cloud")
            
        except Exception as e:
            self.get_logger().error(f"Error processing full point cloud: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def process_point_clouds(self):
        """Process both point clouds if available."""
        if self.latest_full_cloud is None:
            self.get_logger().info("Waiting for full point cloud...")
            return
        
        if self.latest_transformed_cloud is None:
            self.get_logger().info("Waiting for transformed point cloud...")
            return
            
        # We have both clouds, proceed with pose estimation
        self.get_logger().info("Processing point clouds for pose estimation...")
        
        try:
            # Get pose using the transformed cloud for center alignment
            # but the full cloud for ICP
            pose_matrix = self.estimate_pose_with_both_clouds(
                self.latest_transformed_cloud, 
                self.latest_full_cloud
            )
            
            if pose_matrix is not None:
                # Extract and publish pose (same as before)
                position = np.array(pose_matrix[:3, 3], copy=True)
                rotation_matrix = np.array(pose_matrix[:3, :3], copy=True)
                rotation = Rotation.from_matrix(rotation_matrix)
                quat = rotation.as_quat()  # x, y, z, w

                pose_msg = PoseStamped()
                pose_msg.header = Header()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.transform_frame_id  # Use transformed cloud frame
                pose_msg.pose.position.x = float(position[0])
                pose_msg.pose.position.y = float(position[1])
                pose_msg.pose.position.z = float(position[2])
                pose_msg.pose.orientation.x = float(quat[0])
                pose_msg.pose.orientation.y = float(quat[1])
                pose_msg.pose.orientation.z = float(quat[2])
                pose_msg.pose.orientation.w = float(quat[3])

                self.pose_publisher.publish(pose_msg)
                self.get_logger().info(f"Object pose: Position[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] "
                                    f"Orientation(quat)[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")

                if self.publish_tf:
                    self.publish_transform(position, quat, self.transform_frame_id)

                if self.visualize:
                    self.visualize_alignment(self.latest_full_cloud, pose_matrix)
                    
        except Exception as e:
            self.get_logger().error(f"Error in process_point_clouds: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud for alignment (downsampling, removing outliers)."""
        try:
            # Downsample using voxel grid filter
            pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            # Remove outliers if enough points
            if len(np.asarray(pcd_down.points)) > 50:
                pcd_down, _ = pcd_down.remove_statistical_outlier(
                    nb_neighbors=20, 
                    std_ratio=2.0
                )
                
            # Estimate normals for better registration if not already computed
            if not pcd_down.has_normals():
                pcd_down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.voxel_size * 2, 
                        max_nn=30
                    )
                )
                
            return pcd_down
            
        except Exception as e:
            self.get_logger().error(f"Error in preprocess_point_cloud: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return pcd  # Return the original if preprocessing fails
    
    def estimate_pose_with_both_clouds(self, transformed_cloud, full_cloud):
        """Use transformed cloud for initial alignment and full cloud for ICP."""
        if len(transformed_cloud.points) < 10 or len(full_cloud.points) < 10:
            self.get_logger().warn("Not enough points for registration")
            return None
            
        # Create a copy of the reference model
        reference_copy = o3d.geometry.PointCloud()
        reference_copy.points = o3d.utility.Vector3dVector(self.processed_ref_points.copy())
        
        # Get centers from transformed cloud for initial alignment
        target_center = transformed_cloud.get_center()
        source_center = reference_copy.get_center()
        translation = target_center - source_center
        
        # Translate reference model to match transformed cloud center
        reference_copy.translate(translation)
        
        # Calculate initial rotation options (try different orientations)
        best_fitness = 0.0
        best_transformation = None
        
        # Try different rotations around Z axis (common for tabletop objects)
        for angle_z in [0]:
            # Create rotation matrix for Z rotation
            rotation_z = Rotation.from_euler('z', angle_z, degrees=True).as_matrix()
            
            # Create rotated copy
            rotated_reference = o3d.geometry.PointCloud()
            rotated_reference.points = o3d.utility.Vector3dVector(self.processed_ref_points.copy())
            
            # Apply rotation and translation
            rotated_reference.rotate(rotation_z, center=rotated_reference.get_center())
            rotated_reference.translate(translation)
            
            # Prepare for ICP
            rotated_reference.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(30)
            )
            full_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(30)
            )
            
            # Run ICP with higher distance threshold for initial alignment
            result = o3d.pipelines.registration.registration_icp(
                rotated_reference, full_cloud,
                0.1,  # Higher threshold for initial matching
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            
            self.get_logger().info(f"ICP fitness for rotation {angle_z}°: {result.fitness:.3f}")
            
            # Keep track of best result
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_transformation = result.transformation
                best_rotation = angle_z
                best_reference = rotated_reference
        
        # If we got a decent alignment, refine with point-to-plane ICP
        if best_fitness > 0.001:
            self.get_logger().info(f"Best rotation: {best_rotation}° with fitness {best_fitness:.3f}")

            full_cloud.orient_normals_consistent_tangent_plane(k=30)
            best_reference.orient_normals_consistent_tangent_plane(k=30)

            
            # Refine with point-to-plane ICP
            refined = o3d.pipelines.registration.registration_icp(
                best_reference, full_cloud,
                self.icp_distance_threshold, best_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
            )
            
            self.get_logger().info(f"Refined ICP fitness: {refined.fitness:.3f}")
            
            if refined.fitness > best_fitness:
                best_fitness = refined.fitness
                best_transformation = refined.transformation
            
            # Visualize the best alignment before returning
            if self.visualize:
                self.get_logger().info("Visualizing best alignment...")
                o3d.visualization.draw_geometries([
                    full_cloud.paint_uniform_color([1, 0, 0]),  # Red
                    best_reference.paint_uniform_color([0, 1, 0])  # Green
                ])
            
            # Create transformation matrices for each step
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = translation

            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = Rotation.from_euler('z', best_rotation, degrees=True).as_matrix()

            # Compose transformations in the correct order (translation → rotation → ICP refinement)
            final_transform = best_transformation @ translation_matrix
            return final_transform
            
        else:
            self.get_logger().warn(f"Poor registration quality with best fitness: {best_fitness:.3f}")
            return None

    def publish_transform(self, position, quaternion, frame_id):
        try:
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = frame_id
            transform.child_frame_id = 'detected_object'
            transform.transform.translation.x = float(position[0])
            transform.transform.translation.y = float(position[1])
            transform.transform.translation.z = float(position[2])
            transform.transform.rotation.x = float(quaternion[0])
            transform.transform.rotation.y = float(quaternion[1])
            transform.transform.rotation.z = float(quaternion[2])
            transform.transform.rotation.w = float(quaternion[3])
            self.tf_broadcaster.sendTransform(transform)
        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {e}")

    def visualize_alignment(self, observed_cloud, transformation):
        try:
            obs_cloud_vis = o3d.geometry.PointCloud()
            obs_cloud_vis.points = o3d.utility.Vector3dVector(np.asarray(observed_cloud.points).copy())

            ref_model_vis = o3d.geometry.PointCloud()
            ref_model_vis.points = o3d.utility.Vector3dVector(self.processed_ref_points.copy())
            ref_model_vis.transform(transformation)

            obs_cloud_vis.paint_uniform_color([1, 0, 0])  # Red
            ref_model_vis.paint_uniform_color([0, 1, 0])  # Green

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            o3d.visualization.draw_geometries(
                [obs_cloud_vis, ref_model_vis, coord_frame],
                window_name="ICP Alignment Result"
            )
        except Exception as e:
            self.get_logger().error(f"Error in visualization: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = ICPPoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
