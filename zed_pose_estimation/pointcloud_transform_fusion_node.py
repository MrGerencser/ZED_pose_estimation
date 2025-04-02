import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import torch
import open3d as o3d
import yaml

from zed_pose_estimation.pose_estimation_utils import (
    visualize_point_cloud, 
    filter_outliers_sor, 
    fuse_point_clouds_centroid
)


class PointCloudTransformFusionNode(Node):
    def __init__(self):
        super().__init__('pointcloud_transform_fusion_node')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Declare parameters
        self.declare_parameter('camera_topics', ['/zed1/point_cloud/cloud_registered', '/zed2/point_cloud/cloud_registered'])
        # self.declare_parameter('camera_topics', ['/zed1/segmented_point_cloud', '/zed2/segmented_point_cloud'])
        self.declare_parameter('target_frame', 'panda_link0')
        self.declare_parameter('transform_yaml_path', '')
        self.declare_parameter('voxel_size', 0.005)
        self.declare_parameter('max_points_per_cloud', 100000)
        self.declare_parameter('use_workspace_crop', True)
        self.declare_parameter('visualize', False)
        self.declare_parameter('transform_and_fuse_of_full_clouds', False)

        # Get parameters
        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.transform_yaml_path = self.get_parameter('transform_yaml_path').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.max_points_per_cloud = self.get_parameter('max_points_per_cloud').get_parameter_value().integer_value
        self.use_workspace_crop = self.get_parameter('use_workspace_crop').get_parameter_value().bool_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.transform_and_fuse_of_full_clouds = self.get_parameter('transform_and_fuse_of_full_clouds').get_parameter_value().bool_value

        # Variables for storing processed point clouds
        self.processed_cloud1 = None
        self.processed_cloud2 = None
        self.cloud1_header = None
        self.cloud2_header = None

        # Load transformations
        self.load_transformations()

        # Subscriptions
        self.create_subscription(PointCloud2, self.camera_topics[0], self.cam1_callback, 10)
        self.create_subscription(PointCloud2, self.camera_topics[1], self.cam2_callback, 10)
  

        # Publisher
        if self.transform_and_fuse_of_full_clouds:
            self.fused_publisher = self.create_publisher(PointCloud2, '/perception/full_fused_point_cloud', 1)
        else:
            self.fused_publisher = self.create_publisher(PointCloud2, '/perception/transformed_fused_point_cloud', 1)



    def load_transformations(self):
        """Load transformation matrices from the YAML file."""
        try:
            with open(self.transform_yaml_path, 'r') as file:
                transforms = yaml.safe_load(file)['transforms']
                self.T_0S = np.array(transforms['T_0S'])
                self.H1 = np.array(transforms['H1'])
                self.H2 = np.array(transforms['H2'])
                self.get_logger().info("Loaded transformation matrices from YAML")
                # Print transformations
                self.get_logger().info(f"T_0S: {self.T_0S}")
                self.get_logger().info(f"H1: {self.H1}")
                self.get_logger().info(f"H2: {self.H2}")
        except Exception as e:
            self.get_logger().error(f"Failed to load transformations: {e}")
            raise

    def cam1_callback(self, msg):
        """Process point cloud from camera 1."""
        self.transform = self.T_0S @ self.H1
        self.process_point_cloud(msg, self.transform, "Camera 1")

    def cam2_callback(self, msg):
        """Process point cloud from camera 2."""
        self.transform = self.T_0S @ self.H2
        self.process_point_cloud(msg, self.transform, "Camera 2")

    def process_point_cloud(self, msg, transform, camera_name):
        """Process and transform a point cloud."""
        try:
            points = np.array([list(p)[:3] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
            if points.size == 0:
                self.get_logger().warn(f"{camera_name}: Received empty point cloud")
                return

            # Apply transformation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.transform(transform)

            # Optional: Downsample to reduce redundancy
            if self.voxel_size > 0:
                pcd = pcd.voxel_down_sample(self.voxel_size)

            # remove SOR outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)

            # Store processed point cloud based on camera name
            if camera_name == "Camera 1":
                self.processed_cloud1 = pcd
                self.cloud1_header = msg.header
                self.get_logger().info(f"Stored {len(np.asarray(pcd.points))} points from Camera 1")
            else:  # Camera 2
                self.processed_cloud2 = pcd
                self.cloud2_header = msg.header
                self.get_logger().info(f"Stored {len(np.asarray(pcd.points))} points from Camera 2")
                
            # Try to fuse if both clouds are available
            self.try_fusion()

        except Exception as e:
            self.get_logger().error(f"Error processing {camera_name} point cloud: {e}")

    def try_fusion(self):
        """Try to fuse point clouds if both are available."""
        if self.processed_cloud1 is not None and self.processed_cloud2 is not None:
            self.get_logger().info("Both point clouds available, performing fusion")
            
            # Fuse point clouds
            fused_points = self.fuse_point_clouds(self.processed_cloud1, self.processed_cloud2)

            if fused_points is not None:
                # Use header from most recent point cloud
                header_to_use = self.cloud1_header
                
                # Properly compare time objects by converting to total nanoseconds
                time1_ns = self.cloud1_header.stamp.sec * 1e9 + self.cloud1_header.stamp.nanosec
                time2_ns = self.cloud2_header.stamp.sec * 1e9 + self.cloud2_header.stamp.nanosec
                
                if time2_ns > time1_ns:
                    header_to_use = self.cloud2_header
                    
                # Filter outliers
                # fused_points = fused_points.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)

                # # crop point cloud in z direction a bit
                # filtered_points = np.asarray(fused_points.points)
                # filtered_points = filtered_points[filtered_points[:, 2] > 0.005]
                # fused_points.points = o3d.utility.Vector3dVector(filtered_points)
                # Crop the point clouds to the workspace
                
                if self.use_workspace_crop:
                    x_bounds_baseframe = (-0.25, 0.75)
                    y_bounds_baseframe = (-0.5, 1.75)
                    z_bounds_baseframe = (-0.05, 2)
                
                    filtered_points = np.asarray(fused_points.points)
                    filtered_points = filtered_points[
                        (filtered_points[:, 0] > x_bounds_baseframe[0]) & (filtered_points[:, 0] < x_bounds_baseframe[1]) &
                        (filtered_points[:, 1] > y_bounds_baseframe[0]) & (filtered_points[:, 1] < y_bounds_baseframe[1]) &
                        (filtered_points[:, 2] > z_bounds_baseframe[0]) & (filtered_points[:, 2] < z_bounds_baseframe[1])
                    ]
                    fused_points.points = o3d.utility.Vector3dVector(filtered_points
                )

                # Publish fused point cloud
                self.publish_point_cloud(fused_points, header_to_use)


                if self.visualize:
                    visualize_point_cloud(fused_points, f"Fused Point Cloud")
                


        else:
            # publish single transformed point cloud
            if self.processed_cloud1 is not None and self.transform_and_fuse_of_full_clouds == False:
                self.publish_point_cloud(self.processed_cloud1, self.cloud1_header)
            if self.processed_cloud2 is not None and self.transform_and_fuse_of_full_clouds == False:
                self.publish_point_cloud(self.processed_cloud2, self.cloud2_header)
            else:
                self.get_logger().info("Waiting for both point clouds to be available")
 


    def fuse_point_clouds(self, points1, points2):
        """Fuse point clouds based."""
        try:
            # Simple concatenation of point clouds
            combined_pcd = points1 + points2
            
            # Optional: Downsample to reduce redundancy
            if self.voxel_size > 0:
                combined_pcd = combined_pcd.voxel_down_sample(self.voxel_size)
            
            self.get_logger().info(f"Fused point cloud has {len(np.asarray(combined_pcd.points))} points")
            
            return combined_pcd
        except Exception as e:
            self.get_logger().error(f"Error fusing point clouds: {e}")
            return None

    def publish_point_cloud(self, points, header):
        """Publish a transformed point cloud."""
        try:
            header.frame_id = self.target_frame
            points_array = np.asarray(points.points)
            cloud_msg = pc2.create_cloud_xyz32(header, points_array.tolist())
            self.fused_publisher.publish(cloud_msg)
            self.get_logger().info("Published transformed point cloud")
        except Exception as e:
            self.get_logger().error(f"Error publishing point cloud: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTransformFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()