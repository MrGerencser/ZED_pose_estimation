import sys
import os

#!/usr/bin/env python3

import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from rclpy.executors import MultiThreadedExecutor
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from zed_pose_estimation.grasp_pose_utils import *
from zed_pose_estimation.vision_pipeline_utils import crop_point_cloud_gpu, fuse_point_clouds_centroid, subtract_point_clouds_gpu, convert_mask_to_3d_points, downsample_point_cloud_gpu
import open3d as o3d
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R_simple
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import traceback


from zed_pose_estimation.vis2 import visualize_superquadric_grasps

# Add superquadric fitting imports
from EMS.EMS_recovery import EMS_recovery
from zed_pose_estimation.superquadric_grasp_planner import SuperquadricGraspPlanner

class ZedGpuNode(Node):
    def __init__(self):
        super().__init__('zed_gpu_node')
        
        # Original parameters
        self.declare_parameter('camera1_sn', 33137761)
        self.declare_parameter('camera2_sn', 36829049)
        self.declare_parameter('yolo_model_path', '/home/chris/franka_ros2_ws/src/zed_pose_estimation/models/yolo/best.pt')
        self.declare_parameter('confidence_threshold', 0.1)
        self.declare_parameter('processing_rate', 10.0)  # Hz
        self.declare_parameter('voxel_size', 0.003)
        self.declare_parameter('distance_threshold', 0.3)
        self.declare_parameter('workspace_bounds', [-0.25, 0.75, -0.5, 0.5, -0.05, 2.0])
        self.declare_parameter('publish_visualization', False)
        self.declare_parameter('target_frame', 'panda_link0')
        self.declare_parameter('transform_file_path', os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'transform.yaml'))
        
        # Superquadric parameters
        self.declare_parameter('superquadric_enabled', True)
        self.declare_parameter('gripper_jaw_length', 0.254)  # meters
        self.declare_parameter('gripper_max_opening', 0.18)   # meters
        self.declare_parameter('outlier_ratio', 0.2)         # EMS parameter
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('visualize', False)
        
        
        # Get parameters
        self.camera1_sn = self.get_parameter('camera1_sn').get_parameter_value().integer_value
        self.camera2_sn = self.get_parameter('camera2_sn').get_parameter_value().integer_value
        self.model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.processing_rate = self.get_parameter('processing_rate').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        self.workspace_bounds = self.get_parameter('workspace_bounds').get_parameter_value().double_array_value
        self.publish_viz = self.get_parameter('publish_visualization').get_parameter_value().bool_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.transform_file_path = self.get_parameter('transform_file_path').get_parameter_value().string_value
        
        # Get superquadric parameters
        self.superquadric_enabled = self.get_parameter('superquadric_enabled').get_parameter_value().bool_value
        self.gripper_jaw_length = self.get_parameter('gripper_jaw_length').get_parameter_value().double_value
        self.gripper_max_opening = self.get_parameter('gripper_max_opening').get_parameter_value().double_value
        self.outlier_ratio = self.get_parameter('outlier_ratio').get_parameter_value().double_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        
        # Initialize superquadric grasp planner if enabled
        if self.superquadric_enabled:
            try:
                self.grasp_planner = SuperquadricGraspPlanner(
                    jaw_len=self.gripper_jaw_length,
                    max_open=self.gripper_max_opening
                )
                self.get_logger().info("Superquadric grasp planner initialized")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize superquadric grasp planner: {e}")
                self.superquadric_enabled = False
        else:
            self.grasp_planner = None
        
        self.future_shutdown = False

        self.class_names = {0: "Box", 1: "Cone", 2: "Cover", 3: "Plier",
                           4: "Robot", 5: "Screw Driver"}

        # Dictionary to store timings for benchmarking
        self.timings = {
            "Frame Retrieval": [],
            "Depth Retrieval": [],
            "Point Cloud Processing": [],
            "YOLO Inference": [],
            "Mask Processing": [],
            "Point Cloud Fusion": [],
            "Subtraction": [],
            "Superquadric Grasp Generation": [],
            "Total Time": []
        }
        
        # Setup visualization window if enabled
        if self.publish_viz:
            cv2.namedWindow("YOLO Detection")
        
        # FPS calculation
        self.fps_values = []
        
        # Check if CUDA is available and set the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        
        # Setup TF broadcaster and pose publisher
        self.pose_publisher = self.create_publisher(PoseStamped, '/perception/object_pose', 10)
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Initialize ZED cameras
        self.get_logger().info("Initializing ZED cameras...")
        self.init_zed_cameras()
        
        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(self.model_path).to(self.device)
        
        # Setup publishers
        self.fused_workspace_publisher = self.create_publisher(
            PointCloud2, '/perception/fused_workspace_cloud', 10)
        self.fused_objects_publisher = self.create_publisher(
            PointCloud2, '/perception/fused_objects_cloud', 10)
        self.subtracted_cloud_publisher = self.create_publisher(
            PointCloud2, '/perception/subtracted_cloud', 10)
        
        # Set up timer for processing
        self.timer = self.create_timer(1.0/self.processing_rate, self.process_frames)
        self.get_logger().info(f"ZED GPU Node initialized, processing at {self.processing_rate}Hz")

    def init_zed_cameras(self):
        """Initialize the ZED cameras with the SDK"""
        # Initialize the ZED camera objects
        self.zed1 = sl.Camera()
        self.zed2 = sl.Camera()
        
        # Set the initialization parameters for camera 1
        init_params1 = sl.InitParameters()
        init_params1.set_from_serial_number(self.camera1_sn)
        init_params1.camera_resolution = sl.RESOLUTION.HD720
        init_params1.camera_fps = 15
        init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params1.depth_minimum_distance = 0.4
        init_params1.coordinate_units = sl.UNIT.METER
        
        # Set the initialization parameters for camera 2
        init_params2 = sl.InitParameters()
        init_params2.set_from_serial_number(self.camera2_sn)
        init_params2.camera_resolution = sl.RESOLUTION.HD720
        init_params2.camera_fps = 15
        init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params2.depth_minimum_distance = 0.4
        init_params2.coordinate_units = sl.UNIT.METER
        
        # Open the cameras
        self.get_logger().info(f"Opening camera 1 (SN: {self.camera1_sn})...")
        err1 = self.zed1.open(init_params1)
        if err1 != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Error opening ZED camera 1: {err1}")
            raise RuntimeError(f"Failed to open ZED camera 1: {err1}")
            
        self.get_logger().info(f"Opening camera 2 (SN: {self.camera2_sn})...")
        err2 = self.zed2.open(init_params2)
        if err2 != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Error opening ZED camera 2: {err2}")
            self.zed1.close()  # Clean up first camera
            raise RuntimeError(f"Failed to open ZED camera 2: {err2}")
            
        # Get camera calibration parameters
        calib_params1 = self.zed1.get_camera_information().camera_configuration.calibration_parameters
        self.fx1, self.fy1 = calib_params1.left_cam.fx, calib_params1.left_cam.fy
        self.cx1, self.cy1 = calib_params1.left_cam.cx, calib_params1.left_cam.cy
        
        calib_params2 = self.zed2.get_camera_information().camera_configuration.calibration_parameters
        self.fx2, self.fy2 = calib_params2.left_cam.fx, calib_params2.left_cam.fy
        self.cx2, self.cy2 = calib_params2.left_cam.cx, calib_params2.left_cam.cy
        
        self.get_logger().info("Loading camera transformations...")
        self.load_transformations()
        
        # Initialize image and depth objects
        self.image1 = sl.Mat()
        self.depth1 = sl.Mat()
        self.image2 = sl.Mat()
        self.depth2 = sl.Mat()
        
        # Initialize point cloud objects
        resolution = sl.Resolution(640, 360)
        self.point_cloud1_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.point_cloud2_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        
        self.get_logger().info("ZED cameras initialized successfully")

    def load_transformations(self):
        """Load or define the transformation matrices"""
        try:
            with open(self.transform_file_path, 'r') as file:
                transforms = yaml.safe_load(file)
            
            T_chess_cam1_list = transforms['transforms']['T_chess_cam1']
            T_chess_cam2_list = transforms['transforms']['T_chess_cam2']
            T_robot_chess_list = transforms['transforms']['T_robot_chess']

            T_chess_cam1 = np.array(T_chess_cam1_list)
            T_chess_cam2 = np.array(T_chess_cam2_list)
            T_robot_chess = np.array(T_robot_chess_list)
            
            self.get_logger().info(f"Loaded transformations from {self.transform_file_path}")

        except FileNotFoundError:
            self.get_logger().error(f"Transform file not found at {self.transform_file_path}.")
        
        except KeyError as e:
            self.get_logger().error(f"Key error when parsing transform file: {e}.")

        except Exception as e:
            self.get_logger().error(f"Failed to load transformations from YAML: {e}.")



        # Calculate the transformation matrices from camera frames to the robot frame
        T_robot_cam1 = T_robot_chess @ T_chess_cam1
        T_robot_cam2 = T_robot_chess @ T_chess_cam2

        # Extract rotation and origin for camera 1, and convert to torch tensors
        self.rotation1 = T_robot_cam1[:3, :3]
        self.origin1 = T_robot_cam1[:3, 3]
        self.rotation1_torch = torch.tensor(self.rotation1, dtype=torch.float32, device=self.device)
        self.origin1_torch = torch.tensor(self.origin1, dtype=torch.float32, device=self.device)

        # Extract rotation and origin for camera 2, and convert to torch tensors
        self.rotation2 = T_robot_cam2[:3, :3]
        self.origin2 = T_robot_cam2[:3, 3]
        self.rotation2_torch = torch.tensor(self.rotation2, dtype=torch.float32, device=self.device)
        self.origin2_torch = torch.tensor(self.origin2, dtype=torch.float32, device=self.device)

        self.get_logger().info(f"Calculated T_robot_cam1 (Cam1 to Robot):\n{T_robot_cam1}")
        self.get_logger().info(f"Calculated T_robot_cam2 (Cam2 to Robot):\n{T_robot_cam2}")

    def update_camera_intrinsics(self):
        """Update camera intrinsics"""
        try:
            # Get fresh calibration parameters
            calib_params1 = self.zed1.get_camera_information().camera_configuration.calibration_parameters
            self.fx1, self.fy1 = calib_params1.left_cam.fx, calib_params1.left_cam.fy
            self.cx1, self.cy1 = calib_params1.left_cam.cx, calib_params1.left_cam.cy
            
            calib_params2 = self.zed2.get_camera_information().camera_configuration.calibration_parameters
            self.fx2, self.fy2 = calib_params2.left_cam.fx, calib_params2.left_cam.fy
            self.cx2, self.cy2 = calib_params2.left_cam.cx, calib_params2.left_cam.cy
            
            self.get_logger().info("Camera intrinsics updated")
            
        except Exception as e:
            self.get_logger().error(f"Failed to update intrinsics: {e}")
    
    def process_frames(self):
        """Main processing loop - timer callback"""
        try:
            start_time = time.time()
            
            # Check if cameras are ready
            if (self.zed1.grab() != sl.ERROR_CODE.SUCCESS or 
                self.zed2.grab() != sl.ERROR_CODE.SUCCESS):
                self.get_logger().warn("Failed to grab frames from cameras")
                return
            
            # Update intrinsics every 100 frames
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 0
                
            if self.frame_count % 100 == 0:
                self.update_camera_intrinsics()
                
            # Step 1: Frame retrieval
            retrieval_start = time.time()
            self.zed1.retrieve_image(self.image1, view=sl.VIEW.LEFT)
            self.zed2.retrieve_image(self.image2, view=sl.VIEW.LEFT)
            frame1 = self.image1.get_data()
            frame2 = self.image2.get_data()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)
            retrieval_time = time.time() - retrieval_start
            self.timings["Frame Retrieval"].append(retrieval_time)
            
            # Step 2: Depth retrieval
            depth_start = time.time()
            depth_result1 = self.zed1.retrieve_measure(self.depth1, measure=sl.MEASURE.DEPTH)
            depth_result2 = self.zed2.retrieve_measure(self.depth2, measure=sl.MEASURE.DEPTH)
            if depth_result1 != sl.ERROR_CODE.SUCCESS or depth_result2 != sl.ERROR_CODE.SUCCESS:
                self.get_logger().warn(f"Failed to retrieve depth maps: {depth_result1}, {depth_result2}")
                return
            depth_np1 = self.depth1.get_data()
            depth_np2 = self.depth2.get_data()
            depth_time = time.time() - depth_start
            self.timings["Depth Retrieval"].append(depth_time)
            
            # Step 3: Point cloud processing
            pc_start = time.time()
            self.zed1.retrieve_measure(self.point_cloud1_ws, measure=sl.MEASURE.XYZ)
            self.zed2.retrieve_measure(self.point_cloud2_ws, measure=sl.MEASURE.XYZ)
            
            pc1_tensor = torch.tensor(self.point_cloud1_ws.get_data()[:, :, :3], 
                                     dtype=torch.float32, device=self.device).reshape(-1, 3)
            pc2_tensor = torch.tensor(self.point_cloud2_ws.get_data()[:, :, :3], 
                                     dtype=torch.float32, device=self.device).reshape(-1, 3)
            
            valid_mask1 = torch.isfinite(pc1_tensor).all(dim=1)
            valid_mask2 = torch.isfinite(pc2_tensor).all(dim=1)
            pc1_tensor = pc1_tensor[valid_mask1]
            pc2_tensor = pc2_tensor[valid_mask2]
            
            # Transform point clouds to robot frame
            pc1_transformed = torch.mm(pc1_tensor, self.rotation1_torch.T) + self.origin1_torch
            pc2_transformed = torch.mm(pc2_tensor, self.rotation2_torch.T) + self.origin2_torch
            
            # Crop to workspace
            x_bounds = (self.workspace_bounds[0], self.workspace_bounds[1])
            y_bounds = (self.workspace_bounds[2], self.workspace_bounds[3])
            z_bounds = (self.workspace_bounds[4], self.workspace_bounds[5])
            
            pc1_cropped = crop_point_cloud_gpu(pc1_transformed, x_bounds, y_bounds, z_bounds)
            pc2_cropped = crop_point_cloud_gpu(pc2_transformed, x_bounds, y_bounds, z_bounds)
            
            # Fuse workspace point clouds
            fused_workspace = torch.cat((pc1_cropped, pc2_cropped), dim=0)
            fused_workspace_np = fused_workspace.cpu().numpy()
            pcd_fused_workspace = o3d.geometry.PointCloud()
            pcd_fused_workspace.points = o3d.utility.Vector3dVector(fused_workspace_np)
            # pcd_fused_workspace = self.preprocess_point_cloud(pcd_fused_workspace)
            
            #visualize fused workspace
            if self.visualize:
                o3d.visualization.draw_geometries([pcd_fused_workspace], window_name="Fused Workspace Point Cloud")
                
            
            # Save the fused workspace point cloud
            fused_workspace_xyz = o3d.geometry.PointCloud()
            fused_workspace_xyz.points = pcd_fused_workspace.points
            # o3d.io.write_point_cloud("pointclouds/fused_workspace_xyz.ply", fused_workspace_xyz)
            
            pc_time = time.time() - pc_start
            self.timings["Point Cloud Processing"].append(pc_time)
            
            # Step 4: YOLO inference
            yolo_start = time.time()
            frame_batch = [frame1, frame2]
            
            results_batch = self.model.track(
                source=frame_batch,
                classes=[1],  # Cone
                persist=True,
                retina_masks=True,
                conf=self.conf_threshold,
                device=self.device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )
            
            results1 = results_batch[0]
            results2 = results_batch[1]
            
            masks_obj1 = results1.masks 
            masks_obj2 = results2.masks
            boxes_obj1 = results1.boxes
            boxes_obj2 = results2.boxes
            
            class_ids1 = boxes_obj1.cls.cpu().numpy() if boxes_obj1 is not None else np.array([])
            class_ids2 = boxes_obj2.cls.cpu().numpy() if boxes_obj2 is not None else np.array([])
            
            yolo_time = time.time() - yolo_start
            self.timings["YOLO Inference"].append(yolo_time)
            
            # Step 5: Process masks to get object point clouds
            mask_start = time.time()
            point_clouds_camera1 = []
            point_clouds_camera2 = []
            
            # Process camera 1 masks
            if masks_obj1 is not None and masks_obj1.data.numel() > 0:
                depth_map1_torch = torch.tensor(depth_np1, dtype=torch.float32, device=self.device)
                for i, individual_mask_tensor in enumerate(masks_obj1.data):
                    mask_indices_full = torch.nonzero(individual_mask_tensor, as_tuple=False)
                    mask_indices = mask_indices_full

                    if mask_indices.numel() > 0: 
                        with torch.amp.autocast('cuda'):
                            points_3d = convert_mask_to_3d_points(
                                mask_indices, depth_map1_torch, self.cx1, self.cy1, self.fx1, self.fy1
                            )
                        
                        if points_3d.size(0) > 0:
                            transformed = torch.mm(points_3d, self.rotation1_torch.T) + self.origin1_torch
                            point_clouds_camera1.append((transformed.cpu().numpy(), int(class_ids1[i])))
            
            # Process camera 2 masks
            if masks_obj2 is not None and masks_obj2.data.numel() > 0:
                depth_map2_torch = torch.tensor(depth_np2, dtype=torch.float32, device=self.device)
                for i, individual_mask_tensor in enumerate(masks_obj2.data):
                    mask_indices_full = torch.nonzero(individual_mask_tensor, as_tuple=False)
                    mask_indices = mask_indices_full

                    if mask_indices.numel() > 0: 
                        with torch.amp.autocast('cuda'):
                            points_3d = convert_mask_to_3d_points(
                                mask_indices, depth_map2_torch, self.cx2, self.cy2, self.fx2, self.fy2
                            )
                        
                        if points_3d.size(0) > 0:
                            transformed = torch.mm(points_3d, self.rotation2_torch.T) + self.origin2_torch
                            point_clouds_camera2.append((transformed.cpu().numpy(), int(class_ids2[i])))
            
            mask_time = time.time() - mask_start
            self.timings["Mask Processing"].append(mask_time)
            
            # Step 6: Fuse object point clouds
            fusion_start = time.time()
            _, _, fused_objects = fuse_point_clouds_centroid(
                point_clouds_camera1, point_clouds_camera2, self.distance_threshold
            )
            
            fused_object_points = [pc for pc, _ in fused_objects]
            fused_object_classes = [cls for _, cls in fused_objects]
            
            if fused_object_points:
                fused_objects_np = np.vstack(fused_object_points)
            else:
                fused_objects_np = np.empty((0, 3))
                
            fusion_time = time.time() - fusion_start
            self.timings["Point Cloud Fusion"].append(fusion_time)
            
            # Step 7: Subtract objects from workspace
            subtraction_start = time.time()
            subtracted_cloud = subtract_point_clouds_gpu(
                fused_workspace_np, fused_objects_np, distance_threshold=0.06
            )
            subtraction_time = time.time() - subtraction_start
            self.timings["Subtraction"].append(subtraction_time)
            
            # Step 8: Generate grasps using superquadric fitting
            if self.superquadric_enabled and fused_object_points and len(fused_object_points) > 0:
                grasp_start = time.time()
                
                for i, (object_points, class_id) in enumerate(zip(fused_object_points, fused_object_classes)):
                    if len(object_points) < 100:
                        self.get_logger().warn(f"Object {i} has too few points ({len(object_points)}) for superquadric fitting")
                        continue
                    
                    if class_id in [1]:  # Cone
                        self.get_logger().info(f"Processing {self.class_names.get(class_id, 'Unknown')} with superquadric fitting")
                        
                        try:
                            result = self.generate_superquadric_grasps(
                                object_points, 
                                fused_workspace_np,
                                class_id
                            )
                            
                            # Handle the result properly
                            if result and len(result) == 2:
                                grasp_poses, sq_recovered = result
                            else:
                                grasp_poses, sq_recovered = [], None
                            
                            if grasp_poses and len(grasp_poses) > 0:
                                # Get the best grasp pose (already in robot frame, already EE position)
                                best_grasp = grasp_poses[0]
                                
                                print(f"Best grasp for object {i} (class {class_id}): {best_grasp}")
                                
                                # Extract position and quaternion from the 4x4 transformation matrix
                                pos = best_grasp[:3, 3]  # Extract translation
                                rot_matrix = best_grasp[:3, :3]  # Extract rotation matrix
                                
                                # Convert rotation matrix to quaternion
                                from scipy.spatial.transform import Rotation as R_scipy
                                quat = R_scipy.from_matrix(rot_matrix).as_quat()  # Returns [x, y, z, w]
                                euler = R_scipy.from_matrix(rot_matrix).as_euler('xyz')
                                
                                print(f"Best grasp position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                                print(f"Best grasp euler angles: [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]")
                                print(f"Best grasp quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                                
                                franka_home_euler = np.array([np.pi, 0.0, 0.0])  # Home position euler angles
                                euler_diff = euler - franka_home_euler

                                franka_home_rotation = R_scipy.from_euler('xyz', franka_home_euler)
                                
                                # convert the best grasp rotation to the ee frame
                                best_grasp_rotation = R_scipy.from_matrix(rot_matrix)
                                best_grasp_rotation_ee = franka_home_rotation.inv() * best_grasp_rotation

                                print(f"Best grasp rotation in EE frame: {best_grasp_rotation_ee.as_euler('xyz')}")
                                
                                quat_ee = best_grasp_rotation_ee.as_quat()  # Convert to quaternion in EE frame

                                # Get the consistent_euler from sq_recovered
                                consistent_euler = R_scipy.from_matrix(sq_recovered.RotM).as_euler('xyz')

                                # Get points_for_ems from the object_points (same data that was used for fitting)
                                points_for_ems = object_points  # object point cloud data

                                if self.visualize:
                                    visualize_superquadric_grasps(
                                                point_cloud_data=points_for_ems,
                                                superquadric_params={
                                                    'shape': sq_recovered.shape,
                                                    'scale': sq_recovered.scale,
                                                    'euler': consistent_euler,
                                                    'translation': sq_recovered.translation
                                                },
                                                grasp_poses=[best_grasp],  # ← Final corrected pose
                                                show_sweep_volume=False,
                                                window_name="FINAL Published Grasp Pose (After Corrections)",
                                            )

                                
                                # Create and publish pose message
                                pose_msg = PoseStamped()
                                pose_msg.header = Header()
                                pose_msg.header.stamp = self.get_clock().now().to_msg()
                                pose_msg.header.frame_id = self.target_frame
                                pose_msg.pose.position.x = float(pos[0])
                                pose_msg.pose.position.y = float(pos[1])
                                pose_msg.pose.position.z = float(pos[2]) + 0.02  # Add small offset above object
                                pose_msg.pose.orientation.x = float(quat_ee[0])
                                pose_msg.pose.orientation.y = float(quat_ee[1])
                                pose_msg.pose.orientation.z = float(quat_ee[2])
                                pose_msg.pose.orientation.w = float(quat_ee[3])

                                # Publish the grasp pose
                                self.pose_publisher.publish(pose_msg)
                                self.get_logger().info(f"Published grasp pose at: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                                
                                # Optionally publish TF transform as well
                                if self.publish_tf:
                                    self.publish_transform(
                                        position=pos,
                                        quaternion=quat,
                                        parent_frame=self.target_frame,
                                        child_frame=f"object_{class_id}_grasp"
                                    )
                                    
                        
                                    
                        except Exception as e:
                            self.get_logger().error(f"Error in superquadric grasp generation for object {i}: {e}")
                            self.get_logger().error(traceback.format_exc())
                
                grasp_time = time.time() - grasp_start
                self.timings["Superquadric Grasp Generation"].append(grasp_time)
            
            # Create header for point cloud messages
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.target_frame
            
            # Publish the point clouds
            if fused_workspace_np.size > 0:
                self.fused_workspace_publisher.publish(pc2.create_cloud_xyz32(header, fused_workspace_np))
                
            if fused_objects_np.size > 0:
                self.fused_objects_publisher.publish(pc2.create_cloud_xyz32(header, fused_objects_np))
                
            if subtracted_cloud.size > 0:
                self.subtracted_cloud_publisher.publish(pc2.create_cloud_xyz32(header, subtracted_cloud))
            
            # Calculate and update FPS
            total_time = time.time() - start_time
            self.timings["Total Time"].append(total_time)
            fps = 1.0 / total_time
            self.fps_values.append(fps)
            
            if len(self.fps_values) > 10:
                self.fps_values.pop(0)
            avg_fps = sum(self.fps_values) / len(self.fps_values)
            
            # Visualization using OpenCV
            if self.publish_viz:
                try:
                    display1 = frame1.copy()
                    display2 = frame2.copy()
                    
                    if results1[0].boxes is not None and len(results1[0].boxes) > 0:
                        for box in results1[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(display1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if results2[0].boxes is not None and len(results2[0].boxes) > 0:
                        for box in results2[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(display2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cv2.putText(display1, f"FPS: {avg_fps:.1f}", (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    display1 = cv2.resize(display1, (640, 360))
                    display2 = cv2.resize(display2, (640, 360))
                    
                    display = np.hconcat([display1, display2])
                    
                    cv2.imshow("YOLO Detection", display)
                    key = cv2.waitKey(1)
                    
                    if key == ord('q'):
                        self.get_logger().info("User requested shutdown (Q key pressed)")
                        self.future_shutdown = True
                        
                except Exception as viz_error:
                    self.get_logger().error(f"Visualization error: {viz_error}")
                    self.get_logger().error(traceback.format_exc())
            
            point_clouds_camera1.clear()
            point_clouds_camera2.clear()
            
        except Exception as e:
            self.get_logger().error(f"Error in process_frames: {e}")
            self.get_logger().error(traceback.format_exc())

    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud for superquadric fitting"""
        try:
            pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            if len(np.asarray(pcd_down.points)) > 50:
                pcd_down, _ = pcd_down.remove_statistical_outlier(
                    nb_neighbors=20,
                    std_ratio=2.0
                )

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
            return pcd

    def calculate_k_superquadrics(self, point_count):
        """
        Calculate the number of superquadrics K based on point cloud size
        Following the paper's equation (12):
        
        K = { 6                            if |X| < 8000
            { 8 + 2 × ⌊(|X| - 8000)/4000⌋  if |X| ≥ 8000
        
        The more points an object contains, the larger it will be and we assume 
        the more hidden superquadrics are inside the object.
        """
        X_size = int(point_count)  # |X| = number of points
        
        if X_size < 8000:
            K = 1
            self.get_logger().info(f"Point count: {X_size} < 8000, using K = 6")
        else:
            # K = 8 + 2 × ⌊(|X| - 8000) / 4000⌋
            # Floor division in Python is //
            K = 8 + 2 * ((X_size - 8000) // 4000)
            self.get_logger().info(f"Point count: {X_size} ≥ 8000, using K = 8 + 2 × ⌊({X_size} - 8000)/4000⌋ = {K}")
        
        # Optional: Add reasonable upper bound to prevent excessive computation
        K_max = 20  # Practical limit
        if K > K_max:
            self.get_logger().warn(f"Calculated K = {K} exceeds maximum {K_max}, capping to {K_max}")
            K = K_max
        
        self.get_logger().info(f"Final K value: {K} superquadrics for {X_size} points")
        return K

    def initialize_multiple_superquadrics(self, points_for_ems, K=None):
        """
        Initialize K superquadrics at different positions as described in the paper
        Section 3: "6+1 superquadrics at different positions are initialized"
        
        Args:
            points_for_ems: Point cloud data
            K: Number of superquadrics (if None, calculate using paper's formula)
        """
        
        # Calculate K using paper's formula if not provided
        if K is None:
            K = self.calculate_k_superquadrics(len(points_for_ems))
        
        # Calculate point cloud properties
        points_center = np.mean(points_for_ems, axis=0)
        points_bounds = np.max(points_for_ems, axis=0) - np.min(points_for_ems, axis=0)
        
        self.get_logger().info(f"Initializing {K} superquadrics for {len(points_for_ems)} points")
        self.get_logger().info(f"Object center: {points_center}")
        self.get_logger().info(f"Object bounds: {points_bounds}")
        
        # Calculate initial positions for K superquadrics
        initial_positions = []
        
        # Multiple initialization strategies based on K
        
        if K == 6:
            # For smaller objects (< 8000 points): Use 3D grid pattern
            # Place 6 superquadrics in a 2×3 grid pattern around the object
            offsets = [
                [-0.3, -0.2, 0.0],   # Left-back
                [-0.3,  0.2, 0.0],   # Left-front  
                [ 0.0, -0.2, 0.0],   # Center-back
                [ 0.0,  0.2, 0.0],   # Center-front
                [ 0.3, -0.2, 0.0],   # Right-back
                [ 0.3,  0.2, 0.0]    # Right-front
            ]
            
            for i, offset in enumerate(offsets):
                # Scale offset by object bounds
                scaled_offset = np.array(offset) * points_bounds * 0.4
                init_position = points_center + scaled_offset
                initial_positions.append(init_position)
                self.get_logger().info(f"  SQ {i+1}: {init_position} (offset: {offset})")
        
        else:
            # For larger objects (≥ 8000 points): Use more sophisticated grid
            # Calculate grid dimensions based on K
            if K <= 8:
                grid_dims = (2, 2, 2)  # 2×2×2 = 8 positions max
            elif K <= 27:
                grid_dims = (3, 3, 3)  # 3×3×3 = 27 positions max
            else:
                # For very large K, use 4×4×4 or larger
                dim = int(np.ceil(K ** (1/3)))
                grid_dims = (dim, dim, dim)
            
            self.get_logger().info(f"Using {grid_dims[0]}×{grid_dims[1]}×{grid_dims[2]} grid for {K} superquadrics")
            
            # Generate grid positions
            positions_generated = 0
            for x_idx in range(grid_dims[0]):
                for y_idx in range(grid_dims[1]):
                    for z_idx in range(grid_dims[2]):
                        if positions_generated >= K:
                            break
                        
                        # Calculate normalized grid position [-1, 1]
                        x_norm = (2 * x_idx / (grid_dims[0] - 1) - 1) if grid_dims[0] > 1 else 0
                        y_norm = (2 * y_idx / (grid_dims[1] - 1) - 1) if grid_dims[1] > 1 else 0  
                        z_norm = (2 * z_idx / (grid_dims[2] - 1) - 1) if grid_dims[2] > 1 else 0
                        
                        # Scale by object bounds with some margin
                        offset_scale = 0.3  # 30% of object size as maximum offset
                        x_offset = x_norm * points_bounds[0] * offset_scale
                        y_offset = y_norm * points_bounds[1] * offset_scale
                        z_offset = z_norm * points_bounds[2] * offset_scale
                        
                        init_position = points_center + np.array([x_offset, y_offset, z_offset])
                        initial_positions.append(init_position)
                        
                        self.get_logger().info(f"  SQ {positions_generated+1}: {init_position} "
                                            f"(grid: {x_idx},{y_idx},{z_idx})")
                        
                        positions_generated += 1
                        
                    if positions_generated >= K:
                        break
                if positions_generated >= K:
                    break
        
        # Add one additional superquadric at the exact center (the "+1" in "6+1")
        if len(initial_positions) < K:
            initial_positions.append(points_center.copy())
            self.get_logger().info(f"  SQ {len(initial_positions)}: {points_center} (center)")
        
        self.get_logger().info(f"Generated {len(initial_positions)} initial positions for K={K}")
        return initial_positions[:K]  # Ensure we return exactly K positions

    def fit_multiple_superquadrics_ensemble(self, points_for_ems, K=None):
        """
        Fit multiple superquadrics using ensemble approach
        """
        
        # Calculate K using the paper's formula
        if K is None:
            K = self.calculate_k_superquadrics(len(points_for_ems))

        self.get_logger().info(f"Fitting {K} superquadrics using ensemble approach")

        initial_positions = self.initialize_multiple_superquadrics(points_for_ems, K)
        
        fitted_superquadrics = []
        fitting_scores = []
        
        # Initial scale estimate based on point cloud
        points_bounds = np.max(points_for_ems, axis=0) - np.min(points_for_ems, axis=0)
        initial_scale = points_bounds / (2.0 + K * 0.1)  # Smaller scales for more superquadrics
        
        self.get_logger().info(f"Initial scale estimate: {initial_scale}")
        
        for i, init_pos in enumerate(initial_positions):
            try:
                self.get_logger().info(f"Fitting superquadric {i+1}/{K} at position {init_pos}")

                # ADAPTIVE PARAMETERS: Vary parameters to encourage diversity
                parameter_variations = [
                    # Each SQ gets different parameters to explore solution space
                    {'outlier_ratio': 0.1 + (i % 4) * 0.1, 'tolerance': 1e-3, 'sigma': 0.05},
                    {'outlier_ratio': 0.2 + (i % 3) * 0.1, 'tolerance': 1e-2, 'sigma': 0.08},
                    {'outlier_ratio': 0.15 + (i % 5) * 0.05, 'tolerance': 5e-3, 'sigma': 0.06},
                ]
                
                params = parameter_variations[i % len(parameter_variations)]
                
                #   CRITICAL: Pre-translate points to superquadric-centered coordinates
                translated_points = points_for_ems - init_pos
                
                # Fit superquadric to translated points
                sq_candidate, probabilities = EMS_recovery(
                    translated_points,  # Points centered around init_pos
                    OutlierRatio=params['outlier_ratio'],
                    MaxIterationEM=20 + (i % 3) * 5,  # Vary iterations
                    ToleranceEM=params['tolerance'],
                    RelativeToleranceEM=1e-1,
                    MaxOptiIterations=2 + (i % 2),  # Vary optimization iterations
                    Sigma=params['sigma'],
                    MaxiSwitch=1 + (i % 3),  # Vary switches
                    AdaptiveUpperBound=True,
                    Rescale=False
                )
                
                #   CRITICAL: Translate superquadric back to world coordinates
                sq_candidate.translation = sq_candidate.translation + init_pos
                
                # Calculate fitting quality score based on paper's criteria
                explained_points = np.sum(probabilities > 0.5)  # Points well-explained
                total_points = len(points_for_ems)
                coverage_score = explained_points / total_points
                
                # Compactness score (prefer reasonable sizes relative to object)
                sq_volume = np.prod(sq_candidate.scale)
                object_volume_est = np.prod(points_bounds) * (0.2 / K)  # Each SQ should cover 1/K of object
                compactness_score = min(1.0, object_volume_est / sq_volume) if sq_volume > 0 else 0.0
                
                # Shape regularity score (prefer reasonable shape parameters)
                shape_regularity = 1.0 / (1.0 + abs(sq_candidate.shape[0] - 1.0) + abs(sq_candidate.shape[1] - 1.0))
                
                # Position accuracy (prefer SQs that stay near their initialization)
                position_drift = np.linalg.norm(sq_candidate.translation - init_pos)
                max_drift = np.linalg.norm(points_bounds) * 0.5
                position_score = max(0.0, 1.0 - position_drift / max_drift)
                
                # Combined score following paper's multi-SQ objectives
                total_score = (coverage_score * 0.4 +       # Primary: how well does it explain points
                            compactness_score * 0.3 +     # Secondary: reasonable size
                            shape_regularity * 0.2 +      # Tertiary: regular shape
                            position_score * 0.1)         # Quaternary: position stability
                
                fitted_superquadrics.append(sq_candidate)
                fitting_scores.append(total_score)
                
                self.get_logger().info(f"  SQ {i+1}: Coverage={coverage_score:.3f}, "
                                    f"Compact={compactness_score:.3f}, "
                                    f"Regular={shape_regularity:.3f}, "
                                    f"Position={position_score:.3f}, "
                                    f"Total={total_score:.3f}")
                
            except Exception as e:
                self.get_logger().warn(f"Failed to fit superquadric {i+1}: {e}")
                fitting_scores.append(0.0)
                fitted_superquadrics.append(None)
        
        # Filter out failed fits
        valid_fits = [(sq, score) for sq, score in zip(fitted_superquadrics, fitting_scores) 
                    if sq is not None and score > 0.1]
        
        if not valid_fits:
            self.get_logger().error(f"All {K} multi-superquadric fits failed")
            return None, []
        
        # Sort by score (best first)
        valid_fits.sort(key=lambda x: x[1], reverse=True)
        
        self.get_logger().info(f"Multi-superquadric fitting results (K={K}):")
        for i, (sq, score) in enumerate(valid_fits[:min(5, len(valid_fits))]):  # Show top 5
            self.get_logger().info(f"  Rank {i+1}: Score={score:.3f}, "
                                f"Scale={sq.scale}, Center={sq.translation}")
        
        # Return best superquadric and all valid ones for comparison
        best_sq = valid_fits[0][0]
        all_valid_sqs = [sq for sq, score in valid_fits]
        
        self.get_logger().info(f"Selected best superquadric from ensemble of {len(all_valid_sqs)} valid fits")
        return best_sq, all_valid_sqs

    def hierarchical_ems(
        self,
        point,
        OutlierRatio=0.9,           # prior outlier probability [0, 1) (default: 0.1)
        MaxIterationEM=20,           # maximum number of EM iterations (default: 20)
        ToleranceEM=1e-3,            # absolute tolerance of EM (default: 1e-3)
        RelativeToleranceEM=2e-1,    # relative tolerance of EM (default: 1e-1)
        MaxOptiIterations=2,         # maximum number of optimization iterations per M (default: 2)
        Sigma=0.3,                   # initial sigma^2 (default: 0 - auto generate)
        MaxiSwitch=2,                # maximum number of switches allowed (default: 2)
        AdaptiveUpperBound=True,    # Introduce adaptive upper bound to restrict the volume of SQ (default: false)
        Rescale=False,                # normalize the input point cloud (default: true)
        MaxLayer=5,                  # maximum depth
        Eps=1.7,                    # IMPORTANT: varies based on the size of the input pointcloud (DBScan parameter)
        MinPoints=60,               # DBScan parameter required minimum points
    ):
        """
        Hierarchical EMS for multiple superquadrics recovery
        """
        from sklearn.cluster import DBSCAN
        
        point_seg = {key: [] for key in list(range(0, MaxLayer+1))}
        point_outlier = {key: [] for key in list(range(0, MaxLayer+1))}
        point_seg[0] = [point]
        list_quadrics = []
        quadric_count = 1
        
        for h in range(MaxLayer):
            for c in range(len(point_seg[h])):
                if len(point_seg[h][c]) < MinPoints:  # Skip if too few points
                    continue
                    
                self.get_logger().info(f"Processing quadric {quadric_count} at layer {h}/{MaxLayer}")
                quadric_count += 1
                
                # Run EMS on current point segment
                x_raw, p_raw = EMS_recovery(
                    point_seg[h][c],
                    OutlierRatio,
                    MaxIterationEM,
                    ToleranceEM,
                    RelativeToleranceEM,
                    MaxOptiIterations,
                    Sigma,
                    MaxiSwitch,
                    AdaptiveUpperBound,
                    Rescale,
                )
                
                point_previous = point_seg[h][c]
                list_quadrics.append(x_raw)
                outlier = point_seg[h][c][p_raw < 0.1, :]
                point_seg[h][c] = point_seg[h][c][p_raw > 0.1, :]
                
                # Check if we need to continue decomposition
                inlier_sum = np.sum(p_raw > 0.1)
                inlier_ratio = inlier_sum / len(point_previous)
                self.get_logger().info(f"Layer {h}, Quadric {quadric_count-1}: Inlier ratio = {inlier_ratio:.3f}")
                
                # Continue decomposition if inlier ratio is low and we have enough outliers
                if inlier_sum < (0.8 * len(point_previous)) and len(outlier) > MinPoints and h < MaxLayer - 1:
                    # Cluster outliers using DBSCAN
                    clustering = DBSCAN(eps=Eps, min_samples=MinPoints).fit(outlier)
                    labels = list(set(clustering.labels_))
                    labels = [item for item in labels if item >= 0]
                    
                    self.get_logger().info(f"DBSCAN found {len(labels)} clusters in outliers")
                    
                    if len(labels) >= 1:
                        for i in range(len(labels)):
                            cluster_points = outlier[clustering.labels_ == i]
                            if len(cluster_points) >= MinPoints:
                                point_seg[h + 1].append(cluster_points)
                                self.get_logger().info(f"Added cluster {i} with {len(cluster_points)} points to layer {h+1}")
                        point_outlier[h].append(outlier[clustering.labels_ == -1])
                    else:
                        point_outlier[h].append(outlier)
                else:
                    point_outlier[h].append(outlier)
                    if inlier_ratio >= 0.8:
                        self.get_logger().info(f"Good fit achieved (inlier ratio: {inlier_ratio:.3f}), stopping decomposition")
        
        self.get_logger().info(f"Hierarchical EMS completed: {len(list_quadrics)} superquadrics found")
        return point_seg, point_outlier, list_quadrics

    def adaptive_hierarchical_ems(self, point, OutlierRatio=0.9, MaxIterationEM=20, ToleranceEM=1e-3, 
                                RelativeToleranceEM=2e-1, MaxOptiIterations=2, Sigma=0.3, MaxiSwitch=2,
                                AdaptiveUpperBound=True, Rescale=False):
        """
        Adaptive hierarchical EMS that calculates parameters based on point cloud size
        """
        try:
            # Calculate adaptive parameters based on point cloud size
            point_count = len(point)
            
            #   ADAPTIVE PARAMETERS: Adjust based on point cloud density
            if point_count < 1000:
                MaxLayer = 3
                Eps = 0.01  # Smaller epsilon for smaller, denser objects
                MinPoints = 30
            elif point_count < 3000:
                MaxLayer = 4
                Eps = 0.015
                MinPoints = 50
            elif point_count < 8000:
                MaxLayer = 5
                Eps = 0.02
                MinPoints = 60
            else:
                MaxLayer = 6
                Eps = 0.025  # Larger epsilon for bigger, more spread out objects
                MinPoints = 80
            
            self.get_logger().info(f"Adaptive parameters for {point_count} points:")
            self.get_logger().info(f"  MaxLayer: {MaxLayer}")
            self.get_logger().info(f"  DBSCAN Eps: {Eps}")
            self.get_logger().info(f"  MinPoints: {MinPoints}")
            
            # Call the hierarchical EMS with adaptive parameters
            return self.hierarchical_ems(
                point=point,
                OutlierRatio=OutlierRatio,
                MaxIterationEM=MaxIterationEM,
                ToleranceEM=ToleranceEM,
                RelativeToleranceEM=RelativeToleranceEM,
                MaxOptiIterations=MaxOptiIterations,
                Sigma=Sigma,
                MaxiSwitch=MaxiSwitch,
                AdaptiveUpperBound=AdaptiveUpperBound,
                Rescale=Rescale,
                MaxLayer=MaxLayer,
                Eps=Eps,
                MinPoints=MinPoints
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in adaptive hierarchical EMS: {e}")
            return None, None, []

    def visualize_hierarchical_multiquadric_fit(self, object_points, list_quadrics):
        """Enhanced visualization for hierarchical multiquadric fitting"""
        try:
            import random
            
            # Create point cloud from original points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for original points
            
            # Create list of geometries to visualize
            geometries = [pcd]
            
            # Generate distinct colors for each quadric
            colors = []
            for i in range(len(list_quadrics)):
                # Use HSV color space for better color distribution
                hue = (i * 137.5) % 360  # Golden angle for good distribution
                saturation = 0.8
                value = 0.9
                # Convert HSV to RGB
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
                colors.append(list(rgb))
            
            # Add each superquadric mesh with different colors
            successful_quadrics = 0
            for i, quadric in enumerate(list_quadrics):
                try:
                    sq_mesh = self.superquadric_to_open3d_mesh(quadric, arclength=0.15)
                    if sq_mesh is not None:
                        sq_mesh.paint_uniform_color(colors[i])
                        geometries.append(sq_mesh)
                        successful_quadrics += 1
                        
                        # Log quadric parameters
                        self.get_logger().info(f"Quadric {i+1}: shape=({quadric.shape[0]:.3f}, {quadric.shape[1]:.3f}), "
                                            f"scale=({quadric.scale[0]:.3f}, {quadric.scale[1]:.3f}, {quadric.scale[2]:.3f})")
                except Exception as mesh_error:
                    self.get_logger().warn(f"Failed to create mesh for quadric {i+1}: {mesh_error}")
            
            # Save visualization files
            timestamp = int(time.time())
            os.makedirs("pointclouds", exist_ok=True)
            # o3d.io.write_point_cloud(f"pointclouds/adaptive_multiquadric_original_{timestamp}.ply", pcd)
            
            # for i, geom in enumerate(geometries[1:]):  # Skip original point cloud
            #     if hasattr(geom, 'triangles'):  # Check if it's a mesh
            #         o3d.io.write_triangle_mesh(f"pointclouds/adaptive_sq_{i+1}_{timestamp}.ply", geom)
            
            self.get_logger().info(f"Adaptive multiquadric fitting: {successful_quadrics}/{len(list_quadrics)} successful meshes")
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geometries.append(coord_frame)
            
            # Visualize all geometries together
            if self.visualize:
                o3d.visualization.draw_geometries(
                    geometries,
                    window_name=f"Adaptive Multi-Superquadric Fitting (K={len(list_quadrics)})",
                    zoom=0.7,
                    front=[0, -1, 0],
                    lookat=pcd.get_center(),
                    up=[0, 0, 1]
                )
            
            return successful_quadrics
            
        except Exception as e:
            self.get_logger().error(f"Error in adaptive multiquadric visualization: {e}")
            self.get_logger().error(traceback.format_exc())
            return 0
    
    def superquadric_to_open3d_mesh(self, x, threshold=1e-2, num_limit=10000, arclength=0.2):
        """Convert superquadric to Open3D mesh (same as multiquadric_test.py)"""
        try:
            from EMS.utilities import uniformSampledSuperellipse, create_mesh_from_grid
            
            # avoid numerical instability in sampling
            if x.shape[0] < 0.007:
                x.shape[0] = 0.007
            if x.shape[1] < 0.007:
                x.shape[1] = 0.007
            
            # sampling points in superellipse    
            point_eta = uniformSampledSuperellipse(x.shape[0], [1, x.scale[2]], threshold, num_limit, arclength)
            point_omega = uniformSampledSuperellipse(x.shape[1], [x.scale[0], x.scale[1]], threshold, num_limit, arclength)
            
            # preallocate meshgrid
            x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
            y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
            z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))

            for m in range(np.shape(point_omega)[1]):
                for n in range(np.shape(point_eta)[1]):
                    point_temp = np.zeros(3)
                    point_temp[0:2] = point_omega[:, m] * point_eta[0, n]
                    point_temp[2] = point_eta[1, n]
                    point_temp = x.RotM @ point_temp + x.translation

                    x_mesh[m, n] = point_temp[0]
                    y_mesh[m, n] = point_temp[1]
                    z_mesh[m, n] = point_temp[2]
            
            # Create Open3D mesh from the grid
            mesh = create_mesh_from_grid(x_mesh, y_mesh, z_mesh)
            return mesh
            
        except Exception as e:
            self.get_logger().error(f"Error creating superquadric mesh: {e}")
            self.get_logger().error(traceback.format_exc())
            return None
            
    def generate_superquadric_grasps(self, object_points, workspace_points, class_id):
        """
        Generate grasps using superquadric fitting and the learning-free approach
        
        Args:
            object_points: numpy array of detected object points (Nx3)
            workspace_points: numpy array of workspace points for context
            class_id: detected object class ID
            
        Returns:
            List of 4x4 grasp pose transformation matrices
        """
        try:
            # Use full workspace point cloud and crop around detected object
            # instead of just using the detected object points (which might be incomplete)
            
            # filter object pointcloud
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            object_pcd = self.preprocess_point_cloud(object_pcd)
            object_pcd = object_pcd.remove_statistical_outlier(
                nb_neighbors=100, std_ratio=0.5)[0]
            object_points = np.asarray(object_pcd.points)
                

            # Get object center from detected points
            object_center = np.mean(object_points, axis=0)
            self.get_logger().info(f"Object center from detection: {object_center}")
            
            # Create full workspace point cloud
            full_workspace_cloud = o3d.geometry.PointCloud()
            full_workspace_cloud.points = o3d.utility.Vector3dVector(workspace_points)
            
            # Estimate object size from detected points to determine crop size
            if len(object_points) > 10:
                object_bounds = np.max(object_points, axis=0) - np.min(object_points, axis=0)
                # Add some margin around the object
                crop_margin = object_bounds * 1.2
            else:
                # Default crop size if we have very few detected points
                crop_margin = np.array([0.15, 0.15, 0.15])  # 15cm in each direction
            
            self.get_logger().info(f"Object bounds from detection: {object_bounds if len(object_points) > 10 else 'using default'}")
            self.get_logger().info(f"Crop margin: {crop_margin}")
            
            # Crop the full workspace around the detected object center
            # Similar to zed_gpu_node.py approach
            crop_min = object_center - crop_margin
            crop_max = object_center + crop_margin
            
            # Make sure we don't go below the workspace (table level)
            crop_min[2] = max(crop_min[2], 0.0)  # Don't go below table
            
            self.get_logger().info(f"Cropping workspace from {crop_min} to {crop_max}")
            
            # Crop the workspace point cloud to the relevant area
            cropped_workspace = full_workspace_cloud.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=crop_min,
                    max_bound=crop_max
                )
            )
            
            self.get_logger().info(f"Cropped workspace: {len(cropped_workspace.points)} points "
                                f"(from {len(full_workspace_cloud.points)} total)")
            
            # Remove points below a certain Z threshold (table surface) if needed
            if len(cropped_workspace.points) > 0:
                points_array = np.asarray(cropped_workspace.points)
                table_threshold = object_center[2] - crop_margin[2] * 0.8  # A bit above the table
                table_threshold = max(table_threshold, 0.01)  # Ensure we don't go below zero
                above_table_mask = points_array[:, 2] > table_threshold
                
                if np.any(above_table_mask):
                    filtered_points = points_array[above_table_mask]
                    cropped_workspace.points = o3d.utility.Vector3dVector(filtered_points)
                    self.get_logger().info(f"Filtered points above table (z > {table_threshold:.3f}): "
                                        f"{len(filtered_points)} points")
                
            # Preprocess the cropped cloud
            # cropped_workspace = cropped_workspace.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.8)[0]
            # cropped_workspace = cropped_workspace.voxel_down_sample(voxel_size=self.voxel_size)
            # cropped_workspace = self.preprocess_point_cloud(cropped_workspace)
            
            # Save the cropped object points to temporary file for EMS processing
            temp_file = f"/tmp/object_points_{class_id}_{int(time.time())}.ply"
            
            # Save the CROPPED workspace (not just detected points) as PLY file (XYZ only)
            object_xyz = o3d.geometry.PointCloud()
            object_xyz.points = cropped_workspace.points  # Use cropped workspace instead of detected points
            o3d.io.write_point_cloud(temp_file, object_xyz)
            
            # Visualize the cropped area vs detected points for comparison
            if self.visualize:
                # Color the detected points differently
                detected_cloud = o3d.geometry.PointCloud()
                detected_cloud.points = o3d.utility.Vector3dVector(object_points)
                detected_cloud.paint_uniform_color([1, 0, 0])  # Red for detected points
                
                # Color the cropped workspace
                cropped_workspace_vis = o3d.geometry.PointCloud(cropped_workspace)
                cropped_workspace_vis.paint_uniform_color([0, 0, 1])  # Blue for cropped workspace
                
                # Add coordinate frame at object center
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                coord_frame.translate(object_center)
                
                o3d.visualization.draw_geometries(
                    [detected_cloud, cropped_workspace_vis, coord_frame],
                    window_name=f"Cropped Workspace (Blue) - Class {class_id}",
                    zoom=0.7, front=[0, -1, 0], lookat=object_center, up=[0, 0, 1]
                )
                
            self.get_logger().info(f"Original detected points: {len(object_points)}, "
                                f"Cropped workspace points: {len(cropped_workspace.points)}")
            
            # Continue with the rest of your existing superquadric fitting code...
            
            # Fit superquadric using EMS on the cropped workspace
            self.get_logger().info("Fitting superquadric to cropped workspace points...")
            
            # # Load points for EMS (it expects numpy array)
            points_for_ems = np.asarray(cropped_workspace.points)  # Use cropped workspace
            
            # Fit multiple superquadrics using paper's K calculation
            best_sq, all_valid_sqs = self.fit_multiple_superquadrics_ensemble(points_for_ems)

            if best_sq is None:
                self.get_logger().error("Multi-superquadric fitting failed")
                return []

            #   CHANGE: Use ALL superquadrics, not just the best one
            self.get_logger().info(f"Paper's method complete: Using ALL {len(all_valid_sqs)} superquadrics for grasp generation")
            
            # Generate grasps from ALL superquadrics
            all_grasp_poses = []
            all_sq_info = []
            
            for i, sq_recovered in enumerate(all_valid_sqs):
                try:
                    self.get_logger().info(f"Generating grasps from superquadric {i+1}/{len(all_valid_sqs)}")
                    self.get_logger().info(f"  SQ {i+1}: Shape=({sq_recovered.shape[0]:.3f}, {sq_recovered.shape[1]:.3f}), "
                                        f"Scale=({sq_recovered.scale[0]:.3f}, {sq_recovered.scale[1]:.3f}, {sq_recovered.scale[2]:.3f})")
                    
                    consistent_euler = R_simple.from_matrix(sq_recovered.RotM).as_euler('xyz')

                    # Around line 1420, replace this:
                    sq_grasp_poses = self.grasp_planner.plan_grasps(
                        temp_file,
                        sq_recovered.shape,
                        sq_recovered.scale,
                        consistent_euler,        
                        sq_recovered.translation
                    )

                    # With this:
                    sq_grasp_data = self.grasp_planner.plan_grasps(
                        temp_file,
                        sq_recovered.shape,
                        sq_recovered.scale,
                        consistent_euler,        
                        sq_recovered.translation
                    )

                    # Extract poses and scores from the enhanced return data
                    if sq_grasp_data and len(sq_grasp_data) > 0:
                        sq_grasp_poses = [data['pose'] for data in sq_grasp_data]
                        sq_grasp_scores = [data['score'] for data in sq_grasp_data]
                        
                        self.get_logger().info(f"  SQ {i+1}: Generated {len(sq_grasp_poses)} grasps with scores")
                        for j, (pose, score) in enumerate(zip(sq_grasp_poses, sq_grasp_scores)):
                            self.get_logger().info(f"    Grasp {j+1}: score = {score:.8f}")
                        
                        # Add grasp poses with superquadric info AND scores
                        for pose, score in zip(sq_grasp_poses, sq_grasp_scores):
                            all_grasp_poses.append(pose)
                            all_sq_info.append({
                                'sq_index': i,
                                'sq': sq_recovered,
                                'euler': consistent_euler,
                                'grasp_score': score  # Add the actual grasp score
                            })
                    else:
                        self.get_logger().warn(f"  SQ {i+1}: No valid grasps generated")
                        sq_grasp_poses = []
                        sq_grasp_scores = []
                        
                except Exception as sq_error:
                    self.get_logger().error(f"Error generating grasps from SQ {i+1}: {sq_error}")
            
            if not all_grasp_poses:
                self.get_logger().error("No grasps generated from any superquadric")
                return []
            
            self.get_logger().info(f"Total grasps from all superquadrics: {len(all_grasp_poses)}")
            
            # Replace the entire scoring section (lines 1449-1485) with:
            self.get_logger().info(f"Ranking {len(all_grasp_poses)} grasps using comprehensive scores...")

            scored_grasps = []
            for i, (grasp_pose, sq_info) in enumerate(zip(all_grasp_poses, all_sq_info)):
                sq_index = sq_info['sq_index']
                sq = sq_info['sq']
                
                # Use the grasp score from the grasp planner
                grasp_score = sq_info.get('grasp_score', 0.0)
                
                # Optional: Add small superquadric quality bonus
                sq_volume = np.prod(sq.scale)
                sq_quality_bonus = min(0.1, sq_volume / 0.001)  # Small bonus for reasonable SQ size
                
                # Final composite score = main grasp score + small SQ quality bonus
                composite_score = grasp_score + sq_quality_bonus * 0.1
                
                scored_grasps.append((composite_score, grasp_pose, sq_info))
                
                self.get_logger().info(f"Grasp {i+1}: grasp_score={grasp_score:.8f}, "
                                    f"sq_bonus={sq_quality_bonus:.6f}, "
                                    f"final={composite_score:.8f}")

            # Sort by score (best first)
            scored_grasps.sort(key=lambda x: x[0], reverse=True)

            self.get_logger().info("Top 5 grasps after comprehensive scoring:")
            for i, (score, pose, sq_info) in enumerate(scored_grasps[:5]):
                sq_idx = sq_info['sq_index']
                grasp_score = sq_info.get('grasp_score', 0.0)
                grasp_pos = pose[:3, 3]
                self.get_logger().info(f"  Rank {i+1}: SQ{sq_idx+1}, grasp_score={grasp_score:.8f}, "
                                    f"final_score={score:.8f}, pos=[{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
            
            # Get top grasps (but keep diversity across superquadrics)
            final_grasp_poses = []
            used_sq_indices = set()

            # First, take the best grasp from each superquadric
            for score, grasp_pose, sq_info in scored_grasps:
                sq_index = sq_info['sq_index']
                if sq_index not in used_sq_indices:
                    final_grasp_poses.append(grasp_pose)
                    used_sq_indices.add(sq_index)
                    grasp_score = sq_info.get('grasp_score', 0.0)
                    self.get_logger().info(f"Selected best grasp from SQ {sq_index+1} "
                                        f"(grasp_score: {grasp_score:.8f}, final_score: {score:.8f})")

            # Then, add more high-scoring grasps from all superquadrics up to a limit
            max_total_grasps = 15  # Reasonable limit
            for score, grasp_pose, sq_info in scored_grasps:
                if len(final_grasp_poses) >= max_total_grasps:
                    break
                
                # Check for duplicates
                is_duplicate = False
                for existing_grasp in final_grasp_poses:
                    if np.allclose(grasp_pose, existing_grasp, atol=1e-6):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    final_grasp_poses.append(grasp_pose)
                    grasp_score = sq_info.get('grasp_score', 0.0)
                    sq_index = sq_info['sq_index']
                    self.get_logger().info(f"Added additional grasp from SQ {sq_index+1} "
                                        f"(grasp_score: {grasp_score:.8f}, final_score: {score:.8f})")

            self.get_logger().info(f"Final selection: {len(final_grasp_poses)} grasps from {len(used_sq_indices)} superquadrics")

            # Log the BEST grasp that will be published
            if final_grasp_poses:
                best_grasp = final_grasp_poses[0]
                best_score_info = scored_grasps[0]  # First item after sorting is the best
                best_grasp_score = best_score_info[2].get('grasp_score', 0.0)
                best_final_score = best_score_info[0]
                best_sq_index = best_score_info[2]['sq_index']
                
                self.get_logger().info(f"BEST GRASP: From SQ {best_sq_index+1}, "
                                    f"grasp_score={best_grasp_score:.8f}, "
                                    f"final_score={best_final_score:.8f}")
            
            #   MULTI-SUPERQUADRIC VISUALIZATION: Show all SQs and their grasps
            if self.visualize:
                self.visualize_multi_superquadric_grasps(points_for_ems, all_valid_sqs, all_grasp_poses, all_sq_info)
            
            # Return the best superquadric and all grasps
            return final_grasp_poses, best_sq
            
        except Exception as e:
            self.get_logger().error(f"Error in multi-superquadric grasp generation: {e}")
            self.get_logger().error(traceback.format_exc())
            return []
        
    def visualize_multi_superquadric_grasps(self, points_for_ems, all_valid_sqs, all_grasp_poses, all_sq_info):
        """Visualize all superquadrics and their corresponding grasps"""
        try:
            import open3d as o3d
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_for_ems)
            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray points
            geometries = [pcd]

            # Colors for each superquadric
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green  
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
            ]
            
            # Add each superquadric mesh
            for i, sq in enumerate(all_valid_sqs):
                try:
                    sq_mesh = self.superquadric_to_open3d_mesh(sq, arclength=0.2)
                    if sq_mesh is not None:
                        color = colors[i % len(colors)]
                        sq_mesh.paint_uniform_color(color)
                        geometries.append(sq_mesh)
                        
                        self.get_logger().info(f"SQ {i+1}: Color={color}, Center={sq.translation}")
                        
                except Exception as mesh_error:
                    self.get_logger().warn(f"Failed to create mesh for SQ {i+1}: {mesh_error}")
            
            # Add grasp poses as coordinate frames (colored by their source superquadric)
            for grasp_pose, sq_info in zip(all_grasp_poses, all_sq_info):
                try:
                    sq_index = sq_info['sq_index']
                    color = colors[sq_index % len(colors)]
                    
                    # Create coordinate frame for this grasp
                    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
                    
                    # Transform the frame to the grasp pose
                    grasp_frame.transform(grasp_pose)
                    
                    # Color it based on the source superquadric
                    # Note: coordinate frames have their own colors, but we can add a small sphere
                    grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    grasp_sphere.paint_uniform_color(color)
                    grasp_sphere.translate(grasp_pose[:3, 3])
                    
                    geometries.extend([grasp_frame, grasp_sphere])
                    
                except Exception as grasp_error:
                    self.get_logger().warn(f"Failed to visualize grasp from SQ {sq_index+1}: {grasp_error}")
            
            # Add main coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geometries.append(coord_frame)
            
            # Visualize everything
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Multi-Superquadric Grasps ({len(all_valid_sqs)} SQs, {len(all_grasp_poses)} grasps)",
                zoom=0.7,
                front=[0, -1, 0],
                lookat=np.mean(points_for_ems, axis=0),
                up=[0, 0, 1]
            )
            
            # Also create individual visualizations for each superquadric
            for i, sq in enumerate(all_valid_sqs):
                # Find grasps from this superquadric
                sq_grasps = [grasp for grasp, info in zip(all_grasp_poses, all_sq_info) 
                            if info['sq_index'] == i]
                
                if sq_grasps:
                    self.get_logger().info(f"Visualizing SQ {i+1} with {len(sq_grasps)} grasps")
                    
                    consistent_euler = R_simple.from_matrix(sq.RotM).as_euler('xyz')
                    
                    from zed_pose_estimation.vis2 import visualize_superquadric_grasps
                    visualize_superquadric_grasps(
                        point_cloud_data=points_for_ems,
                        superquadric_params={
                            'shape': sq.shape,
                            'scale': sq.scale,
                            'euler': consistent_euler,
                            'translation': sq.translation
                        },
                        grasp_poses=sq_grasps,
                        show_sweep_volume=False,
                        window_name=f"SQ {i+1} Grasps ({len(sq_grasps)} grasps)",
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Error in multi-superquadric visualization: {e}")

    def publish_transform(self, position, quaternion, parent_frame, child_frame):
        """Publish a transform to the tf tree"""
        try:
            # Create transform message
            transform = TransformStamped()
            
            # Set header
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent_frame
            
            # Set child frame
            transform.child_frame_id = child_frame
            
            # Set translation
            transform.transform.translation.x = float(position[0])
            transform.transform.translation.y = float(position[1])
            transform.transform.translation.z = float(position[2]) 
            
            # Set rotation
            transform.transform.rotation.x = float(quaternion[0])
            transform.transform.rotation.y = float(quaternion[1])
            transform.transform.rotation.z = float(quaternion[2])
            transform.transform.rotation.w = float(quaternion[3])
            
            # Broadcast the transform
            # print whole transform for debugging
            self.get_logger().info(f"Publishing transform {transform.child_frame_id} "
                                f"from {transform.header.frame_id} at {transform.header.stamp.sec}.{transform.header.stamp.nanosec}")
            self.get_logger().info(f"Position: {transform.transform.translation.x}, "
                                f"{transform.transform.translation.y}, {transform.transform.translation.z}")
            self.get_logger().info(f"Rotation: {transform.transform.rotation.x}, "
                                f"{transform.transform.rotation.y}, {transform.transform.rotation.z}, "
                                f"{transform.transform.rotation.w}")
            # Send the transform
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {e}")
            self.get_logger().error(traceback.format_exc())    

            
    def destroy_node(self):
        """Clean up resources when the node is destroyed"""
        self.get_logger().info("Shutting down ZED cameras...")
        if hasattr(self, 'zed1'):
            self.zed1.close()
        if hasattr(self, 'zed2'):
            self.zed2.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = None # Initialize node to None
    try:
        node = ZedGpuNode()
        
        # Use multi-threaded executor for better performance
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            # Use spin_once with timeout to check shutdown flag regularly
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
                if node.future_shutdown:
                    break
        finally:
            executor.shutdown()
            # node.destroy_node() is called below, no need for cv2.destroyAllWindows() here
            # as destroy_node() handles it.
    
    except Exception as e:
        if node:
            node.get_logger().error(f"Error in main: {e}")
            node.get_logger().error(traceback.format_exc())
        else:
            print(f"Error in main (node not initialized): {e}")
            print(traceback.format_exc())
    
    finally:
        if node:
            node.destroy_node() # This already calls cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()