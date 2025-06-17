#!/usr/bin/env python3

import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import os
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from rclpy.executors import MultiThreadedExecutor
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from zed_pose_estimation.vision_pipeline_utils import crop_point_cloud_gpu, fuse_point_clouds_centroid, subtract_point_clouds_gpu, convert_mask_to_3d_points, downsample_point_cloud_gpu
import open3d as o3d
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation
import traceback

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
            os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'transform.yaml')) # Default path
        
        # New ICP parameters
        self.declare_parameter('icp_enabled', True)
        self.declare_parameter('model_path', '/home/chris/franka_ros2_ws/src/zed_pose_estimation/pointclouds/cone with planar surface_orange.ply')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('icp_distance_threshold', 0.03)
        self.declare_parameter('visualize_icp', False)  # Separate from other visualization
        
        # Get original parameters
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

        
        # Get new ICP parameters
        self.icp_enabled = self.get_parameter('icp_enabled').get_parameter_value().bool_value
        self.icp_model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.icp_distance_threshold = self.get_parameter('icp_distance_threshold').get_parameter_value().double_value
        self.visualize_icp = self.get_parameter('visualize_icp').get_parameter_value().bool_value
        
        self.future_shutdown = False

        self.class_names = {0: "Box", 1: "Cone", 2: "Cover", 3: "Plier", 
                           4: "Robot", 5: "Scre Driver"}
        
        # Dictionary to store timings for benchmarking
        self.timings = {
            "Frame Retrieval": [],
            "Depth Retrieval": [],
            "Point Cloud Processing": [],
            "YOLO Inference": [],
            "Mask Processing": [],
            "Point Cloud Fusion": [],
            "Subtraction": [],
            "ICP": [],
            "Total Time": []
        }
        
        # Setup visualization window if enabled
        if self.publish_viz:
            # Create a window to display the output
            cv2.namedWindow("YOLO Detection")
        
        # FPS calculation
        self.fps_values = []
        
        # Check if CUDA is available and set the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load ICP reference model if enabled
        if self.icp_enabled:
            self.load_reference_model()
            
            # Setup TF broadcaster and pose publisher for ICP
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

    def load_reference_model(self):
        """Load reference model for ICP alignment"""
        try:
            # Load reference model
            original_model = o3d.io.read_point_cloud(self.icp_model_path)
            if not original_model.has_points():
                self.get_logger().error(f"Model file {self.icp_model_path} loaded but contains no points!")
                self.icp_enabled = False
                return

            self.get_logger().info(f"Loaded model with {len(original_model.points)} points from {self.icp_model_path}")

            # Scale model if needed (mm to meters)
            scale_factor = 0.001  # Convert mm to meters
            scaled_points = np.asarray(original_model.points) * scale_factor

            self.reference_model = o3d.geometry.PointCloud()
            self.reference_model.points = o3d.utility.Vector3dVector(scaled_points)
            # Downsample
            self.reference_model = self.reference_model.voxel_down_sample(voxel_size=self.voxel_size)
            self.processed_ref_points = np.asarray(self.reference_model.points).copy()

        except Exception as e:
            self.get_logger().error(f"Failed to load reference model: {e}")
            self.get_logger().error(traceback.format_exc())
            self.icp_enabled = False

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
        resolution = sl.Resolution(640, 360)  # Lower resolution for faster processing
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

    def process_frames(self):
        """Main processing loop - timer callback"""
        try:
            start_time = time.time()
            
            # Check if cameras are ready
            if (self.zed1.grab() != sl.ERROR_CODE.SUCCESS or 
                self.zed2.grab() != sl.ERROR_CODE.SUCCESS):
                self.get_logger().warn("Failed to grab frames from cameras")
                return
                
            # Step 1: Frame retrieval
            retrieval_start = time.time()
            # Retrieve RGB images
            self.zed1.retrieve_image(self.image1, view=sl.VIEW.LEFT)
            self.zed2.retrieve_image(self.image2, view=sl.VIEW.LEFT)
            frame1 = self.image1.get_data()
            frame2 = self.image2.get_data()
            # Convert from RGBA to RGB
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
            
            # Step 3: Point cloud processing (with color)
            pc_start = time.time()
            # Retrieve point clouds with color information
            self.zed1.retrieve_measure(self.point_cloud1_ws, measure=sl.MEASURE.XYZRGBA)
            self.zed2.retrieve_measure(self.point_cloud2_ws, measure=sl.MEASURE.XYZRGBA)
            
            # Get point cloud data with color
            pc1_data = self.point_cloud1_ws.get_data()
            pc2_data = self.point_cloud2_ws.get_data()
            
            # Extract XYZ coordinates
            pc1_xyz = pc1_data[:, :, :3]  # View, not copy
            pc1_tensor = torch.from_numpy(pc1_xyz).float().to(self.device).reshape(-1, 3)
            pc2_xyz = pc2_data[:, :, :3]  # View, not copy
            pc2_tensor = torch.from_numpy(pc2_xyz).float().to(self.device).reshape(-1, 3)

            # Extract RGBA color information (if needed)
            pc1_colors = pc1_data[:, :, 3].reshape(-1)  # RGBA packed as uint32
            pc2_colors = pc2_data[:, :, 3].reshape(-1)
            
            # Debug: Check data types
            self.get_logger().info(f"pc1_colors dtype: {pc1_colors.dtype}, shape: {pc1_colors.shape}")
            self.get_logger().info(f"pc2_colors dtype: {pc2_colors.dtype}, shape: {pc2_colors.shape}")
            
            # Convert packed RGBA to separate RGB channels (if you need RGB values)
            def unpack_rgba(packed_colors):
                try:
                    if packed_colors.dtype != np.uint32:
                        packed_colors = packed_colors.view(np.uint32)
                    # Unpack RGBA from uint32
                    r = (packed_colors & 0xFF).astype(np.uint8)
                    g = ((packed_colors >> 8) & 0xFF).astype(np.uint8)
                    b = ((packed_colors >> 16) & 0xFF).astype(np.uint8)
                    a = ((packed_colors >> 24) & 0xFF).astype(np.uint8)
                    return np.column_stack([r, g, b, a])
                except ValueError as e:
                    self.get_logger().error(f"RGBA unpacking failed: {e}")
                    return np.zeros((len(packed_colors), 4), dtype=np.uint8)
            
            pc1_rgba = unpack_rgba(pc1_colors)
            pc2_rgba = unpack_rgba(pc2_colors)
            
            # Filter invalid points
            valid_mask1 = torch.isfinite(pc1_tensor).all(dim=1)
            valid_mask2 = torch.isfinite(pc2_tensor).all(dim=1)
            pc1_tensor = pc1_tensor[valid_mask1]
            pc2_tensor = pc2_tensor[valid_mask2]
            
            # Also filter colors to match valid points
            pc1_rgba_valid_np = pc1_rgba[valid_mask1.cpu().numpy()]
            pc2_rgba_valid_np = pc2_rgba[valid_mask2.cpu().numpy()]

            # Create Open3D point clouds for raw data
            raw_cloud1 = o3d.geometry.PointCloud()
            raw_cloud1.points = o3d.utility.Vector3dVector(pc1_tensor.cpu().numpy())  # Use the actual 3D points
            raw_cloud1.colors = o3d.utility.Vector3dVector(pc1_rgba_valid_np[:, :3] / 255.0)  # Use RGB columns (0:3)
            
            raw_cloud2 = o3d.geometry.PointCloud()
            raw_cloud2.points = o3d.utility.Vector3dVector(pc2_tensor.cpu().numpy())  # Use the actual 3D points
            raw_cloud2.colors = o3d.utility.Vector3dVector(pc2_rgba_valid_np[:, :3] / 255.0)  # Use RGB columns (0:3)


            # Create coordinate frame for reference
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # Option 1: Visualize each camera separately
            # Camera 1
            o3d.visualization.draw_geometries(
                [raw_cloud1, coord_frame],
                zoom=0.35,
                front=[0, 0, -1],
                lookat=[0, 0, 0],
                up=[0, -1, 0],
                window_name="Camera 1 - Raw Point Cloud with Colors"
            )
            
            # Camera 2
            o3d.visualization.draw_geometries(
                [raw_cloud2, coord_frame],
                zoom=0.35,
                front=[0, 0, -1],
                lookat=[0, 0, 0],
                up=[0, -1, 0],
                window_name="Camera 2 - Raw Point Cloud with Colors"
            )
            
            # Transform point clouds to robot frame
            pc1_transformed = torch.mm(pc1_tensor, self.rotation1_torch.T) + self.origin1_torch
            pc2_transformed = torch.mm(pc2_tensor, self.rotation2_torch.T) + self.origin2_torch
            
            # Crop to workspace - you'll need to modify your crop function to return indices
            x_bounds = (self.workspace_bounds[0], self.workspace_bounds[1])
            y_bounds = (self.workspace_bounds[2], self.workspace_bounds[3])
            z_bounds = (self.workspace_bounds[4], self.workspace_bounds[5])
            
            # Get cropping masks to track which points remain
            crop_mask1 = (
                (pc1_transformed[:, 0] >= x_bounds[0]) & (pc1_transformed[:, 0] <= x_bounds[1]) &
                (pc1_transformed[:, 1] >= y_bounds[0]) & (pc1_transformed[:, 1] <= y_bounds[1]) &
                (pc1_transformed[:, 2] >= z_bounds[0]) & (pc1_transformed[:, 2] <= z_bounds[1])
            )
            crop_mask2 = (
                (pc2_transformed[:, 0] >= x_bounds[0]) & (pc2_transformed[:, 0] <= x_bounds[1]) &
                (pc2_transformed[:, 1] >= y_bounds[0]) & (pc2_transformed[:, 1] <= y_bounds[1]) &
                (pc2_transformed[:, 2] >= z_bounds[0]) & (pc2_transformed[:, 2] <= z_bounds[1])
            )
            
            # Apply cropping to both points and colors
            pc1_cropped = pc1_transformed[crop_mask1]
            pc2_cropped = pc2_transformed[crop_mask2]
            pc1_colors_cropped = pc1_rgba_valid_np[crop_mask1.cpu().numpy()]
            pc2_colors_cropped = pc2_rgba_valid_np[crop_mask2.cpu().numpy()]
            
            # For downsampling with color preservation, you'll need a custom function
            # or use Open3D which handles colors automatically
            def downsample_with_colors(points_tensor, colors_np, voxel_size):
                # Convert to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_tensor.cpu().numpy())
                pcd.colors = o3d.utility.Vector3dVector(colors_np[:, :3] / 255.0)  # Normalize RGB
                
                # Downsample
                pcd_down = pcd.voxel_down_sample(voxel_size)
                
                return pcd_down
            
            # Downsample with colors
            pc1_downsampled_colored = downsample_with_colors(pc1_cropped, pc1_colors_cropped, self.voxel_size)
            pc2_downsampled_colored = downsample_with_colors(pc2_cropped, pc2_colors_cropped, self.voxel_size)
            
            # Create colored workspace point clouds
            pc1_workspace_colored = o3d.geometry.PointCloud()
            pc1_workspace_colored.points = o3d.utility.Vector3dVector(pc1_cropped.cpu().numpy())
            pc1_workspace_colored.colors = o3d.utility.Vector3dVector(pc1_colors_cropped[:, :3] / 255.0)
            
            pc2_workspace_colored = o3d.geometry.PointCloud()
            pc2_workspace_colored.points = o3d.utility.Vector3dVector(pc2_cropped.cpu().numpy())
            pc2_workspace_colored.colors = o3d.utility.Vector3dVector(pc2_colors_cropped[:, :3] / 255.0)
            
            # Fuse colored workspace point clouds
            fused_workspace_colored = pc1_workspace_colored + pc2_workspace_colored
            
            # Optional: Remove duplicate points while preserving colors
            fused_workspace_colored = fused_workspace_colored.voxel_down_sample(self.voxel_size * 0.5)
            
            # Preprocess if needed (this might affect colors)
            fused_workspace_colored_processed = self.preprocess_point_cloud(fused_workspace_colored)
            
            # Visualize the colored workspace point cloud
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # Option 1: Show raw fused colored workspace
            o3d.visualization.draw_geometries(
                [fused_workspace_colored, coordinate_frame],
                window_name="Fused Workspace - Colored (Raw)",
                zoom=0.7,
                front=[0, -1, 0],
                lookat=[0, 0, 0],
                up=[0, 0, 1]
            )
            
            # Option 2: Show processed colored workspace
            o3d.visualization.draw_geometries(
                [fused_workspace_colored_processed, coordinate_frame],
                window_name="Fused Workspace - Colored (Processed)",
                zoom=0.7,
                front=[0, -1, 0],
                lookat=[0, 0, 0],
                up=[0, 0, 1]
            )
            
            # Option 3: Show both cameras with different tints for identification
            pc1_tinted = o3d.geometry.PointCloud(pc1_workspace_colored)
            pc2_tinted = o3d.geometry.PointCloud(pc2_workspace_colored)
            
            # Add red tint to camera 1, blue tint to camera 2
            colors1 = np.asarray(pc1_tinted.colors)
            colors2 = np.asarray(pc2_tinted.colors)
            colors1[:, 0] = np.minimum(colors1[:, 0] + 0.2, 1.0)  # Add red
            colors2[:, 2] = np.minimum(colors2[:, 2] + 0.2, 1.0)  # Add blue
            pc1_tinted.colors = o3d.utility.Vector3dVector(colors1)
            pc2_tinted.colors = o3d.utility.Vector3dVector(colors2)
            
            o3d.visualization.draw_geometries(
                [pc1_tinted, pc2_tinted, coordinate_frame],
                window_name="Workspace - Camera 1 (Red) + Camera 2 (Blue)",
                zoom=0.7,
                front=[0, -1, 0],
                lookat=[0, 0, 0],
                up=[0, 0, 1]
            )
            
            # Save colored point clouds for external visualization
            # o3d.io.write_point_cloud("pointclouds/workspace_cam1_colored.ply", pc1_workspace_colored)
            # o3d.io.write_point_cloud("pointclouds/workspace_cam2_colored.ply", pc2_workspace_colored)
            # o3d.io.write_point_cloud("pointclouds/workspace_fused_colored.ply", fused_workspace_colored)
            # o3d.io.write_point_cloud("pointclouds/workspace_fused_colored_processed.ply", fused_workspace_colored_processed)
            # #fuse tinted point clouds
            # fused_tinted_workspace = pc1_tinted + pc2_tinted
            # o3d.io.write_point_cloud("pointclouds/workspace_fused_tinted.ply", fused_tinted_workspace)
            
            
            # For the rest of your pipeline, you might want to keep the non-colored version
            # for compatibility with existing functions
            fused_workspace = torch.cat((pc1_cropped, pc2_cropped), dim=0)
            fused_workspace_np = fused_workspace.cpu().numpy()
            pcd_fused_workspace = o3d.geometry.PointCloud()
            pcd_fused_workspace.points = o3d.utility.Vector3dVector(fused_workspace_np)
            pcd_fused_workspace = self.preprocess_point_cloud(pcd_fused_workspace)
            
            #visualize the workspace point cloud
            # Visualize the workspace point cloud
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd_fused_workspace, coordinate_frame])
            
            pc_time = time.time() - pc_start
            self.timings["Point Cloud Processing"].append(pc_time)
            
            # Check model names
            # self.get_logger().info(f"Model names: {self.model.names}")
            # Step 4: YOLO inference
            yolo_start = time.time()
            frame_batch = [frame1, frame2] # Create a batch of frames
            
            results_batch = self.model.track(
                source=frame_batch, # Process batch
                classes=[1],  # Cone
                persist=True, # Tracker state is persisted by the model
                retina_masks=True,
                conf=self.conf_threshold,
                device=self.device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml" # Tracker config
            )
            
            # Unpack results for each camera
            results1 = results_batch[0]  # This is a Results object for camera 1
            results2 = results_batch[1]  # This is a Results object for camera 2
            
            # Get Masks and Boxes objects directly from the Results objects
            # These will be None if no detections, or the corresponding Ultralytics objects
            masks_obj1 = results1.masks 
            masks_obj2 = results2.masks
            
            boxes_obj1 = results1.boxes
            boxes_obj2 = results2.boxes
            
            # Extract class IDs if boxes exist
            # .cls is a tensor of class indices, .cpu().numpy() converts it
            # Ensure class_ids are numpy arrays even if empty
            class_ids1 = boxes_obj1.cls.cpu().numpy() if boxes_obj1 is not None else np.array([])
            class_ids2 = boxes_obj2.cls.cpu().numpy() if boxes_obj2 is not None else np.array([])
            
            yolo_time = time.time() - yolo_start
            self.timings["YOLO Inference"].append(yolo_time)
            
            # Step 5: Process masks to get object point clouds
            mask_start = time.time()
            point_clouds_camera1 = []
            point_clouds_camera2 = []
            
            # Process camera 1 masks
            # masks_obj1 is a Masks object. masks_obj1.data is a tensor of shape (N, H, W)
            if masks_obj1 is not None and masks_obj1.data.numel() > 0:
                depth_map1_torch = torch.tensor(depth_np1, dtype=torch.float32, device=self.device)
                for i, individual_mask_tensor in enumerate(masks_obj1.data): # Iterate over each detected object's mask
                    # individual_mask_tensor is one mask (H, W)
                    # The corresponding class ID is class_ids1[i]
                    
                    mask_indices_full = torch.nonzero(individual_mask_tensor, as_tuple=False)
                    mask_indices = mask_indices_full # Use all points from the mask


                    if mask_indices.numel() > 0: 
                        with torch.amp.autocast('cuda'):
                            points_3d = convert_mask_to_3d_points(
                                mask_indices, depth_map1_torch, self.cx1, self.cy1, self.fx1, self.fy1
                            )
                        
                        if points_3d.size(0) > 0:
                            transformed = torch.mm(points_3d, self.rotation1_torch.T) + self.origin1_torch
                            downsampled = downsample_point_cloud_gpu(transformed, self.voxel_size)
                            point_clouds_camera1.append((downsampled.cpu().numpy(), int(class_ids1[i])))
            
            # Process camera 2 masks
            if masks_obj2 is not None and masks_obj2.data.numel() > 0:
                depth_map2_torch = torch.tensor(depth_np2, dtype=torch.float32, device=self.device)
                for i, individual_mask_tensor in enumerate(masks_obj2.data):
                    mask_indices_full = torch.nonzero(individual_mask_tensor, as_tuple=False)
                    mask_indices = mask_indices_full # Use all points from the mask

                    if mask_indices.numel() > 0: 
                        with torch.amp.autocast('cuda'):
                            points_3d = convert_mask_to_3d_points(
                                mask_indices, depth_map2_torch, self.cx2, self.cy2, self.fx2, self.fy2
                            )
                        
                        if points_3d.size(0) > 0:
                            transformed = torch.mm(points_3d, self.rotation2_torch.T) + self.origin2_torch
                            downsampled = downsample_point_cloud_gpu(transformed, self.voxel_size)
                            point_clouds_camera2.append((downsampled.cpu().numpy(), int(class_ids2[i])))
            
            mask_time = time.time() - mask_start
            self.timings["Mask Processing"].append(mask_time)
            
            # Step 6: Fuse object point clouds
            fusion_start = time.time()
            _, _, fused_objects = fuse_point_clouds_centroid(
                point_clouds_camera1, point_clouds_camera2, self.distance_threshold
            )
            
            # Extract point clouds from fused objects
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
            
            # Step 8: Perform ICP on detected objects if enabled
            if self.icp_enabled and fused_object_points and len(self.processed_ref_points) > 0:
                icp_start = time.time()
                
                # Process each detected object with ICP
                for i, (object_points, class_id) in enumerate(zip(fused_object_points, fused_object_classes)):
                    # Skip if too few points
                    if len(object_points) < 50:
                        self.get_logger().warn(f"Object {i} has too few points ({len(object_points)}) for ICP")
                        continue
                        
                    # Convert object points to Open3D format
                    object_cloud = o3d.geometry.PointCloud()
                    object_cloud.points = o3d.utility.Vector3dVector(object_points)
                    
                    # Preprocess the point cloud
                    object_cloud = self.preprocess_point_cloud(object_cloud)
                    
                    # Class-specific processing for objects (e.g., for cups or bottles)
                    if class_id in [1]:  # Cone
                        self.get_logger().info(f"Processing {self.class_names.get(class_id, 'Unknown')} for ICP")
                        
                        # Run Colored ICP to get pose
                        pose_matrix = self.estimate_pose_with_colored_icp(object_cloud, fused_workspace_colored_processed)
                        
                        # # Fallback to geometric ICP if colored fails
                        # if pose_matrix is None:
                        #     self.get_logger().info("Colored ICP failed, trying geometric ICP")
                        #     pose_matrix = self.estimate_pose_with_icp(object_cloud, fused_workspace_np)
                        
                        if pose_matrix is not None:
                            # Extract and publish pose
                            position = pose_matrix[:3, 3]
                            # # add a little bit to x postion
                            # position[0] += 0.02
                            # add half of height to z position
                            position = pose_matrix[:3, 3].copy()  # Create a writable copy
                            # position[2] += self.reference_model.get_max_bound()[2] / 2
                            rotation_matrix = pose_matrix[:3, :3]
                            rotation = Rotation.from_matrix(rotation_matrix)
                            quat = rotation.as_quat()  # x, y, z, w
                            euler = rotation.as_euler('xyz', degrees=True)
                            
                            # if euler_z > 90: subtract 180
                            if euler[2] > 90 or euler[2] < -90:
                                euler[2] -= 180
                                # calculate quaternion from euler angles
                                rotation = Rotation.from_euler('xyz', euler, degrees=True)
                                quat = rotation.as_quat()  # x, y, z, w
                                
                            # set the orientation to zero for now
                            quat[0] = 1
                            quat[1] = 0
                            quat[2] = 0
                            quat[3] = 0
                            
                            
                            # Create pose message
                            pose_msg = PoseStamped()
                            pose_msg.header = Header()
                            pose_msg.header.stamp = self.get_clock().now().to_msg()
                            pose_msg.header.frame_id = self.target_frame
                            pose_msg.pose.position.x = float(position[0])
                            pose_msg.pose.position.y = float(position[1])
                            pose_msg.pose.position.z =  0.05
                            pose_msg.pose.orientation.x = float(quat[0])
                            pose_msg.pose.orientation.y = float(quat[1])
                            pose_msg.pose.orientation.z = float(quat[2])
                            pose_msg.pose.orientation.w = float(quat[3])
                            
                            # Publish pose and TF
                            self.pose_publisher.publish(pose_msg)
                            self.get_logger().info(f"Object {i} ({self.class_names.get(class_id, 'Unknown')}) pose: "
                                                  f"Position[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
                                                  f" Euler[{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]")
                            
                            if self.publish_tf:
                                self.publish_transform(position, quat, self.target_frame, f"detected_{self.class_names.get(class_id, 'object')}_{i}")
                            
                            # Visualize alignment if requested
                            # if self.visualize_icp:
                            #     self.visualize_alignment(pcd_fused_workspace, pose_matrix)
                
                icp_time = time.time() - icp_start
                self.timings["ICP"].append(icp_time)
            
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
            
            # Keep only last 10 FPS values
            if len(self.fps_values) > 10:
                self.fps_values.pop(0)
            avg_fps = sum(self.fps_values) / len(self.fps_values)
            
            # Visualization using OpenCV
            if self.publish_viz:
                try:
                    # Get raw frames for display
                    display1 = frame1.copy()
                    display2 = frame2.copy()
                    
                    # Draw bounding boxes on frames
                    if results1[0].boxes is not None and len(results1[0].boxes) > 0:
                        for box in results1[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(display1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if results2[0].boxes is not None and len(results2[0].boxes) > 0:
                        for box in results2[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(display2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add FPS overlay
                    cv2.putText(display1, f"FPS: {avg_fps:.1f}", (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Resize for display
                    display1 = cv2.resize(display1, (640, 360))
                    display2 = cv2.resize(display2, (640, 360))
                    
                    # Concatenate frames horizontally
                    display = np.hconcat([display1, display2])
                    
                    # Show window
                    cv2.imshow("YOLO Detection", display)
                    key = cv2.waitKey(1)
                    
                    # Check for quit key
                    if key == ord('q'):
                        self.get_logger().info("User requested shutdown (Q key pressed)")
                        self.future_shutdown = True
                        
                except Exception as viz_error:
                    self.get_logger().error(f"Visualization error: {viz_error}")
                    self.get_logger().error(traceback.format_exc())
            
            # Clean up to prevent memory buildup
            point_clouds_camera1.clear()
            point_clouds_camera2.clear()
            
        except Exception as e:
            self.get_logger().error(f"Error in process_frames: {e}")
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
            self.get_logger().error(traceback.format_exc())
            return pcd  # Return the original if preprocessing fails
        
    def draw_pca_axes(self, cloud, axis_length=0.05, origin_color=[0, 0, 0]):
        # Convert to numpy array
        points = np.asarray(cloud.points)
        center = np.mean(points, axis=0)

        # Perform PCA
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Create line segments along the principal components
        axes = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB for x, y, z
        for i in range(3):
            axis = o3d.geometry.LineSet()
            pts = [center, center + axis_length * eigenvectors[:, i]]
            axis.points = o3d.utility.Vector3dVector(pts)
            axis.lines = o3d.utility.Vector2iVector([[0, 1]])
            axis.colors = o3d.utility.Vector3dVector([colors[i]])
            axes.append(axis)

        return axes
    
    def get_pca_basis(self,points):
        """Compute PCA basis vectors sorted by descending eigenvalues."""
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def estimate_pose_with_colored_icp(self, observed_cloud, full_cloud):
        if len(observed_cloud.points) < 10:
            self.get_logger().warn("Not enough points for ICP registration")
            return None

        # --- Prepare reference point cloud with colors
        ref_points = np.asarray(self.processed_ref_points.copy())
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
        
        # Add orange color to reference model if it doesn't have colors
        if not ref_pcd.has_colors():
            orange_color = np.tile([1.0, 0.65, 0.0], (len(ref_points), 1))  # Orange color
            ref_pcd.colors = o3d.utility.Vector3dVector(orange_color)

        # --- Get centers
        source_center = ref_pcd.get_center()
        target_center = observed_cloud.get_center()

        # --- Crop full cloud around observed object (maintain colors)
        ref_size = np.array(ref_pcd.get_max_bound()) - np.array(ref_pcd.get_min_bound())
        cropped_cloud = full_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=target_center - 1.2 * ref_size,
            max_bound=target_center + 1.2 * ref_size
        ))
        
        # visualize cropped cloud
        o3d.visualization.draw_geometries([cropped_cloud], window_name="Cropped Cloud for ICP")
        

        # Ensure both clouds have colors for colored ICP
        if not observed_cloud.has_colors():
            gray_color = np.tile([0.5, 0.5, 0.5], (len(observed_cloud.points), 1))
            observed_cloud.colors = o3d.utility.Vector3dVector(gray_color)
            
        if not cropped_cloud.has_colors():
            white_color = np.tile([1.0, 1.0, 1.0], (len(cropped_cloud.points), 1))
            cropped_cloud.colors = o3d.utility.Vector3dVector(white_color)

        # --- PCA for initial alignment (using geometry only, not colors)
        obs_evals, obs_basis = self.get_pca_basis(np.asarray(observed_cloud.points))
        ref_evals, ref_basis = self.get_pca_basis(ref_points)

        self.get_logger().info(f"Observed eigenvalues: {obs_evals}")
        self.get_logger().info(f"Reference eigenvalues: {ref_evals}")

        # Try different PCA alignments for initial transformation
        pca_axis_orders = [(0, 1), (1, 2), (2, 0)]
        flip_configs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        best_initial_result = {
            "fitness": 0.0,
            "transformation": None,
            "config": ""
        }

        for axis_pair in pca_axis_orders:
            i1, i2 = axis_pair
            for flip1, flip2 in flip_configs:
                # --- Observed basis
                pc1_obs = obs_basis[:, i1] * flip1
                pc2_obs = obs_basis[:, i2] * flip2
                pc2_obs -= np.dot(pc2_obs, pc1_obs) * pc1_obs
                pc2_obs /= np.linalg.norm(pc2_obs)
                pc3_obs = np.cross(pc1_obs, pc2_obs)
                obs_rot = np.column_stack([pc1_obs, pc2_obs, pc3_obs])

                # --- Reference basis
                pc1_ref = ref_basis[:, 0]
                pc2_ref = ref_basis[:, 1]
                pc2_ref -= np.dot(pc2_ref, pc1_ref) * pc1_ref
                pc2_ref /= np.linalg.norm(pc2_ref)
                pc3_ref = np.cross(pc1_ref, pc2_ref)
                ref_rot = np.column_stack([pc1_ref, pc2_ref, pc3_ref])

                # --- Initial transformation
                R = obs_rot @ ref_rot.T
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = target_center - R @ source_center

                # Quick test with largest voxel size
                try:
                    transformed_ref = o3d.geometry.PointCloud(ref_pcd)
                    transformed_ref.transform(T)
                    
                    # Test with coarse resolution for speed
                    test_source = transformed_ref.voxel_down_sample(0.02)
                    test_target = cropped_cloud.voxel_down_sample(0.02)
                    
                    test_result = o3d.pipelines.registration.registration_colored_icp(
                        test_source, test_target, 0.04, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
                    )
                    
                    if test_result.fitness > best_initial_result["fitness"]:
                        best_initial_result.update({
                            "fitness": test_result.fitness,
                            "transformation": T,
                            "config": f"PC{i1}-{i2}_flip1={flip1}_flip2={flip2}"
                        })
                        
                except RuntimeError:
                    continue  # Skip this configuration if it fails

        if best_initial_result["transformation"] is None:
            self.get_logger().warn("Failed to find initial alignment for colored ICP")
            return None

        self.get_logger().info(f"Best initial config: {best_initial_result['config']} with fitness {best_initial_result['fitness']:.3f}")

        # --- Use much smaller voxel sizes for small objects like cones
        voxel_radius = [0.005, 0.004, 0.003]  # Much smaller: 8mm, 5mm, 3mm
        max_iter = [30, 20, 15]               # Fewer iterations since we have more points
        current_transformation = best_initial_result["transformation"]
        
        self.get_logger().info("Starting multi-scale colored point cloud registration with fine voxel sizes")
        
        # Transform reference with initial alignment
        source = o3d.geometry.PointCloud(ref_pcd)
        source.transform(current_transformation)
        target = cropped_cloud
        
        # Debug: Check initial point counts
        self.get_logger().info(f"Initial point counts - Source: {len(source.points)}, Target: {len(target.points)}")
        
        # visualize initial alignment
        o3d.visualization.draw_geometries(
            [source, target],
            window_name="Initial Alignment for Colored ICP",
            zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1]
        )
        self.get_logger().info("Initial alignment done, starting multi-scale ICP")
        
        try:
            for scale in range(3):
                iter_count = max_iter[scale]
                radius = voxel_radius[scale]
                
                self.get_logger().info(f"Scale {scale}: radius={radius:.3f}, iterations={iter_count}")

                # Smart downsampling - only downsample if we have enough points
                if len(source.points) > 200:
                    source_down = source.voxel_down_sample(radius)
                    self.get_logger().info(f"Downsampled source from {len(source.points)} to {len(source_down.points)} points")
                else:
                    source_down = source
                    self.get_logger().info(f"Source too small ({len(source.points)} points), skipping downsampling")

                if len(target.points) > 200:
                    target_down = target.voxel_down_sample(radius)
                    self.get_logger().info(f"Downsampled target from {len(target.points)} to {len(target_down.points)} points")
                else:
                    target_down = target
                    self.get_logger().info(f"Target too small ({len(target.points)} points), skipping downsampling")
                
                # Check if we have enough points for meaningful ICP
                if len(source_down.points) < 20 or len(target_down.points) < 20:
                    self.get_logger().warn(f"Too few points for scale {scale} - Source: {len(source_down.points)}, Target: {len(target_down.points)}")
                    self.get_logger().warn("Skipping this scale")
                    continue
                
                # visualize downsampled clouds
                o3d.visualization.draw_geometries(
                    [source_down, target_down],
                    window_name=f"Downsampled Clouds - Scale {scale}",
                    zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1]
                )
                self.get_logger().info(f"Downsampled clouds - Source: {len(source_down.points)}, Target: {len(target_down.points)}")

                # Estimate normals AFTER downsampling
                source_down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius * 3, max_nn=30  # Larger search radius for small point clouds
                    )
                )
                target_down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius * 3, max_nn=30
                    )
                )
                
                # Apply colored point cloud registration
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, np.eye(4),  # Use identity for relative transformation
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=iter_count
                    )
                )
                
                # Update cumulative transformation
                current_transformation = result_icp.transformation @ current_transformation
                
                # Transform source for next iteration
                source.transform(result_icp.transformation)
                
                self.get_logger().info(
                    f"Scale {scale} result - Fitness: {result_icp.fitness:.3f}, "
                    f"RMSE: {result_icp.inlier_rmse:.5f}"
                )
                
                # Early termination if fitness is very good
                if result_icp.fitness > 0.95:
                    self.get_logger().info(f"Excellent fitness achieved at scale {scale}, terminating early")
                    break
            
            # Final result
            final_fitness = result_icp.fitness
            final_rmse = result_icp.inlier_rmse
            
            self.get_logger().info(
                f"Multi-scale colored ICP completed - Final fitness: {final_fitness:.3f}, "
                f"RMSE: {final_rmse:.5f}"
            )
            
            # Visualization if enabled
            if self.visualize_icp:
                final_ref = o3d.geometry.PointCloud(ref_pcd)
                final_ref.transform(current_transformation)
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

                o3d.visualization.draw_geometries(
                    [target, final_ref, coord],
                    window_name="Multi-scale Colored ICP Result",
                    zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1]
                )
            
            return current_transformation
            
        except RuntimeError as e:
            self.get_logger().warn(f"Multi-scale colored ICP failed: {e}")
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected error in multi-scale colored ICP: {e}")
            return None


    
    def estimate_pose_with_icp(self, observed_cloud, full_cloud):
        if len(observed_cloud.points) < 10:
            self.get_logger().warn("Not enough points for ICP registration")
            return None

        # --- Prepare reference point cloud
        ref_points = np.asarray(self.processed_ref_points.copy())
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points)

        # --- Get centers
        source_center = ref_pcd.get_center()
        target_center = observed_cloud.get_center()

        # --- Crop full cloud around observed object
        ref_size = np.array(ref_pcd.get_max_bound()) - np.array(ref_pcd.get_min_bound())
        cropped_cloud = full_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=target_center - 1.2 * ref_size,
            max_bound=target_center + 1.2 * ref_size
        ))

        # --- PCA for both clouds
        obs_evals, obs_basis = self.get_pca_basis(np.asarray(observed_cloud.points))
        ref_evals, ref_basis = self.get_pca_basis(ref_points)

        self.get_logger().info(f"Observed eigenvalues: {obs_evals}")
        self.get_logger().info(f"Reference eigenvalues: {ref_evals}")

        # Try different PCA alignments
        pca_axis_orders = [
            (0, 1),
            (1, 2),
            (2, 0),
        ]
        flip_configs = [
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1)
        ]

        best_result = {
            "fitness": 0.0,
            "rmse": float("inf"),
            "transformation": None,
            "config": "",
            "reference": None,
            "threshold": 0.01
        }

        for axis_pair in pca_axis_orders:
            i1, i2 = axis_pair
            for flip1, flip2 in flip_configs:
                # --- Observed basis
                pc1_obs = obs_basis[:, i1] * flip1
                pc2_obs = obs_basis[:, i2] * flip2
                pc2_obs -= np.dot(pc2_obs, pc1_obs) * pc1_obs
                pc2_obs /= np.linalg.norm(pc2_obs)
                pc3_obs = np.cross(pc1_obs, pc2_obs)
                obs_rot = np.column_stack([pc1_obs, pc2_obs, pc3_obs])

                # --- Reference basis
                pc1_ref = ref_basis[:, 0]
                pc2_ref = ref_basis[:, 1]
                pc2_ref -= np.dot(pc2_ref, pc1_ref) * pc1_ref
                pc2_ref /= np.linalg.norm(pc2_ref)
                pc3_ref = np.cross(pc1_ref, pc2_ref)
                ref_rot = np.column_stack([pc1_ref, pc2_ref, pc3_ref])

                # --- Transformation
                R = obs_rot @ ref_rot.T
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = target_center - R @ source_center

                # --- Transform reference cloud
                transformed_ref = o3d.geometry.PointCloud()
                transformed_ref.points = o3d.utility.Vector3dVector(ref_points.copy())
                transformed_ref.transform(T)

                # --- Visualization
                # if self.visualize_icp:
                #     coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                #     transformed_ref.paint_uniform_color([1, 0, 1])
                #     o3d.visualization.draw_geometries(
                #         [observed_cloud, transformed_ref, coord],
                #         window_name=f"Initial Config PC{i1}-{i2} flip1={flip1} flip2={flip2}",
                #         zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1]
                #     )

                # --- Run ICP
                threshold = 0.01
                result = o3d.pipelines.registration.registration_icp(
                    transformed_ref, cropped_cloud,
                    threshold,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                )

                config_name = f"PC{i1}-{i2}_flip1={flip1}_flip2={flip2}"
                self.get_logger().info(
                    f"Config {config_name} -> Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f}"
                )

                if result.fitness > best_result["fitness"] or (
                    result.fitness == best_result["fitness"] and result.inlier_rmse < best_result["rmse"]
                ):
                    best_result.update({
                        "fitness": result.fitness,
                        "rmse": result.inlier_rmse,
                        "transformation": result.transformation @ T,
                        "config": config_name,
                        "reference": transformed_ref,
                        "threshold": threshold
                    })

        # --- Point-to-plane refinement
        if best_result["fitness"] > 0.4:
            cropped_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
            best_result["reference"].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))

            refined = o3d.pipelines.registration.registration_icp(
                best_result["reference"], cropped_cloud,
                best_result["threshold"] * 0.8,
                best_result["transformation"],
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )

            self.get_logger().info(f"Refined ICP - Fitness: {refined.fitness:.3f}, RMSE: {refined.inlier_rmse:.5f}")

            if refined.fitness > best_result["fitness"] * 0.9:
                best_result["transformation"] = refined.transformation
                best_result["fitness"] = refined.fitness
                best_result["rmse"] = refined.inlier_rmse

        # --- Final visualization
        if best_result["transformation"] is None:
            self.get_logger().warn("ICP failed to find a good alignment.")
            return None

        self.get_logger().info(
            f"Best config: {best_result['config']} | Fitness: {best_result['fitness']:.3f}, RMSE: {best_result['rmse']:.5f}"
        )

        if self.visualize_icp:
            final_ref = o3d.geometry.PointCloud()
            final_ref.points = o3d.utility.Vector3dVector(ref_points.copy())
            final_ref.transform(best_result["transformation"])
            final_ref.paint_uniform_color([0, 0, 1])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

            o3d.visualization.draw_geometries(
                [cropped_cloud, final_ref, coord],
                window_name="Final ICP Result",
                zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1]
            )

        return best_result["transformation"]



        

        
    def estimate_pose_with_icp2(self, observed_cloud, full_cloud):
        if len(observed_cloud.points) < 10:
            self.get_logger().warn("Not enough points for ICP registration")
            return None

        # Copy reference point cloud
        reference_points = self.processed_ref_points.copy()
        reference_copy = o3d.geometry.PointCloud()
        reference_copy.points = o3d.utility.Vector3dVector(reference_points)

        # Target center
        target_center = observed_cloud.get_center()

        # Crop full cloud for ICP target
        ref_size = np.array(reference_copy.get_max_bound()) - np.array(reference_copy.get_min_bound())
        cropped_cloud = full_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=target_center - ref_size * 1.5,
            max_bound=target_center + ref_size * 1.5
        ))

        # Visual debugging: show PCA of input clouds
        if self.visualize_icp:
            self.get_logger().info(f"Point counts - Observed: {len(observed_cloud.points)}, Cropped: {len(cropped_cloud.points)}")
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            observed_axes = self.draw_pca_axes(observed_cloud, axis_length=0.05)
            cropped_axes = self.draw_pca_axes(cropped_cloud, axis_length=0.05)

            o3d.visualization.draw_geometries(
                [observed_cloud, coord_frame] + observed_axes,
                zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1],
                window_name="Observed Object Cloud with PCA"
            )
            o3d.visualization.draw_geometries(
                [cropped_cloud, coord_frame] + cropped_axes,
                zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1],
                window_name="Cropped Workspace Cloud with PCA"
            )

        # Compute PCA bases
        obs_evals, obs_basis = self.get_pca_basis(np.asarray(observed_cloud.points))
        ref_evals, ref_basis = self.get_pca_basis(reference_points)
        self.get_logger().info(f"Observed eigenvalues: {obs_evals}")
        self.get_logger().info(f"Reference eigenvalues: {ref_evals}")

        configs = [
            {"flip1": 1, "flip2": 1},
            {"flip1": 1, "flip2": -1},
            {"flip1": -1, "flip2": 1},
            {"flip1": -1, "flip2": -1}
        ]
        config_colors = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

        best_result = {
            "fitness": 0.0,
            "rmse": float("inf"),
            "transformation": None,
            "config": "",
            "reference": None,
            "threshold": 0.01
        }

        for i, cfg in enumerate(configs):
            # Flip first two PCA axes
            pc1_obs = obs_basis[:, 0] * cfg["flip1"]
            pc2_obs = obs_basis[:, 1] * cfg["flip2"]
            pc2_obs -= np.dot(pc2_obs, pc1_obs) * pc1_obs
            pc2_obs /= np.linalg.norm(pc2_obs)
            pc3_obs = np.cross(pc1_obs, pc2_obs)
            obs_rot = np.column_stack([pc1_obs, pc2_obs, pc3_obs])

            pc1_ref = ref_basis[:, 0]
            pc2_ref = ref_basis[:, 1]
            pc2_ref -= np.dot(pc2_ref, pc1_ref) * pc1_ref
            pc2_ref /= np.linalg.norm(pc2_ref)
            pc3_ref = np.cross(pc1_ref, pc2_ref)
            ref_rot = np.column_stack([pc1_ref, pc2_ref, pc3_ref])

            R = obs_rot @ ref_rot.T

            rotated_ref = o3d.geometry.PointCloud()
            rotated_ref.points = o3d.utility.Vector3dVector(reference_points.copy())
            source_center = rotated_ref.get_center()
            rotated_ref.rotate(R, center=source_center)
            translation = target_center - source_center
            rotated_ref.translate(translation)

            config_name = f"PCA_flip1={cfg['flip1']}_flip2={cfg['flip2']}"

            # if self.visualize_icp:
            #     rotated_colored = o3d.geometry.PointCloud(rotated_ref)
            #     rotated_colored.paint_uniform_color(config_colors[i])
            #     o3d.visualization.draw_geometries(
            #         [cropped_cloud, rotated_colored, coord_frame],
            #         zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1],
            #         window_name=f"Initial Alignment - {config_name}"
            #     )

            for dist_thresh in [0.02, 0.01, 0.005]:
                result = o3d.pipelines.registration.registration_icp(
                    rotated_ref, cropped_cloud, dist_thresh, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                )

                self.get_logger().info(f"{config_name}, Threshold: {dist_thresh:.3f}m - Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f}")

                # if self.visualize_icp:
                #     post_icp = o3d.geometry.PointCloud(rotated_ref)
                #     post_icp.transform(result.transformation)
                #     post_icp.paint_uniform_color([0, 1, 1])
                #     o3d.visualization.draw_geometries(
                #         [cropped_cloud, post_icp, coord_frame],
                #         zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1],
                #         window_name=f"After ICP - {config_name}, {dist_thresh}m"
                #     )

                if result.fitness > 0.3 and result.inlier_rmse < best_result["rmse"]:
                    best_result.update({
                        "fitness": result.fitness,
                        "rmse": result.inlier_rmse,
                        "transformation": result.transformation,
                        "config": config_name,
                        "reference": rotated_ref,
                        "threshold": dist_thresh
                    })

        if best_result["transformation"] is None:
            self.get_logger().warn("No good alignment found.")
            return None

        self.get_logger().info(f"Best config: {best_result['config']}, Fitness: {best_result['fitness']:.3f}, RMSE: {best_result['rmse']:.5f}")

        # Optional: point-to-plane ICP refinement
        if best_result["fitness"] > 0.4:
            cropped_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
            best_result["reference"].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))

            refined = o3d.pipelines.registration.registration_icp(
                best_result["reference"], cropped_cloud,
                best_result["threshold"] * 0.8,
                best_result["transformation"],
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )

            self.get_logger().info(f"Refined ICP - Fitness: {refined.fitness:.3f}, RMSE: {refined.inlier_rmse:.5f}")

            if refined.fitness > best_result["fitness"] * 0.9:
                best_result["transformation"] = refined.transformation
                best_result["fitness"] = refined.fitness
                best_result["rmse"] = refined.inlier_rmse

        final = o3d.geometry.PointCloud(best_result["reference"])
        final.transform(best_result["transformation"])
        final.paint_uniform_color([0, 0, 1])

        if self.visualize_icp:
            o3d.visualization.draw_geometries(
                [cropped_cloud, final, coord_frame],
                zoom=0.7, front=[0, -1, 0], lookat=target_center, up=[0, 0, 1],
                window_name="Final Alignment Result"
            )

        self.get_logger().info(f"Final transform:\n{best_result['transformation']}")
        return best_result["transformation"]

        
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