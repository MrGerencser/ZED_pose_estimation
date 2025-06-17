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
from std_msgs.msg import Header, Bool
from rclpy.executors import MultiThreadedExecutor
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from zed_pose_estimation.vision_pipeline_utils import crop_point_cloud_gpu, fuse_point_clouds_centroid, subtract_point_clouds_gpu, convert_mask_to_3d_points, downsample_point_cloud_gpu
import open3d as o3d
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R_simple
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import traceback

import threading
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

from zed_pose_estimation.vis2 import visualize_superquadric_grasps

# superquadric fitting imports
from EMS.EMS_recovery import EMS_recovery
from zed_pose_estimation.superquadric_grasp_planner import SuperquadricGraspPlanner

class ZedGpuNode(Node):
    def __init__(self):
        super().__init__('zed_gpu_node')
        
        # Original parameters
        self.declare_parameter('camera1_sn', 33137761)
        self.declare_parameter('camera2_sn', 36829049)
        self.declare_parameter('yolo_model_path', '/home/chris/franka_ros2_ws/src/zed_pose_estimation/models/yolo/new_seg_model.pt')
        # self.declare_parameter('yolo_model_path', 'models/yolo/yolo11x-seg.pt')
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
        self.declare_parameter('poisson_reconstruction', False)  # Default False: use basic pointcloud filtering
        self.declare_parameter('use_kmeans_clustering', False)  # Default False: single global SQ
        self.declare_parameter('superquadric_enabled', True)
        self.declare_parameter('gripper_jaw_length', 0.041)  # meters
        self.declare_parameter('gripper_max_opening', 0.08)   # meters
        self.declare_parameter('outlier_ratio', 0.9)         # EMS parameter
        self.declare_parameter('visualize_yolo', True)
        self.declare_parameter('visualize', True)
        
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
        self.poisson_reconstruction = self.get_parameter('poisson_reconstruction').get_parameter_value().bool_value
        self.use_kmeans_clustering = self.get_parameter('use_kmeans_clustering').get_parameter_value().bool_value
        self.superquadric_enabled = self.get_parameter('superquadric_enabled').get_parameter_value().bool_value
        self.gripper_jaw_length = self.get_parameter('gripper_jaw_length').get_parameter_value().double_value
        self.gripper_max_opening = self.get_parameter('gripper_max_opening').get_parameter_value().double_value
        self.outlier_ratio = self.get_parameter('outlier_ratio').get_parameter_value().double_value
        self.visualize_yolo = self.get_parameter('visualize_yolo').get_parameter_value().bool_value
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

        self.class_names = {0: 'Cone', 1: 'Cup', 2: 'Mallet', 3: 'Screw Driver', 4: 'Sunscreen'}

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
    
        
        # FPS calculation
        self.fps_values = []
        
        # Check if CUDA is available and set the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        
        # Setup TF broadcaster and pose publisher
        self.pose_publisher = self.create_publisher(PoseStamped, '/perception/object_pose', 10)
        
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
        
        # Add execution state tracking
        self.grasp_executing = False
        
        # Subscribe to grasp execution state
        self.execution_state_subscriber = self.create_subscription(
            Bool, '/robot/grasp_executing',
            self.grasp_execution_callback, 10
        )
        
        # Web-based live visualization setup with separate capture
        self.web_enabled = self.visualize_yolo
        self.live_capture_enabled = False
        self.web_server_thread = None
        self.live_capture_thread = None
        self.latest_web_frame = None
        self.web_frame_lock = threading.Lock()
        
        if self.web_enabled:
            self._setup_web_server()
            self._start_live_capture()
        
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
        init_params1.camera_fps = 10
        init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params1.depth_minimum_distance = 0.4
        init_params1.coordinate_units = sl.UNIT.METER
        
        # Set the initialization parameters for camera 2
        init_params2 = sl.InitParameters()
        init_params2.set_from_serial_number(self.camera2_sn)
        init_params2.camera_resolution = sl.RESOLUTION.HD720
        init_params2.camera_fps = 10
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
            
            # Step 1: Capture and validate frames
            frame1, frame2 = self._capture_and_retrieve_frames()
            if frame1 is None or frame2 is None:
                return
        
            # Step 2: Retrieve and process depth maps
            depth_np1, depth_np2 = self._retrieve_depth_maps()
            if depth_np1 is None or depth_np2 is None:
                return
            
            # Step 3: Process point clouds
            pc1_cropped, pc2_cropped, fused_workspace_np, pcd_fused_workspace = self._process_point_clouds()
            if pc1_cropped is None:
                return
                
            # Step 4: Run YOLO detection
            results1, results2, class_ids1, class_ids2 = self._run_yolo_detection([frame1, frame2])
            
            # # Step 5: Update web display immediately after YOLO (non-blocking)
            # if self.web_enabled:
            #     self._update_web_display(frame1, frame2, results1, results2)
            
            # Step 6: Extract object point clouds from detections
            fused_object_points, fused_object_classes, fused_objects_np = self._extract_object_point_clouds(
                results1, results2, class_ids1, class_ids2, depth_np1, depth_np2
            )

            # Step 7: Generate grasps using superquadric fitting
            if self.superquadric_enabled and fused_object_points:
                self._process_superquadric_grasps(fused_object_points, fused_object_classes, fused_workspace_np)
            
            # Step 8: Publish point clouds
            self._publish_point_clouds(fused_workspace_np, fused_objects_np)
            
            # Step 9: Update FPS calculation
            total_time = time.time() - start_time
            self.timings["Total Time"].append(total_time)
            fps = 1.0 / total_time
            self.fps_values.append(fps)
            
            if len(self.fps_values) > 10:
                self.fps_values.pop(0)
                
        except Exception as e:
            self.get_logger().error(f"Error in process_frames: {e}")
            self.get_logger().error(traceback.format_exc())

    def _capture_and_retrieve_frames(self):
        """Capture frames from both cameras and convert them to OpenCV format"""
        try:
            # Update frame count for intrinsics refresh
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 0
                
            if self.frame_count % 100 == 0:
                self.update_camera_intrinsics()
            
            # Check if cameras are ready
            if (self.zed1.grab() != sl.ERROR_CODE.SUCCESS or 
                self.zed2.grab() != sl.ERROR_CODE.SUCCESS):
                self.get_logger().warn("Failed to grab frames from cameras")
                return None, None
            
            # Retrieve frames
            retrieval_start = time.time()
            self.zed1.retrieve_image(self.image1, view=sl.VIEW.LEFT)
            self.zed2.retrieve_image(self.image2, view=sl.VIEW.LEFT)
            
            # Convert to OpenCV format
            frame1 = self.image1.get_data()
            frame2 = self.image2.get_data()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)
            
            retrieval_time = time.time() - retrieval_start
            self.timings["Frame Retrieval"].append(retrieval_time)
            
            return frame1, frame2
            
        except Exception as e:
            self.get_logger().error(f"Error capturing frames: {e}")
            return None, None

    def _retrieve_depth_maps(self):
        """Retrieve depth maps from both cameras"""
        try:
            depth_start = time.time()
            
            depth_result1 = self.zed1.retrieve_measure(self.depth1, measure=sl.MEASURE.DEPTH)
            depth_result2 = self.zed2.retrieve_measure(self.depth2, measure=sl.MEASURE.DEPTH)
            
            if depth_result1 != sl.ERROR_CODE.SUCCESS or depth_result2 != sl.ERROR_CODE.SUCCESS:
                self.get_logger().warn(f"Failed to retrieve depth maps: {depth_result1}, {depth_result2}")
                return None, None
                
            depth_np1 = self.depth1.get_data()
            depth_np2 = self.depth2.get_data()
            
            depth_time = time.time() - depth_start
            self.timings["Depth Retrieval"].append(depth_time)
            
            return depth_np1, depth_np2
            
        except Exception as e:
            self.get_logger().error(f"Error retrieving depth maps: {e}")
            return None, None

    def _process_point_clouds(self):
        """Process point clouds from both cameras and fuse them"""
        try:
            pc_start = time.time()
            
            # Retrieve point clouds
            self.zed1.retrieve_measure(self.point_cloud1_ws, measure=sl.MEASURE.XYZ)
            self.zed2.retrieve_measure(self.point_cloud2_ws, measure=sl.MEASURE.XYZ)
            
            # Convert to torch tensors
            pc1_tensor = torch.tensor(self.point_cloud1_ws.get_data()[:, :, :3], 
                                     dtype=torch.float32, device=self.device).reshape(-1, 3)
            pc2_tensor = torch.tensor(self.point_cloud2_ws.get_data()[:, :, :3], 
                                     dtype=torch.float32, device=self.device).reshape(-1, 3)
            
            # Filter valid points
            valid_mask1 = torch.isfinite(pc1_tensor).all(dim=1)
            valid_mask2 = torch.isfinite(pc2_tensor).all(dim=1)
            pc1_tensor = pc1_tensor[valid_mask1]
            pc2_tensor = pc2_tensor[valid_mask2]
            
            # Transform point clouds to robot frame
            pc1_transformed = torch.mm(pc1_tensor, self.rotation1_torch.T) + self.origin1_torch
            pc2_transformed = torch.mm(pc2_tensor, self.rotation2_torch.T) + self.origin2_torch
            
            # Crop to workspace bounds
            x_bounds = (self.workspace_bounds[0], self.workspace_bounds[1])
            y_bounds = (self.workspace_bounds[2], self.workspace_bounds[3])
            z_bounds = (self.workspace_bounds[4], self.workspace_bounds[5])
            
            pc1_cropped = crop_point_cloud_gpu(pc1_transformed, x_bounds, y_bounds, z_bounds)
            pc2_cropped = crop_point_cloud_gpu(pc2_transformed, x_bounds, y_bounds, z_bounds)
            
            # Fuse workspace point clouds
            fused_workspace = torch.cat((pc1_cropped, pc2_cropped), dim=0)
            fused_workspace_np = fused_workspace.cpu().numpy()
            
            # Create Open3D point cloud
            pcd_fused_workspace = o3d.geometry.PointCloud()
            pcd_fused_workspace.points = o3d.utility.Vector3dVector(fused_workspace_np)
            
            # Optional visualization
            # if self.visualize:
            #     o3d.visualization.draw_geometries([pcd_fused_workspace], window_name="Fused Workspace Point Cloud")
            
            pc_time = time.time() - pc_start
            self.timings["Point Cloud Processing"].append(pc_time)
            
            return pc1_cropped, pc2_cropped, fused_workspace_np, pcd_fused_workspace
            
        except Exception as e:
            self.get_logger().error(f"Error processing point clouds: {e}")
            return None, None, None, None

    def _run_yolo_detection(self, frames):
        """Run YOLO detection on both camera frames"""
        try:
            yolo_start = time.time()
            
            results_batch = self.model.track(
                source=frames,
                classes=[0, 1, 2, 3, 4],  # Cone, Cup, Mallet, Screw Driver, Sunscreen
                persist=True,
                retina_masks=True,
                conf=self.conf_threshold,
                device=self.device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )
            
            results1, results2 = results_batch[0], results_batch[1]
            
            # Extract class IDs
            class_ids1 = (results1.boxes.cls.cpu().numpy() 
                         if results1.boxes is not None else np.array([]))
            class_ids2 = (results2.boxes.cls.cpu().numpy() 
                         if results2.boxes is not None else np.array([]))
            
            yolo_time = time.time() - yolo_start
            self.timings["YOLO Inference"].append(yolo_time)
            
            return results1, results2, class_ids1, class_ids2
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLO detection: {e}")
            return None, None, None, None

    def _extract_object_point_clouds(self, results1, results2, class_ids1, class_ids2, depth_np1, depth_np2):
        """Extract object point clouds from detection masks"""
        try:
            mask_start = time.time()
            
            point_clouds_camera1 = []
            point_clouds_camera2 = []
            
            # Process camera 1 masks
            if results1.masks is not None and results1.masks.data.numel() > 0:
                depth_map1_torch = torch.tensor(depth_np1, dtype=torch.float32, device=self.device)
                
                for i, individual_mask_tensor in enumerate(results1.masks.data):
                    mask_indices = torch.nonzero(individual_mask_tensor, as_tuple=False)

                    if mask_indices.numel() > 0: 
                        with torch.amp.autocast('cuda'):
                            points_3d = convert_mask_to_3d_points(
                                mask_indices, depth_map1_torch, 
                                self.cx1, self.cy1, self.fx1, self.fy1
                            )
                        
                        if points_3d.size(0) > 0:
                            transformed = torch.mm(points_3d, self.rotation1_torch.T) + self.origin1_torch
                            
                            # EARLY WORKSPACE FILTER - CAMERA 1
                            transformed_np = transformed.cpu().numpy()
                            if self._is_object_in_workspace(transformed_np, coverage_threshold=0.3):
                                point_clouds_camera1.append((transformed_np, int(class_ids1[i])))
                            else:
                                object_name = self.class_names.get(int(class_ids1[i]), f'Class_{int(class_ids1[i])}')
                                object_center = np.mean(transformed_np, axis=0)
                                self.get_logger().info(
                                    f"[EARLY FILTER] Rejected {object_name} from camera 1 - "
                                    f"center [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}] outside workspace"
                                )
            
            # Process camera 2 masks
            if results2.masks is not None and results2.masks.data.numel() > 0:
                depth_map2_torch = torch.tensor(depth_np2, dtype=torch.float32, device=self.device)
                
                for i, individual_mask_tensor in enumerate(results2.masks.data):
                    mask_indices = torch.nonzero(individual_mask_tensor, as_tuple=False)

                    if mask_indices.numel() > 0: 
                        with torch.amp.autocast('cuda'):
                            points_3d = convert_mask_to_3d_points(
                                mask_indices, depth_map2_torch,
                                self.cx2, self.cy2, self.fx2, self.fy2
                            )
                        
                        if points_3d.size(0) > 0:
                            transformed = torch.mm(points_3d, self.rotation2_torch.T) + self.origin2_torch
                            
                            # EARLY WORKSPACE FILTER - CAMERA 2
                            transformed_np = transformed.cpu().numpy()
                            if self._is_object_in_workspace(transformed_np, coverage_threshold=0.3):
                                point_clouds_camera2.append((transformed_np, int(class_ids2[i])))
                            else:
                                object_name = self.class_names.get(int(class_ids2[i]), f'Class_{int(class_ids2[i])}')
                                object_center = np.mean(transformed_np, axis=0)
                                self.get_logger().info(
                                    f"[EARLY FILTER] Rejected {object_name} from camera 2 - "
                                    f"center [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}] outside workspace"
                                )
            
            # Fuse object point clouds (now only workspace-valid objects)
            _, _, fused_objects = fuse_point_clouds_centroid(
                point_clouds_camera1, point_clouds_camera2, self.distance_threshold
            )
            
            fused_object_points = [pc for pc, _ in fused_objects]
            fused_object_classes = [cls for _, cls in fused_objects]
            
            fused_objects_np = (np.vstack(fused_object_points) 
                               if fused_object_points else np.empty((0, 3)))
            
            mask_time = time.time() - mask_start
            self.timings["Mask Processing"].append(mask_time)
            
            return fused_object_points, fused_object_classes, fused_objects_np
            
        except Exception as e:
            self.get_logger().error(f"Error extracting object point clouds: {e}")
            return [], [], np.empty((0, 3))

    def _is_object_in_workspace(self, object_points, coverage_threshold=0.5):
        """
        Check if detected object is within workspace bounds
        
        Args:
            object_points: numpy array of object points (Nx3)
            coverage_threshold: minimum fraction of points that must be in workspace (0.0-1.0)
            
        Returns:
            bool: True if object is sufficiently within workspace, False otherwise
        """
        try:
            if len(object_points) == 0:
                return False
            
            # Extract workspace bounds
            x_min, x_max = self.workspace_bounds[0], self.workspace_bounds[1]
            y_min, y_max = self.workspace_bounds[2], self.workspace_bounds[3]
            z_min, z_max = self.workspace_bounds[4], self.workspace_bounds[5]
            
            # Check which points are within workspace bounds
            x_in_bounds = (object_points[:, 0] >= x_min) & (object_points[:, 0] <= x_max)
            y_in_bounds = (object_points[:, 1] >= y_min) & (object_points[:, 1] <= y_max)
            z_in_bounds = (object_points[:, 2] >= z_min) & (object_points[:, 2] <= z_max)
            
            # Points are in workspace if they satisfy all three bounds
            points_in_workspace = x_in_bounds & y_in_bounds & z_in_bounds
            
            # Calculate coverage
            coverage = np.sum(points_in_workspace) / len(object_points)
            
            # Check object center position as well
            object_center = np.mean(object_points, axis=0)
            center_in_workspace = (
                x_min <= object_center[0] <= x_max and
                y_min <= object_center[1] <= y_max and
                z_min <= object_center[2] <= z_max
            )
            
            # Object is valid if either:
            # 1. Center is in workspace AND sufficient point coverage
            # 2. Very high point coverage (even if center is slightly outside)
            is_valid = (center_in_workspace and coverage >= coverage_threshold) or (coverage >= 0.8)
            
            # Log detailed info for debugging
            if not is_valid:
                self.get_logger().debug(
                    f"Object rejected: center=[{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}], "
                    f"coverage={coverage:.3f}, workspace=[{x_min:.3f}-{x_max:.3f}, {y_min:.3f}-{y_max:.3f}, {z_min:.3f}-{z_max:.3f}]"
                )
            
            return is_valid
            
        except Exception as e:
            self.get_logger().error(f"Error in workspace bounds check: {e}")
            return False  # Conservative: reject if check fails

    def _process_superquadric_grasps(self, fused_object_points, fused_object_classes, fused_workspace_np):
        """Process objects for superquadric fitting and grasp generation"""
        try:
            # EARLY EXIT if grasp is executing
            if self.grasp_executing:
                self.get_logger().info("Skipping grasp generation - robot is executing grasp")
                return
            
            grasp_start = time.time()

            graspable_classes = [0, 1, 2, 3, 4]  # Cone, Cup, Mallet, Screw Driver, Sunscreen

            for i, (object_points, class_id) in enumerate(zip(fused_object_points, fused_object_classes)):
                if len(object_points) < 100:
                    self.get_logger().warn(f"Object {i} has too few points ({len(object_points)}) for superquadric fitting")
                    continue
                    
                if class_id in graspable_classes:
                    object_name = self.class_names.get(class_id, f'Class_{class_id}')
                    self.get_logger().info(f"Processing {object_name} (class {class_id}) with superquadric fitting")
                    
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
                            self._publish_best_grasp_pose(grasp_poses, sq_recovered, i, class_id)
                            
                    except Exception as e:
                        self.get_logger().error(f"Error in superquadric grasp generation for {object_name}: {e}")
                        self.get_logger().error(traceback.format_exc())
                else:
                    object_name = self.class_names.get(class_id, f'Class_{class_id}')
                    if class_id == 4:  # Robot
                        self.get_logger().info(f"Skipping {object_name} (class {class_id}) - robot not graspable")
                    else:
                        self.get_logger().info(f"Skipping {object_name} (class {class_id}) - not in graspable classes")
            
            grasp_time = time.time() - grasp_start
            self.timings["Superquadric Grasp Generation"].append(grasp_time)
            
        except Exception as e:
            self.get_logger().error(f"Error in superquadric grasp processing: {e}")

    def _publish_best_grasp_pose(self, grasp_poses, sq_recovered, object_index, class_id):
        """Publish the best grasp pose for a detected object"""
        try:
            # Get the best grasp pose (already in robot frame, already EE position)
            best_grasp = grasp_poses[0]
            
            print(f"Best grasp for object {object_index} (class {class_id}): {best_grasp}")
            
            # Extract position and quaternion from the 4x4 transformation matrix
            pos = best_grasp[:3, 3]  # Extract translation
            rot_matrix = best_grasp[:3, :3]  # Extract rotation matrix
            
            # Convert rotation matrix to quaternion directly - NO EXTRA TRANSFORMATION NEEDED
            quat = R_simple.from_matrix(rot_matrix).as_quat()  # Returns [x, y, z, w]
            euler = R_simple.from_matrix(rot_matrix).as_euler('xyz')
            
            print(f"Best grasp position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"Best grasp euler angles: [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]")
            print(f"Best grasp quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
            
            # Create and publish pose message
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.target_frame
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2]) 
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])

            # Publish the grasp pose
            self.pose_publisher.publish(pose_msg)
            self.get_logger().info(f"Published grasp pose at: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    
        except Exception as e:
            self.get_logger().error(f"Error publishing grasp pose: {e}")

    def _publish_point_clouds(self, fused_workspace_np, fused_objects_np):
        """Publish all point clouds as ROS messages"""
        try:
            # Create header for point cloud messages
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.target_frame
            
            # Publish the point clouds
            if fused_workspace_np.size > 0:
                self.fused_workspace_publisher.publish(pc2.create_cloud_xyz32(header, fused_workspace_np))
                
            if fused_objects_np.size > 0:
                self.fused_objects_publisher.publish(pc2.create_cloud_xyz32(header, fused_objects_np))
                
                
        except Exception as e:
            self.get_logger().error(f"Error publishing point clouds: {e}")

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
    
    def _update_visualization_and_fps(self, frame1, frame2, results1, results2, start_time):
        """Update visualization display and calculate FPS"""
        try:
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
                    
                    # Draw detection boxes on frame 1
                    if results1[0].boxes is not None and len(results1[0].boxes) > 0:
                        for box in results1[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(display1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw detection boxes on frame 2
                    if results2[0].boxes is not None and len(results2[0].boxes) > 0:
                        for box in results2[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(display2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add FPS text
                    cv2.putText(display1, f"FPS: {avg_fps:.1f}", (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Resize and combine displays
                    display1 = cv2.resize(display1, (640, 360))
                    display2 = cv2.resize(display2, (640, 360))
                    display = np.hconcat([display1, display2])
                    
                    # Show display and check for quit
                    cv2.imshow("YOLO Detection", display)
                    key = cv2.waitKey(1)
                    
                    if key == ord('q'):
                        self.get_logger().info("User requested shutdown (Q key pressed)")
                        self.future_shutdown = True
                        
                except Exception as viz_error:
                    self.get_logger().error(f"Visualization error: {viz_error}")
                    self.get_logger().error(traceback.format_exc())
                    
        except Exception as e:
            self.get_logger().error(f"Error updating visualization and FPS: {e}")
    

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
            K = 6
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
        """Initialize K+1 superquadrics using the paper's exact method"""
        from sklearn.cluster import KMeans
        
        if K is None:
            K = self.calculate_k_superquadrics(len(points_for_ems))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=42)
        cluster_labels = kmeans.fit_predict(points_for_ems)
        
        initial_superquadrics = []
        
        # Initialize K superquadrics from clusters
        for i in range(K):
            cluster_points = points_for_ems[cluster_labels == i]
            if len(cluster_points) > 10:
                # Calculate MoI for this cluster
                cluster_moi = self.calculate_moment_of_inertia(cluster_points)
                
                # Initialize ellipsoid with MoI = cluster_MoI / 2
                ellipsoid_params = self.moi_to_ellipsoid_params(cluster_moi / 2.0)
                
                initial_superquadrics.append({
                    'translation': np.mean(cluster_points, axis=0),
                    'scale': ellipsoid_params['scale'],
                    'shape': [1.0, 1.0],  # Ellipsoid shape parameters
                    'rotation': ellipsoid_params['rotation'],
                    'type': 'cluster'
                })
        
        # Add extra SQ for whole object (K+1)
        global_moi = self.calculate_moment_of_inertia(points_for_ems)
        global_ellipsoid = self.moi_to_ellipsoid_params(global_moi / 2.0)
        
        initial_superquadrics.append({
            'translation': np.mean(points_for_ems, axis=0),
            'scale': global_ellipsoid['scale'],
            'shape': [1.0, 1.0],
            'rotation': global_ellipsoid['rotation'],
            'type': 'global'
        })
        
        return initial_superquadrics

    def calculate_moment_of_inertia(self, points):
        """Calculate moment of inertia tensor for point cloud"""
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        # Calculate inertia tensor
        I = np.zeros((3, 3))
        for p in centered_points:
            I += np.outer(p, p)
        I /= len(points)
        
        return I

    def moi_to_ellipsoid_params(self, inertia_tensor):
        """Convert moment of inertia to ellipsoid parameters with proper rotation matrix"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(inertia_tensor)
            
            # Ensure eigenvalues are positive
            eigenvals = np.abs(eigenvals)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Avoid zero eigenvalues
            
            # Scale from eigenvalues (semi-axes of ellipsoid)
            scale = np.sqrt(eigenvals)
            
            # Ensure proper rotation matrix (right-handed, determinant = +1)
            rotation_matrix = eigenvecs.copy()
            
            # Check if the matrix is left-handed (negative determinant)
            if np.linalg.det(rotation_matrix) < 0:
                # Fix by flipping one column (usually the last one)
                rotation_matrix[:, -1] = -rotation_matrix[:, -1]
                self.get_logger().debug("Fixed left-handed rotation matrix by flipping last column")
            
            # Double-check that we now have a proper rotation matrix
            det = np.linalg.det(rotation_matrix)
            if abs(det - 1.0) > 1e-6:
                self.get_logger().warn(f"Rotation matrix determinant: {det}, orthogonalizing...")
                # Use SVD to get the closest proper rotation matrix
                U, _, Vt = np.linalg.svd(rotation_matrix)
                rotation_matrix = U @ Vt
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, -1] = -rotation_matrix[:, -1]
            
            return {
                'scale': scale,
                'rotation': rotation_matrix
            }
            
        except Exception as e:
            self.get_logger().error(f"Error in moi_to_ellipsoid_params: {e}")
            # Return identity as fallback
            return {
                'scale': np.array([0.01, 0.01, 0.01]),
                'rotation': np.eye(3)
            }


    def fit_multiple_superquadrics_ensemble(self, points_for_ems, K=None):
        """Implement the EXACT paper methodology"""
        
        try:
            # Add point cloud preprocessing and validation
            if len(points_for_ems) < 100:
                self.get_logger().warn(f"Too few points ({len(points_for_ems)}) for multi-superquadric fitting")
                return None, []
            
            # Normalize/center the point cloud to improve numerical stability
            points_center = np.mean(points_for_ems, axis=0)
            points_std = np.std(points_for_ems, axis=0)
            
            # Check for degenerate cases
            if np.any(points_std < 1e-6):
                self.get_logger().warn("Degenerate point cloud (very small std dev)")
                return None, []
            
            self.get_logger().info(f"Point cloud stats: center={points_center}, std={points_std}")
            
            # BRANCH: Choose between single SQ or K-means based on parameter
            if not self.use_kmeans_clustering:
                # SIMPLIFIED: Just fit ONE superquadric to the entire point cloud
                self.get_logger().info("Using single global superquadric (no K-means clustering)")
                
                # Center the points for EMS
                centered_points = points_for_ems - points_center
                
                # Additional validation before EMS
                point_span = np.max(centered_points, axis=0) - np.min(centered_points, axis=0)
                if np.any(point_span < 1e-6):
                    self.get_logger().warn("Degenerate point span, cannot fit superquadric")
                    return None, []
                
                self.get_logger().info(f"Centered points span: {point_span}")
                
                try:
                    # Single EMS call for the entire point cloud
                    self.get_logger().info("Fitting single superquadric to entire point cloud...")
                    sq_candidate, probabilities = EMS_recovery(
                        centered_points,
                        # Let EMS use its default initialization
                        OutlierRatio=self.outlier_ratio,
                        MaxIterationEM=60,
                        ToleranceEM=1e-3,
                        RelativeToleranceEM=1e-1,
                        MaxOptiIterations=5,
                        Sigma=0.02,
                        MaxiSwitch=1,
                        AdaptiveUpperBound=True,
                        Rescale=True
                    )
                    
                    # Translate back to world coordinates
                    sq_candidate.translation = sq_candidate.translation + points_center
                    
                    # Validate the result
                    if self._validate_superquadric_strict(sq_candidate, points_for_ems):
                        fitted_superquadrics = [sq_candidate]
                        
                        coverage = self._evaluate_sq_coverage(sq_candidate, points_for_ems)
                        self.get_logger().info(f"SUCCESS: Single SQ coverage={coverage:.3f}")
                        self.get_logger().info(f"  Final SQ: Center={sq_candidate.translation}, Scale={sq_candidate.scale}")
                        
                        return sq_candidate, fitted_superquadrics
                    else:
                        self.get_logger().warn("Single superquadric failed validation")
                        return None, []
                        
                except Exception as e:
                    self.get_logger().error(f"Failed to fit single superquadric: {e}")
                    return None, []
            
            else:
                # ORIGINAL K-MEANS METHOD: Keep all existing code exactly the same
                if K is None:
                    K = self.calculate_k_superquadrics(len(points_for_ems))
                
                self.get_logger().info(f"Using K-means clustering with K={K} clusters")
            
                # K-means clustering to get {Φ1, Φ2, ..., ΦK}
                kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(points_for_ems)

                # STEP 2: Initialize K+1 ellipsoids with improved validation
                initial_ellipsoids = []
                
                # For each subset Φi, initialize ellipsoid θi with validation
                for i in range(K):
                    cluster_points = points_for_ems[cluster_labels == i]
                    
                    # Validate cluster size and properties
                    if len(cluster_points) < 50:  # Increased minimum threshold
                        self.get_logger().warn(f"Cluster {i} too small ({len(cluster_points)} points), skipping")
                        continue
                    
                    cluster_center = np.mean(cluster_points, axis=0)
                    cluster_std = np.std(cluster_points, axis=0)
                    
                    # Check cluster validity
                    if np.any(cluster_std < 1e-6):
                        self.get_logger().warn(f"Cluster {i} degenerate (std={cluster_std}), skipping")
                        continue
                    
                    # In the cluster processing loop, replace the MoI calculation section:
                    try:
                        # Calculate MoI for this cluster subset Φi
                        cluster_moi = self.calculate_moment_of_inertia(cluster_points)
                        
                        # Validate MoI eigenvalues
                        eigenvals, eigenvecs = np.linalg.eigh(cluster_moi)
                        if np.any(eigenvals <= 1e-12):
                            self.get_logger().warn(f"Cluster {i} invalid MoI eigenvals={eigenvals}, using default")
                            # Use default initialization based on cluster bounds
                            cluster_bounds = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                            scale = cluster_bounds / 4.0  # Conservative scale
                            rotation_matrix = np.eye(3)  # Identity rotation
                        else:
                            # Initialize ellipsoid with MoI = cluster_MoI / 2 (EXACT as paper)
                            ellipsoid_params = self.moi_to_ellipsoid_params(cluster_moi / 2.0)
                            scale = ellipsoid_params['scale']
                            rotation_matrix = ellipsoid_params['rotation']
                        
                        # Validate and clamp scale values
                        scale = np.clip(scale, 0.005, 0.2)  # Reasonable bounds: 5mm to 20cm
                        
                        initial_ellipsoids.append({
                            'translation': cluster_center,
                            'scale': scale,
                            'shape': [1.0, 1.0],  # Ellipsoid (ε1=1, ε2=1)
                            'rotation': rotation_matrix,
                            'subset': cluster_points,
                            'type': f'cluster_{i}'
                        })
                        
                        self.get_logger().info(f"Cluster {i}: center={cluster_center}, scale={scale}, det={np.linalg.det(rotation_matrix):.6f}")
                        
                    except Exception as cluster_error:
                        self.get_logger().warn(f"Failed to initialize cluster {i}: {cluster_error}")
                        continue
                
                # STEP 3: Add extra ellipsoid θextra for whole point set X with validation
                try:
                    global_center = np.mean(points_for_ems, axis=0)
                    
                    # Calculate MoI for the ENTIRE point set X
                    global_moi = self.calculate_moment_of_inertia(points_for_ems)
                    
                    # Initialize ellipsoid with MoI = global_MoI / 2 (EXACT as paper)
                    global_ellipsoid_params = self.moi_to_ellipsoid_params(global_moi / 2.0)
                    global_scale = global_ellipsoid_params['scale']
                    
                    # Validate and clamp scale values (optional safety)
                    global_scale = np.clip(global_scale, 0.01, 0.3)
                    
                    initial_ellipsoids.append({
                        'translation': global_center,
                        'scale': global_scale,
                        'shape': [1.0, 1.0],
                        'rotation': global_ellipsoid_params['rotation'],  # Use MoI-derived rotation
                        'subset': points_for_ems,
                        'type': 'global_extra'
                    })
                    
                    self.get_logger().info(f"Global ellipsoid (MoI/2): center={global_center}, scale={global_scale}")
                    
                except Exception as global_error:
                    self.get_logger().warn(f"Failed to initialize global ellipsoid: {global_error}")
                
                if not initial_ellipsoids:
                    self.get_logger().error("No valid initial ellipsoids created")
                    return None, []
            
            self.get_logger().info(f"Initialized {len(initial_ellipsoids)} valid ellipsoids")
            
            # STEP 4: Sequential processing with improved error handling
            fitted_superquadrics = []
            
            for i, ellipsoid_init in enumerate(initial_ellipsoids):
                try:
                    self.get_logger().info(f"Processing ellipsoid {i+1}/{len(initial_ellipsoids)} ({ellipsoid_init['type']})")
                    
                    subset_points = ellipsoid_init['subset']
                    init_translation = ellipsoid_init['translation']
                    init_scale = ellipsoid_init['scale']
                    
                    # Center the points for EMS (this often helps with bounds issues)
                    centered_points = subset_points - init_translation
                    
                    # Additional validation before EMS
                    point_span = np.max(centered_points, axis=0) - np.min(centered_points, axis=0)
                    if np.any(point_span < 1e-6):
                        self.get_logger().warn(f"Ellipsoid {i+1}: degenerate point span, skipping")
                        continue
                    
                    self.get_logger().info(f"  Centered points span: {point_span}")
                    self.get_logger().info(f"  Init scale: {init_scale}")
                    
                    
                    # Convert rotation matrix to Euler angles for EMS
                    init_euler = R_simple.from_matrix(ellipsoid_init['rotation']).as_euler('xyz')

                    sq_candidate, probabilities = EMS_recovery(
                        centered_points,
                        # MoI/2 ellipsoid initialization
                        # InitialShape=ellipsoid_init['shape'],        # [1.0, 1.0] for ellipsoid
                        # InitialScale=init_scale,                     # MoI/2 derived scale
                        # InitialEuler=init_euler,                     # MoI/2 derived rotation
                        # InitialTranslation=np.zeros(3),             # Already centered
                        # Regular EMS parameters
                        OutlierRatio=self.outlier_ratio,
                        MaxIterationEM=60,
                        ToleranceEM=1e-3,
                        RelativeToleranceEM=1e-1,
                        MaxOptiIterations=5,
                        Sigma=0.02,
                        MaxiSwitch=1,
                        AdaptiveUpperBound=True,
                        Rescale=True
                    )
                    
                    # Translate back to world coordinates
                    sq_candidate.translation = sq_candidate.translation + init_translation
                    
                    # Validate the result
                    if self._validate_superquadric_strict(sq_candidate, subset_points):
                        fitted_superquadrics.append(sq_candidate)
                        
                        subset_coverage = np.sum(probabilities > 0.3) / len(centered_points)
                        full_coverage = self._evaluate_sq_coverage(sq_candidate, subset_points)

                        self.get_logger().info(f"  SUCCESS: Subset coverage={subset_coverage:.3f}, "
                                            f"Full coverage={full_coverage:.3f}")
                        self.get_logger().info(f"    Final SQ: Center={sq_candidate.translation}, Scale={sq_candidate.scale}")
                    else:
                        self.get_logger().warn(f"  REJECTED: Failed validation")
                    
                except Exception as e:
                    self.get_logger().warn(f"Failed to process ellipsoid {i+1}: {e}")
                    continue
            
            if not fitted_superquadrics:
                self.get_logger().error("All ellipsoid initializations failed")
                return None, []
            
            # SELECT BEST SQ based on coverage
            best_sq = None
            best_coverage = 0.0
            
            for i, sq in enumerate(fitted_superquadrics):
                coverage = self._evaluate_sq_coverage(sq, points_for_ems)
                self.get_logger().info(f"SQ {i+1}: coverage = {coverage:.3f}")
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_sq = sq
            
            self.get_logger().info(f"Successfully fitted {len(fitted_superquadrics)} superquadrics")
            return best_sq, fitted_superquadrics

        except Exception as e:
            self.get_logger().error(f"Error in ensemble method: {e}")
            return None, []

    def _validate_superquadric_strict(self, sq, points):
        """Stricter validation to catch problematic superquadrics"""
        try:
            # Check scale bounds more strictly
            if np.any(sq.scale < 0.003) or np.any(sq.scale > 0.5):
                return False
            
            # Check shape parameters
            if np.any(np.array(sq.shape) < 0.1) or np.any(np.array(sq.shape) > 4.0):
                return False
            
            # Check that translation is within reasonable bounds of the data
            point_center = np.mean(points, axis=0)
            point_bounds = np.max(points, axis=0) - np.min(points, axis=0)
            max_deviation = np.linalg.norm(point_bounds) * 0.5
            
            if np.linalg.norm(sq.translation - point_center) > max_deviation:
                return False
            
            return True
            
        except Exception:
            return False


    def _evaluate_sq_coverage(self, sq, points):
        """Evaluate how well a superquadric covers the full point cloud"""
        try:
            # Transform points to superquadric coordinate system
            centered_points = points - sq.translation
            
            # Rotate points to align with superquadric axes
            if hasattr(sq, 'RotM'):
                rotated_points = centered_points @ sq.RotM.T
            else:
                rotated_points = centered_points
            
            # Calculate superquadric inside/outside function
            # F(x,y,z) = ((x/a1)^(2/ε2) + (y/a2)^(2/ε2))^(ε2/ε1) + (z/a3)^(2/ε1)
            
            # Avoid division by zero
            safe_scale = np.maximum(sq.scale, 1e-6)
            
            # Normalized coordinates
            normalized = np.abs(rotated_points) / safe_scale
            
            # Shape parameters (epsilon values)
            eps1, eps2 = sq.shape[0], sq.shape[1]
            eps1 = max(eps1, 0.1)  # Avoid numerical issues
            eps2 = max(eps2, 0.1)
            
            # Superquadric function evaluation
            xy_term = (normalized[:, 0]**(2/eps2) + normalized[:, 1]**(2/eps2))**(eps2/eps1)
            z_term = normalized[:, 2]**(2/eps1)
            F_values = xy_term + z_term
            
            # Points are "inside" the superquadric if F <= 1
            # Use a slightly larger threshold for coverage
            inside_mask = F_values <= 1.2
            coverage = np.sum(inside_mask) / len(points)
            
            return coverage
            
        except Exception as e:
            self.get_logger().warn(f"Error evaluating SQ coverage: {e}")
            return 0.0
    
    
    def superquadric_to_open3d_mesh(self, x, threshold=1e-2, num_limit=10000, arclength=0.2):
        """Convert superquadric to Open3D mesh"""
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
            Tuple[List[np.ndarray], Any]: (final_grasp_poses, best_sq) or []
        """
        try:
            # =============================================================================
            # STEP 1: PREPROCESS POINT CLOUD
            # =============================================================================
            self.get_logger().info(f"Processing object with {len(object_points)} points")
            
            # Store original points BEFORE filtering
            original_object_points = np.array(object_points)  # Keep reference to original
            
            # Create and filter point cloud
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            object_pcd, outlier_indices = object_pcd.remove_statistical_outlier(
                nb_neighbors=20, 
                std_ratio=1.0
            )

            # Stage 2: Remove radius-based outliers
            object_pcd, _ = object_pcd.remove_radius_outlier(
                nb_points=16,      # Minimum neighbors within radius
                radius=0.05      # Tight radius for noise removal
            )

            object_pcd = object_pcd.voxel_down_sample(voxel_size=0.002)  # Downsample for efficiency

            if self.poisson_reconstruction:
                # =============================================================================
                # STEP 3: Estimate normals before Poisson reconstruction
                self.get_logger().info("Estimating surface normals for Poisson reconstruction...")
                object_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.01,  # Search radius for normal estimation
                        max_nn=30     # Maximum number of neighbors
                    )
                )

                # Orient normals consistently (important for Poisson reconstruction)
                object_pcd.orient_normals_consistent_tangent_plane(k=15)

                # STEP 4: Mesh reconstruction and smoothing
                self.get_logger().info("Performing Poisson surface reconstruction...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    object_pcd, depth=8, width=0, scale=1.1, linear_fit=False
                )

                # Optional: Remove low-density vertices (often noise)
                if len(densities) > 0:
                    density_threshold = np.quantile(densities, 0.01)  # Remove bottom 1%
                    vertices_to_remove = np.where(np.array(densities) < density_threshold)[0]
                    mesh.remove_vertices_by_index(vertices_to_remove)

                # Smooth the mesh
                mesh = mesh.filter_smooth_simple(number_of_iterations=1)

                # STEP 5: Sample back to point cloud
                self.get_logger().info("Sampling points from reconstructed mesh...")
                object_pcd = mesh.sample_points_poisson_disk(
                    number_of_points=len(object_pcd.points))
            
            # Validate sufficient points remain
            if len(object_pcd.points) < 100:
                self.get_logger().warn(f"Too few points after preprocessing: {len(object_pcd.points)}")
                return []
            
            # Get filtered points
            filtered_object_points = np.asarray(object_pcd.points)
            object_center = np.mean(filtered_object_points, axis=0)
            self.get_logger().info(f"Object center: {object_center}")
            
            # Save temporary file for grasp planner
            temp_file = f"/tmp/object_points_{class_id}_{int(time.time())}.ply"
            o3d.io.write_point_cloud(temp_file, object_pcd)
            
            # =============================================================================
            # STEP 2: CORRECTED VISUALIZATION (TRUE BEFORE/AFTER)
            # =============================================================================
            if self.visualize:
                # Show TRUE before/after comparison
                original_pcd = o3d.geometry.PointCloud()
                original_pcd.points = o3d.utility.Vector3dVector(original_object_points)  # Original noisy points
                original_pcd.paint_uniform_color([1, 0, 0])  # Red for original
                
                filtered_pcd = o3d.geometry.PointCloud()
                filtered_pcd.points = o3d.utility.Vector3dVector(filtered_object_points)  # Filtered points
                filtered_pcd.paint_uniform_color([0, 1, 0])  # Green for filtered
                
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                coord_frame.translate(object_center)
                
                # Show both point clouds together
                o3d.visualization.draw_geometries(
                    [original_pcd, filtered_pcd, coord_frame],
                    window_name=f"Before/After Filtering - Class {class_id} (Original: {len(original_object_points)}, Filtered: {len(filtered_object_points)})",
                    zoom=0.7, front=[0, -1, 0], lookat=object_center, up=[0, 0, 1]
                )
            
            # =============================================================================
            # STEP 3: FIT MULTIPLE SUPERQUADRICS (use filtered points)
            # =============================================================================
            self.get_logger().info("Fitting multiple superquadrics...")
            points_for_ems = filtered_object_points  # Use filtered points for EMS
            
            best_sq, all_valid_sqs = self.fit_multiple_superquadrics_ensemble(points_for_ems)
            
            if not all_valid_sqs:
                self.get_logger().error("Multi-superquadric fitting failed")
                return []
            
            self.get_logger().info(f"Successfully fitted {len(all_valid_sqs)} superquadrics")
            
            # Optional superquadric fitting visualization
            if self.visualize:
                self.visualize_hierarchical_multiquadric_fit(points_for_ems, all_valid_sqs)
            
            # =============================================================================
            # STEP 4: GENERATE GRASPS FROM ALL SUPERQUADRICS
            # =============================================================================
            execution_grasps = {'poses': [], 'info': []}
            visualization_grasps = {'poses': [], 'info': []}
            
            for i, sq_recovered in enumerate(all_valid_sqs):
                try:
                    self.get_logger().info(f"Processing SQ {i+1}/{len(all_valid_sqs)}")
                    self.get_logger().info(f"  Shape=({sq_recovered.shape[0]:.3f}, {sq_recovered.shape[1]:.3f}), "
                                        f"Scale=({sq_recovered.scale[0]:.3f}, {sq_recovered.scale[1]:.3f}, {sq_recovered.scale[2]:.3f})")
                    
                    consistent_euler = R_simple.from_matrix(sq_recovered.RotM).as_euler('xyz')
                    
                    # Generate grasps with intelligent selection
                    all_sq_grasps, best_sq_grasp = self.grasp_planner.plan_grasps_with_best_selection(
                        temp_file, sq_recovered.shape, sq_recovered.scale,
                        consistent_euler, sq_recovered.translation
                    )

                    # Use the best_sq_grasp for execution and all_sq_grasps for visualization:
                    if best_sq_grasp:
                        execution_grasps['poses'].append(best_sq_grasp['pose'])
                        execution_grasps['info'].append({
                            'sq_index': i,
                            'sq': sq_recovered,
                            'euler': consistent_euler,
                            'grasp_score': best_sq_grasp['score'],
                            'selection_info': best_sq_grasp.get('selection_info', {}),
                            'is_visualization': False
                        })
                        
                        self.get_logger().info(f"  Selected BEST grasp with score: {best_sq_grasp['score']:.8f}")
                        if 'selection_info' in best_sq_grasp:
                            info = best_sq_grasp['selection_info']
                            self.get_logger().info(f"    Energy: {info['total_energy']:.4f}, Rank: {info['rank']}")
                    else:
                        self.get_logger().warn(f"  No best grasp selected for SQ {i+1}")

                    # Store all grasps for visualization
                    for grasp_data in all_sq_grasps:
                        visualization_grasps['poses'].append(grasp_data['pose'])
                        visualization_grasps['info'].append({
                            'sq_index': i,
                            'sq': sq_recovered,
                            'euler': consistent_euler,
                            'grasp_score': grasp_data.get('score', 0.0),
                            'is_visualization': True
                        })

                    self.get_logger().info(f"  Generated {len(all_sq_grasps)} total grasps for visualization")
                        
                except Exception as sq_error:
                    self.get_logger().error(f"Error processing SQ {i+1}: {sq_error}")
            
            # Validate we have execution grasps
            if not execution_grasps['poses']:
                self.get_logger().error("No execution grasps generated from any superquadric")
                return []
            
            # =============================================================================
            # STEP 5: SCORE AND RANK ALL GRASPS
            # =============================================================================
            self.get_logger().info(f"Ranking {len(execution_grasps['poses'])} grasps...")
            
            scored_grasps = []
            for grasp_pose, sq_info in zip(execution_grasps['poses'], execution_grasps['info']):
                # Base grasp score from planner
                grasp_score = sq_info.get('grasp_score', 0.0)
                
                # Small superquadric quality bonus
                sq_volume = np.prod(sq_info['sq'].scale)
                sq_quality_bonus = min(0.1, sq_volume / 0.001)
                composite_score = grasp_score + sq_quality_bonus * 0.1
                
                scored_grasps.append((composite_score, grasp_pose, sq_info))
            
            # Sort by score (best first)
            scored_grasps.sort(key=lambda x: x[0], reverse=True)
            
            # Log top scores
            self.get_logger().info("Top 5 grasps:")
            for i, (score, pose, sq_info) in enumerate(scored_grasps[:5]):
                sq_idx = sq_info['sq_index']
                grasp_score = sq_info.get('grasp_score', 0.0)
                pos = pose[:3, 3]
                self.get_logger().info(
                    f"  Rank {i+1}: SQ{sq_idx+1}, grasp_score={grasp_score:.8f}, "
                    f"final_score={score:.8f}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                )
            
            # =============================================================================
            # STEP 6: SELECT DIVERSE GRASPS
            # =============================================================================
            final_grasp_poses = []
            used_sq_indices = set()
            max_grasps = 15
            
            # First pass: best grasp from each superquadric
            for score, grasp_pose, sq_info in scored_grasps:
                sq_index = sq_info['sq_index']
                if sq_index not in used_sq_indices:
                    final_grasp_poses.append(grasp_pose)
                    used_sq_indices.add(sq_index)
                    grasp_score = sq_info.get('grasp_score', 0.0)
                    self.get_logger().info(
                        f"Selected best from SQ {sq_index+1} "
                        f"(grasp_score: {grasp_score:.8f}, final_score: {score:.8f})"
                    )
            
            # Second pass: additional high-scoring grasps
            for score, grasp_pose, sq_info in scored_grasps:
                if len(final_grasp_poses) >= max_grasps:
                    break
                
                # Check for duplicates
                is_duplicate = any(
                    np.allclose(grasp_pose, existing, atol=1e-6) 
                    for existing in final_grasp_poses
                )
                
                if not is_duplicate:
                    final_grasp_poses.append(grasp_pose)
                    grasp_score = sq_info.get('grasp_score', 0.0)
                    sq_index = sq_info['sq_index']
                    self.get_logger().info(
                        f"Added additional from SQ {sq_index+1} "
                        f"(grasp_score: {grasp_score:.8f}, final_score: {score:.8f})"
                    )
            
            # =============================================================================
            # STEP 7: LOG RESULTS AND VISUALIZE
            # =============================================================================
            self.get_logger().info(
                f"Final selection: {len(final_grasp_poses)} grasps from {len(used_sq_indices)} superquadrics"
            )
            
            # Log best grasp
            if final_grasp_poses and scored_grasps:
                best_score_info = scored_grasps[0]
                best_grasp_score = best_score_info[2].get('grasp_score', 0.0)
                best_final_score = best_score_info[0]
                best_sq_index = best_score_info[2]['sq_index']
                
                self.get_logger().info(
                    f"BEST GRASP: From SQ {best_sq_index+1}, "
                    f"grasp_score={best_grasp_score:.8f}, "
                    f"final_score={best_final_score:.8f}"
                )
            
            # Cleanup temporary file
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                self.get_logger().warn(f"Failed to cleanup temp file {temp_file}: {cleanup_error}")

            # Visualize using the planner's method (now has access to proper S objects)
            if self.visualize and visualization_grasps['poses']:
                try:
                    # Convert node superquadrics to planner format
                    planner_superquadrics = []
                    for sq in all_valid_sqs:
                        # Convert from node format to planner format
                        from zed_pose_estimation.superquadric_grasp_planner import Superquadric
                        
                        euler_angles = R_simple.from_matrix(sq.RotM).as_euler('xyz')

                        planner_sq = Superquadric(
                            ε=sq.shape,         # [ε1, ε2]
                            a=sq.scale,         # [ax, ay, az]
                            euler=euler_angles,  # [roll, pitch, yaw]
                            t=sq.translation    # [tx, ty, tz]
                        )
                        planner_superquadrics.append(planner_sq)
                    
                    # Prepare grasp data for multi-SQ visualization
                    all_grasp_visualization_data = []
                    final_pose_set = set()
                    
                    # Mark final poses for easy lookup
                    for pose in final_grasp_poses:
                        pose_key = tuple(pose.flatten())
                        final_pose_set.add(pose_key)
                    
                    # Convert visualization grasps to required format
                    for pose, sq_info in zip(visualization_grasps['poses'], visualization_grasps['info']):
                        pose_key = tuple(pose.flatten())
                        is_final = pose_key in final_pose_set
                        
                        all_grasp_visualization_data.append({
                            'pose': pose,
                            'sq_index': sq_info['sq_index'],
                            'score': sq_info.get('grasp_score', 0.0),
                            'is_final': is_final
                        })
                    
                    # 1. Show ALL generated grasps using planner's method
                    self.get_logger().info(
                        f"Visualizing ALL {len(all_grasp_visualization_data)} grasps "
                        f"from {len(planner_superquadrics)} superquadrics"
                    )
                    self.grasp_planner.visualize_multi_sq_grasps(
                        points_for_ems,
                        planner_superquadrics,  # Use converted S objects
                        all_grasp_visualization_data,
                        window_name=f"ALL Generated Grasps - Class {class_id}",
                        highlight_final=False,  # Show all grasps equally
                        final_count=len(final_grasp_poses)
                    )
                    
                    # 2. Show ONLY final selected grasps
                    final_grasp_visualization_data = []
                    for i, pose in enumerate(final_grasp_poses):
                        # Find the corresponding sq_info for this pose
                        pose_key = tuple(pose.flatten())
                        for viz_pose, sq_info in zip(visualization_grasps['poses'], visualization_grasps['info']):
                            if tuple(viz_pose.flatten()) == pose_key:
                                final_grasp_visualization_data.append({
                                    'pose': pose,
                                    'sq_index': sq_info['sq_index'],
                                    'score': sq_info.get('grasp_score', 0.0),
                                    'is_final': True,
                                    'rank': i + 1  # Add ranking information
                                })
                                break
                    
                    self.get_logger().info(
                        f"Visualizing FINAL {len(final_grasp_visualization_data)} selected grasps"
                    )
                    self.grasp_planner.visualize_multi_sq_grasps(
                        points_for_ems,
                        planner_superquadrics,
                        final_grasp_visualization_data,
                        window_name=f"FINAL Selected Grasps - Class {class_id}",
                        highlight_final=True,  # Highlight the best (rank 1) grasp
                        final_count=len(final_grasp_poses)
                    )
                    
                except Exception as viz_error:
                    self.get_logger().error(f"Visualization error: {viz_error}")
            
            return final_grasp_poses, best_sq
            
        except Exception as e:
            self.get_logger().error(f"Error in superquadric grasp generation: {e}")
            self.get_logger().error(traceback.format_exc())
            return []
        
    def visualize_multi_superquadric_grasps(self, points_for_ems, all_valid_sqs, all_grasp_poses, all_sq_info, 
                                           window_name="Multi-Superquadric Grasps", highlight_best=False):
        """Visualize all superquadrics and their corresponding grasps"""
        try:
            # Ensure points_for_ems is a proper numpy array with correct dtype
            if not isinstance(points_for_ems, np.ndarray):
                points_for_ems = np.array(points_for_ems)
            
            # Ensure correct shape and dtype
            if points_for_ems.ndim != 2 or points_for_ems.shape[1] != 3:
                self.get_logger().error(f"Invalid point cloud shape: {points_for_ems.shape}, expected (N, 3)")
                return
                
            # Ensure float64 dtype (Open3D prefers this)
            points_for_ems = points_for_ems.astype(np.float64)
            
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
            for idx, (grasp_pose, sq_info) in enumerate(zip(all_grasp_poses, all_sq_info)):
                try:
                    sq_index = sq_info['sq_index']
                    color = colors[sq_index % len(colors)]
                    
                    # Different visualization for best grasp vs others
                    if highlight_best and idx == 0:
                        # BEST GRASP: Larger coordinate frame and sphere
                        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
                        sphere_radius = 0.008
                        sphere_color = [1.0, 0.0, 0.0]  # Bright red for best
                    else:
                        # Regular grasps
                        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
                        sphere_radius = 0.005
                        sphere_color = color
                    
                    # Transform the frame to the grasp pose
                    grasp_frame.transform(grasp_pose)
                    
                    # Add colored sphere to mark grasp position
                    grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                    grasp_sphere.paint_uniform_color(sphere_color)
                    grasp_sphere.translate(grasp_pose[:3, 3])
                    
                    geometries.extend([grasp_frame, grasp_sphere])
                    
                    # Log grasp details
                    pos = grasp_pose[:3, 3]
                    grasp_score = sq_info.get('grasp_score', 0.0)
                    
                    if highlight_best and idx == 0:
                        self.get_logger().info(f"★ BEST GRASP: SQ{sq_index+1}, score={grasp_score:.6f}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    elif idx < 5:  # Log first few
                        self.get_logger().info(f"  Grasp {idx+1}: SQ{sq_index+1}, score={grasp_score:.6f}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    
                except Exception as grasp_error:
                    self.get_logger().warn(f"Failed to visualize grasp {idx+1} from SQ {sq_index+1}: {grasp_error}")
            
            # Add main coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geometries.append(coord_frame)
            
            # Create legend text (in window title)
            grasp_type = "FINAL SELECTED" if highlight_best else "ALL GENERATED"
            full_window_name = f"{window_name} ({len(all_valid_sqs)} SQs, {len(all_grasp_poses)} {grasp_type} grasps)"
            
            # Visualize everything
            o3d.visualization.draw_geometries(
                geometries,
                window_name=full_window_name,
                zoom=0.7,
                front=[0, -1, 0],
                lookat=np.mean(points_for_ems, axis=0),
                up=[0, 0, 1]
            )
            
            # Print legend to console
            print(f"\n{'='*60}")
            print(f"{grasp_type} GRASPS VISUALIZATION LEGEND:")
            print(f"{'='*60}")
            for i, sq in enumerate(all_valid_sqs):
                color = colors[i % len(colors)]
                print(f"  SQ {i+1}: RGB{color} - Center={sq.translation}")
            if highlight_best:
                print(f"  ★ BEST GRASP: Large red sphere with large coordinate frame")
                print(f"  Other grasps: Small colored spheres with small coordinate frames")
            else:
                print(f"  All grasps: Small colored spheres (colored by source SQ)")
            print(f"{'='*60}")
                    
        except Exception as e:
            self.get_logger().error(f"Error in multi-superquadric visualization: {e}")
            
            
    def grasp_execution_callback(self, msg):
        """Track grasp execution state"""
        self.grasp_executing = msg.data
        if msg.data:
            self.get_logger().info("Grasp execution started - pausing pose generation")
        else:
            self.get_logger().info("Grasp execution finished - resuming pose generation")
            
            
    def _start_live_capture(self):
        """Start separate high-frequency capture thread for web display"""
        try:
            self.live_capture_enabled = True
            self.live_capture_thread = threading.Thread(target=self._live_capture_worker, daemon=True)
            self.live_capture_thread.start()
            self.get_logger().info("Started live capture thread for web display at ~15 FPS")
            
        except Exception as e:
            self.get_logger().error(f"Failed to start live capture: {e}")
            self.live_capture_enabled = False

    def _live_capture_worker(self):
        """High-frequency capture worker for web display (runs at ~15 FPS)"""
        frame_count = 0
        
        while self.live_capture_enabled and rclpy.ok():
            try:
                start_time = time.time()
                
                # Quick frame capture (non-blocking)
                if (self.zed1.grab() == sl.ERROR_CODE.SUCCESS and 
                    self.zed2.grab() == sl.ERROR_CODE.SUCCESS):
                    
                    # Retrieve frames quickly
                    self.zed1.retrieve_image(self.image1, view=sl.VIEW.LEFT)
                    self.zed2.retrieve_image(self.image2, view=sl.VIEW.LEFT)
                    
                    frame1 = cv2.cvtColor(self.image1.get_data(), cv2.COLOR_BGRA2BGR)
                    frame2 = cv2.cvtColor(self.image2.get_data(), cv2.COLOR_BGRA2BGR)
                    
                    # Store original frame dimensions
                    original_height, original_width = frame1.shape[:2]
                    
                    # Quick YOLO inference for live display (lower resolution for speed)
                    yolo_width, yolo_height = 320, 180
                    frame1_small = cv2.resize(frame1, (yolo_width, yolo_height))
                    frame2_small = cv2.resize(frame2, (yolo_width, yolo_height))
                    
                    # Calculate proper scaling factors
                    width_scale = original_width / yolo_width    # e.g., 1920 / 320 = 6.0
                    height_scale = original_height / yolo_height # e.g., 1080 / 180 = 6.0
                    
                    # Fast YOLO inference
                    results_batch = self.model.predict(
                        source=[frame1_small, frame2_small],
                        classes=[0, 1, 2, 3, 4],
                        conf=self.conf_threshold,
                        device=self.device,
                        verbose=False,  # Reduce output
                        imgsz=320       # Smaller image size for speed
                    )
                    
                    results1_live, results2_live = results_batch[0], results_batch[1]
                    
                    # Create live display frame with proper scaling
                    live_display = self._create_live_web_display(
                        frame1, frame2, results1_live, results2_live, frame_count, 
                        width_scale=width_scale, height_scale=height_scale
                    )
                    
                    # Thread-safe update of web frame AND save to disk immediately
                    with self.web_frame_lock:
                        self.latest_web_frame = live_display
                        
                        # Save frame to disk for web server (every frame for true live display)
                        web_image_path = f"{self.web_dir}/latest.jpg"
                        cv2.imwrite(web_image_path, live_display, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    # Less frequent logging
                    if frame_count % 60 == 0:  # Every 4 seconds
                        self.get_logger().info(f"Live frame {frame_count} updated (15 FPS)")
                    
                    frame_count += 1
                
                # Target ~15 FPS for live capture
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0/15.0 - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                if frame_count < 5:  # Only log first few errors
                    self.get_logger().warn(f"Live capture error (frame {frame_count}): {e}")
                time.sleep(0.1)  # Brief pause on error
                
        self.get_logger().info("Live capture worker stopped")

    def _draw_yolo_detections_web(self, frame, results, camera_name, width_scale=1.0, height_scale=1.0):
        """Draw YOLO detection boxes and labels on frame for web display with proper scaling"""
        try:
            if results is None:
                return
                
            if results.boxes is not None and len(results.boxes) > 0:
                # Get boxes as numpy array and scale them properly
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                # Apply proper scaling (different for width and height if needed)
                boxes[:, [0, 2]] *= width_scale   # Scale x coordinates (x1, x2)
                boxes[:, [1, 3]] *= height_scale  # Scale y coordinates (y1, y2)
                
                for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    
                    # Ensure coordinates are within frame bounds
                    frame_height, frame_width = frame.shape[:2]
                    x1 = max(0, min(x1, frame_width - 1))
                    y1 = max(0, min(y1, frame_height - 1))
                    x2 = max(0, min(x2, frame_width - 1))
                    y2 = max(0, min(y2, frame_height - 1))
                    
                    class_name = self.class_names.get(class_id, f'Class_{class_id}')
                    
                    # Color based on class with better visibility
                    colors = [
                        (0, 255, 0),    # Green - Cone
                        (255, 0, 0),    # Blue - Cup  
                        (0, 0, 255),    # Red - Mallet
                        (255, 255, 0),  # Cyan - Screw Driver
                        (255, 0, 255)   # Magenta - Sunscreen
                    ]
                    color = colors[class_id % len(colors)]
                    
                    # Draw thicker bounding box for web visibility
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with better background
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Ensure label doesn't go outside frame
                    label_y = max(y1, label_size[1] + 15)
                    label_x1 = max(0, x1)
                    label_x2 = min(frame_width, x1 + label_size[0] + 10)
                    
                    # Draw background rectangle for label
                    cv2.rectangle(frame, (label_x1, label_y - label_size[1] - 15), 
                                (label_x2, label_y), color, -1)
                    
                    # Draw label text in black for better contrast
                    cv2.putText(frame, label, (label_x1 + 5, label_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
        except Exception as e:
            self.get_logger().error(f"Error drawing YOLO detections for web: {e}")

    def _create_live_web_display(self, frame1, frame2, results1, results2, frame_count, width_scale=1.0, height_scale=1.0):
        """Create live display frame for web interface with proper scaling"""
        try:
            display1 = frame1.copy()
            display2 = frame2.copy()
            
            # Draw YOLO detections with proper scaling
            self._draw_yolo_detections_web(display1, results1, "Camera 1", width_scale, height_scale)
            self._draw_yolo_detections_web(display2, results2, "Camera 2", width_scale, height_scale)
            
            # Add live feed info
            current_time = time.time()
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(current_time))
            
            # Frame 1 overlay
            cv2.putText(display1, f"LIVE Camera 1 - {timestamp_str}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display1, f"Live FPS: ~15", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display1, f"Frame: {frame_count}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display1, f"Scale: {width_scale:.1f}x{height_scale:.1f}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Frame 2 overlay  
            cv2.putText(display2, f"LIVE Camera 2 - {timestamp_str}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add detection counts
            det_count1 = len(results1.boxes) if results1 and results1.boxes is not None else 0
            det_count2 = len(results2.boxes) if results2 and results2.boxes is not None else 0
            
            cv2.putText(display1, f"Live Detections: {det_count1}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(display2, f"Live Detections: {det_count2}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Add processing status if available
            if hasattr(self, 'fps_values') and self.fps_values:
                processing_fps = sum(self.fps_values) / len(self.fps_values)
                cv2.putText(display2, f"Processing: {processing_fps:.1f} FPS", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Resize for web (balance between quality and speed)
            display1_resized = cv2.resize(display1, (640, 360))
            display2_resized = cv2.resize(display2, (640, 360))
            
            # Combine
            combined_display = cv2.hconcat([display1_resized, display2_resized])
            
            # Add title bar
            border_height = 40
            total_height = combined_display.shape[0] + border_height
            total_width = combined_display.shape[1]
            
            final_display = np.zeros((total_height, total_width, 3), dtype=np.uint8)
            final_display[border_height:, :] = combined_display
            
            # Live indicator
            cv2.rectangle(final_display, (0, 0), (total_width, border_height), (0, 100, 0), -1)  # Green bar
            cv2.putText(final_display, f"LIVE - ZED Stereo YOLO Detection (Frame {frame_count})", (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return final_display
            
        except Exception as e:
            self.get_logger().error(f"Error creating live display: {e}")
            # Return error frame
            error_frame = np.zeros((400, 1280, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Live Display Error", (400, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            return error_frame

    def _setup_web_server(self):
        """Setup web server with updated HTML for live display"""
        try:
            self.web_dir = "/tmp/yolo_live"
            os.makedirs(self.web_dir, exist_ok=True)
            
            # Create a placeholder image first
            placeholder_frame = np.zeros((400, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, "Initializing cameras and YOLO...", (400, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.imwrite(f"{self.web_dir}/latest.jpg", placeholder_frame)
            
            html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LIVE YOLO Detection - ZED Cameras</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
            }
            .live-indicator {
                text-align: center;
                background-color: #28a745;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-weight: bold;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            .image-container {
                text-align: center;
                margin-bottom: 20px;
            }
            #live_image {
                max-width: 100%;
                height: auto;
                border: 2px solid #28a745;
                border-radius: 5px;
            }
            .controls {
                text-align: center;
                margin: 20px 0;
            }
            button {
                background-color: #28a745;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin: 0 10px;
                font-size: 16px;
            }
            button:hover {
                background-color: #218838;
            }
            .status {
                text-align: center;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .status.connected {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.disconnected {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .fps-display {
                font-weight: bold;
                color: #28a745;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>[LIVE] YOLO Detection - ZED Stereo Cameras</h1>
            
            <div class="live-indicator">
                [LIVE FEED] - Real-time at ~15 FPS
            </div>
            
            <div id="status" class="status connected">
                Live feed active - Independent of processing loop
            </div>
            
            <div class="controls">
                <button onclick="toggleRefresh()">Pause/Resume</button>
                <button onclick="saveImage()">Save Frame</button>
                <button onclick="toggleFullscreen()">Fullscreen</button>
                <button onclick="resetConnection()">Reset</button>
            </div>
            
            <div class="image-container">
                <img id="live_image" src="latest.jpg" alt="Loading live feed..." />
            </div>
            
            <div class="fps-display">
                Display FPS: <span id="fps">--</span> | 
                Last updated: <span id="timestamp">--</span> |
                Total frames: <span id="frame_count">0</span>
            </div>
        </div>

        <script>
            let refreshInterval;
            let isRefreshing = true;
            let frameCount = 0;
            let lastFrameTime = Date.now();
            
            function updateImage() {
                if (!isRefreshing) return;
                
                const img = document.getElementById('live_image');
                const timestamp = Date.now();
                
                // Calculate display FPS
                const timeDiff = timestamp - lastFrameTime;
                const displayFPS = timeDiff > 0 ? (1000 / timeDiff).toFixed(1) : 0;
                
                // Add timestamp to prevent caching
                img.src = 'latest.jpg?' + timestamp;
                
                // Update displays
                document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();
                document.getElementById('fps').textContent = displayFPS;
                document.getElementById('frame_count').textContent = frameCount++;
                
                lastFrameTime = timestamp;
            }
            
            function toggleRefresh() {
                isRefreshing = !isRefreshing;
                const button = event.target;
                const status = document.getElementById('status');
                
                if (isRefreshing) {
                    button.textContent = 'Pause';
                    status.textContent = 'Live feed active - Independent of processing loop';
                    status.className = 'status connected';
                } else {
                    button.textContent = 'Resume';
                    status.textContent = 'Paused - Click resume to continue';
                    status.className = 'status disconnected';
                }
            }
            
            function saveImage() {
                const link = document.createElement('a');
                link.href = 'latest.jpg?' + Date.now();
                link.download = 'live_yolo_detection_' + Date.now() + '.jpg';
                link.click();
            }
            
            function toggleFullscreen() {
                const img = document.getElementById('live_image');
                if (img.requestFullscreen) {
                    img.requestFullscreen();
                }
            }
            
            function resetConnection() {
                frameCount = 0;
                document.getElementById('frame_count').textContent = '0';
                updateImage();
            }
            
            // Handle image load events
            document.getElementById('live_image').onerror = function() {
                const status = document.getElementById('status');
                status.textContent = 'Connection lost - Check if cameras are running';
                status.className = 'status disconnected';
            };
            
            document.getElementById('live_image').onload = function() {
                const status = document.getElementById('status');
                if (isRefreshing) {
                    status.textContent = 'Live feed active - Independent of processing loop';
                    status.className = 'status connected';
                }
            };
            
            // High refresh rate for truly live display (66ms = ~15 FPS to match capture)
            refreshInterval = setInterval(updateImage, 66);
            
            // Initial load
            updateImage();
        </script>
    </body>
    </html>
            """
            
            with open(f"{self.web_dir}/index.html", "w", encoding='utf-8') as f:
                f.write(html_content)
            
            self.web_server_thread = threading.Thread(target=self._run_web_server, daemon=True)
            self.web_server_thread.start()
            
            self.get_logger().info("LIVE web interface available at: http://localhost:8080")
            self.get_logger().info("   This will show true live feed at ~15 FPS independent of ROS processing")
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup web server: {e}")
            self.web_enabled = False

    def _run_web_server(self):
        """Run simple web server with better error handling"""
        try:
            class Handler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory="/tmp/yolo_live", **kwargs)
                
                def log_message(self, format, *args):
                    # Suppress server logs to keep ROS logs clean
                    pass
                    
                def do_GET(self):
                    """Handle GET requests with better error handling"""
                    try:
                        super().do_GET()
                    except (BrokenPipeError, ConnectionResetError):
                        # Browser closed connection - this is normal, don't log as error
                        pass
                    except Exception as e:
                        # Only log unexpected errors
                        print(f"Web server request error: {e}")
            
            httpd = HTTPServer(("localhost", 8080), Handler)
            self.get_logger().info("Web server started on port 8080")
            httpd.serve_forever()
            
        except Exception as e:
            self.get_logger().error(f"Web server error: {e}")


            
    def destroy_node(self):
        """Clean up resources"""
        self.get_logger().info("Shutting down ZED cameras...")
        
        # Stop live capture
        if self.live_capture_enabled:
            self.live_capture_enabled = False
            if self.live_capture_thread and self.live_capture_thread.is_alive():
                self.live_capture_thread.join(timeout=2.0)
        
        # Stop web server
        if self.web_enabled:
            self.get_logger().info("Stopping web server...")
        
        # Clean up cameras
        if hasattr(self, 'zed1'):
            self.zed1.close()
        if hasattr(self, 'zed2'):
            self.zed2.close()
            
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