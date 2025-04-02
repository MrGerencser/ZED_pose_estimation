# Converting camera data to ROS messages

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct
from .camera_manager import ZEDCameraManager
import yaml
from ament_index_python.packages import get_package_share_directory
import os
import open3d as o3d
from std_srvs.srv import Trigger
import threading
import time
from std_msgs.msg import Header
import numpy as np
import struct

# Point cloud field definitions
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

FIELDS_XYZRGB = FIELDS_XYZ + [
    PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
]

class CameraManagerNode(Node):
    def __init__(self):
        super().__init__('camera_manager_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_id', 1),
                ('serial_number', ''),
                ('camera_name', 'zed'),
                ('publish_rate', 10.0),
                ('config_file', ''),      # Path to camera config file
                ('resolution', 'HD720'),  # Default resolution
                ('fps', 15.0)             # Default FPS
            ]
        )
        
        self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        self.serial_number = self.get_parameter('serial_number').get_parameter_value().string_value
        # Get serial number and handle it appropriately
        self.serial_number = self.get_parameter('serial_number').get_parameter_value().string_value
        if self.serial_number and self.serial_number != 'None':
            try:
                self.serial_number = int(self.serial_number)  # ZED SDK expects int serial numbers
            except ValueError:
                self.get_logger().error(f"Invalid serial number format: {self.serial_number}")
                self.serial_number = None
        else:
            self.serial_number = None
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value
        resolution = self.get_parameter('resolution').get_parameter_value().string_value
        fps = self.get_parameter('fps').get_parameter_value().double_value
        
        # Try to load configuration from file
        camera_config = {}
        if self.config_file:
            try:
                # Check if path is absolute
                if os.path.isfile(self.config_file):
                    config_path = self.config_file
                else:
                    # Try to resolve relative to package
                    pkg_share = get_package_share_directory('zed_pose_estimation')
                    config_path = os.path.join(pkg_share, self.config_file)
                
                with open(config_path, 'r') as f:
                    camera_config = yaml.safe_load(f)
                self.get_logger().info(f"Loaded camera configuration from {config_path}")
                
                # Override parameters from config file
                if 'video' in camera_config:
                    if 'resolution' in camera_config['video']:
                        res_code = camera_config['video']['resolution']
                        if res_code == 1:
                            resolution = 'VGA'
                        elif res_code == 2:
                            resolution = 'HD720'
                        elif res_code == 3:
                            resolution = 'HD1080'
                        elif res_code == 4:
                            resolution = 'HD2K'
                    
                    if 'frame_rate' in camera_config['video']:
                        fps = float(camera_config['video']['frame_rate'])
                
            except Exception as e:
                self.get_logger().error(f"Failed to load camera config: {e}")
        
        self.get_logger().info(f"Initializing camera {self.camera_id} with resolution {resolution} at {fps} FPS")
        
        # Initialize camera with configuration
        self.camera = ZEDCameraManager(
            camera_id=self.camera_id,
            serial_number=self.serial_number,
            resolution=resolution,
            fps=fps
        )
        self.bridge = CvBridge()
        
        if not self.camera.open_camera():
            self.get_logger().error(f"Failed to open camera {self.camera_id}. Exiting.")
            rclpy.shutdown()
            return
        
        # Get and log camera intrinsics
        calib_params = self.camera.get_calibration_parameters()
        if calib_params:
            self.get_logger().info(f"Camera {self.camera_id} Intrinsics:")
            self.get_logger().info(f"  fx: {calib_params['fx']}")
            self.get_logger().info(f"  fy: {calib_params['fy']}")
            self.get_logger().info(f"  cx: {calib_params['cx']}")
            self.get_logger().info(f"  cy: {calib_params['cy']}")
        
        # Create publishers for RGB and depth images and point cloud
        self.rgb_pub = self.create_publisher(
            Image, f'{self.camera_name}/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(
            Image, f'{self.camera_name}/depth/image_raw', 10)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, f'{self.camera_name}/point_cloud/cloud_registered', 10)
        
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.get_logger().info(f"Camera {self.camera_id} initialized successfully")
        
        # Initialize point cloud visualization variables
        self.latest_cloud_points = None
        self.cloud_lock = threading.Lock()

    def timer_callback(self):
        image, depth, point_cloud = self.camera.capture()
        if image is not None and depth is not None:
            # Convert ZED image to OpenCV format
            image_ocv = image.get_data()
            depth_ocv = depth.get_data()
            
            # Convert BGRA to BGR (ZED images are 4-channel)
            image_bgr = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            
            # Normalize depth for visualization - handling NaN values
            depth_np = np.array(depth_ocv).astype(np.float32)
            
            # Replace infinity and NaN with zeros
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip to valid range and normalize to 16-bit
            normalized_depth = np.clip(depth_np, 0, 20.0)
            normalized_depth = (normalized_depth / 20.0 * 65535).astype(np.uint16)
            
            # Publish images
            rgb_msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(normalized_depth, encoding="mono16")

            # Create timestamp and headers
            timestamp = self.get_clock().now().to_msg()
            frame_id = f"{self.camera_name}_camera_link"
            
            # Set message headers
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = frame_id
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = frame_id
            
            # Publish RGB and depth
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
            
            # Publish point cloud if available
            if point_cloud is not None:
                cloud_msg = self.create_point_cloud_msg(point_cloud, timestamp, frame_id)
                
                # Publish point cloud message
                self.pointcloud_pub.publish(cloud_msg)
            else:
                self.get_logger().warn("Point cloud data is None")
        else:
            self.get_logger().warn(f"Failed to capture images from camera {self.camera_id}")

        
    def create_point_cloud_msg(self, point_cloud_data, timestamp, frame_id):
        """Convert ZED point cloud data to ROS2 PointCloud2 message."""
        try:
            # Downsampling point cloud
            # point_cloud_data = point_cloud_data.voxel_down_sample(0.01)

            # Get point cloud data as a NumPy array
            cloud_data = np.array(point_cloud_data.get_data(), dtype=np.float32)

            height, width, channels = cloud_data.shape

            # Debug info about the cloud data
            self.get_logger().info(f"Point cloud shape: {cloud_data.shape}, dtype: {cloud_data.dtype}")

            # Reshape cloud data to a 2D array of points
            cloud_data = cloud_data.reshape(-1, channels)

            # Extract valid points (XYZ + optional RGB)
            points_list = []
            for point in cloud_data:
                # Check if the point is valid
                if not np.isnan(point[0]) and not np.isinf(point[0]):
                    x, y, z = point[:3]
                    if channels >= 6:
                        r, g, b = point[3:6]
                        # Pack RGB into a single integer
                        rgb = struct.unpack('I', struct.pack('BBBB', int(b), int(g), int(r), 255))[0]
                        points_list.append([x, y, z, rgb])
                    else:
                        points_list.append([x, y, z])

            self.get_logger().info(f"Extracted {len(points_list)} valid points from point cloud")

            if not points_list:
                self.get_logger().warn("No valid points found in point cloud")
                return PointCloud2()

            # Define the header
            header = Header()
            header.stamp = timestamp
            header.frame_id = frame_id

            # Determine fields and point step
            if channels >= 6:
                fields = FIELDS_XYZRGB
                point_step = 16  # 4 floats (x, y, z, rgb) * 4 bytes
            else:
                fields = FIELDS_XYZ
                point_step = 12  # 3 floats (x, y, z) * 4 bytes

            # Create PointCloud2 message
            cloud_msg = pc2.create_cloud(header, fields, points_list)
            cloud_msg.height = 1
            cloud_msg.width = len(points_list)
            cloud_msg.is_dense = True

            return cloud_msg

        except Exception as e:
            self.get_logger().error(f"Error creating point cloud message: {e}")
            return PointCloud2()
    
    
    def destroy_node(self):
        self.camera.close_camera()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()