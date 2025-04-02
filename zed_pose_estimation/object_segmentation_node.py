import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import threading
import time
import open3d as o3d
from std_msgs.msg import Header

class ObjectSegmentationNode(Node):
    def __init__(self):
        super().__init__('object_segmentation_node')

        # Declare parameters
        self.declare_parameter('camera_namespaces', ['zed1', 'zed2'])
        self.declare_parameter('yolo_model_path', '/home/chris/franka_ros2_ws/src/zed_pose_estimation/models/yolo/best.pt')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('publish_rate', 1.0)

        # Get parameters
        self.camera_namespaces = self.get_parameter('camera_namespaces').get_parameter_value().string_array_value
        self.model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        #FOR NOW MANUAL INPUT OF CAMERA INSTRINSICS
        self.camera_intrinsics = {
            'zed1': {  # Camera 33137761
                'fx': 946.0263671875,
                'fy': 946.0263671875,
                'cx': 652.24755859375,
                'cy': 351.9144592285156
            },
            'zed2': {  # Camera 36829049
                'fx': 960.224853515625, 
                'fy': 960.224853515625,
                'cx': 651.682373046875,
                'cy': 354.01409912109375
            }
        }
        

        # Setup CV bridge, YOLO model, and locks
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"Loaded YOLO model from {self.model_path}")
        
        # QoS profile for camera topics
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Data storage for each camera
        self.cameras = {}
        for namespace in self.camera_namespaces:
            self.cameras[namespace] = {
                'rgb_image': None,
                'point_cloud': None,
                'detections': None,
                'rgb_lock': threading.Lock(),
                'pc_lock': threading.Lock(),
                'detection_lock': threading.Lock(),
                'segmented_pc_publisher': self.create_publisher(
                    PointCloud2, 
                    f'/{namespace}/segmented_point_cloud', 
                    10
                )
            }
            
            # Create subscribers for each camera
            self.create_subscription(
                Image,
                f'/{namespace}/{namespace}/rgb/image_raw',
                lambda msg, ns=namespace: self.rgb_callback(msg, ns),
                self.qos_profile
            )
            
            self.create_subscription(
                PointCloud2,
                f'/{namespace}/{namespace}/point_cloud/cloud_registered',
                lambda msg, ns=namespace: self.pointcloud_callback(msg, ns),
                self.qos_profile
            )
        
        # Timer for processing and publishing at regular intervals
        self.timer = self.create_timer(1.0 / self.publish_rate, self.process_and_publish)
        
    def rgb_callback(self, msg, namespace):
        """Process incoming RGB image"""
        try:
            with self.cameras[namespace]['rgb_lock']:
                self.cameras[namespace]['rgb_image'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
            # Perform object detection
            frame = self.cameras[namespace]['rgb_image']
            if frame is not None:
                results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)


                # Annotate frame for visualization
                annotated_frame = results[0].plot(line_width=2, font_size=18)
                window_name = f"YOLO Detection/Segmentation of {namespace}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Ensure a new window for each camera
                cv2.imshow(window_name, annotated_frame)
                cv2.waitKey(1)  # Refresh the window
                
                
                with self.cameras[namespace]['detection_lock']:
                    self.cameras[namespace]['detections'] = results[0]
                
                # Log detection results
                if len(results[0].boxes) > 0:
                    self.get_logger().debug(f"{namespace}: Detected {len(results[0].boxes)} objects")
                
        except Exception as e:
            self.get_logger().error(f"Error in RGB callback: {e}")
    
    def pointcloud_callback(self, msg, namespace):
        """Process incoming point cloud"""
        try:
            with self.cameras[namespace]['pc_lock']:
                self.cameras[namespace]['point_cloud'] = msg
        except Exception as e:
            self.get_logger().error(f"Error in point cloud callback: {e}")
    
    def process_and_publish(self):
        """Process data and publish segmented point clouds"""
        for namespace, data in self.cameras.items():
            try:
                with data['rgb_lock'], data['pc_lock'], data['detection_lock']:
                    if data['rgb_image'] is None or data['point_cloud'] is None or data['detections'] is None:
                        continue

                    rgb_image = data['rgb_image'].copy()
                    point_cloud_msg = data['point_cloud']
                    detections = data['detections']

                # Parse point cloud into N x 3 array (x, y, z)
                points = np.array([
                    list(p)[:3] for p in pc2.read_points(
                        point_cloud_msg,
                        field_names=("x", "y", "z"),
                        skip_nans=True
                    )
                ], dtype=np.float32)

                if points.size == 0:
                    self.get_logger().warn(f"{namespace}: Empty or invalid point cloud")
                    continue

                # Use the correct camera intrinsics for this camera
                img_height, img_width = rgb_image.shape[:2]
                intrinsics = self.camera_intrinsics.get(namespace, {
                    'fx': 700.0,  # Fallback defaults if not found
                    'fy': 700.0,
                    'cx': img_width / 2.0,
                    'cy': img_height / 2.0
                })
                
                fx = intrinsics['fx']
                fy = intrinsics['fy']
                cx = intrinsics['cx']
                cy = intrinsics['cy']
                
                self.get_logger().debug(f"Using intrinsics for {namespace}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            

                if detections is None or detections.masks is None or len(detections.masks.data) == 0:
                    self.get_logger().debug(f"{namespace}: No detections with masks")
                    continue

                segmented_point_clouds = []

                for i, mask_tensor in enumerate(detections.masks.data):
                    try:
                        obj_mask = mask_tensor.cpu().numpy().astype(np.uint8)
                        obj_mask = cv2.resize(obj_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                        class_id = float(detections.boxes.cls[i].item()) if hasattr(detections.boxes, 'cls') else -1.0

                        segmented_points = []

                        for point in points:
                            x, y, z = point
                            if z <= 0 or np.isnan(z):
                                continue

                            # Project point into image space
                            u = int(fx * x / z + cx)
                            v = int(fy * y / z + cy)

                            if 0 <= u < img_width and 0 <= v < img_height:
                                if obj_mask[v, u] > 0:
                                    segmented_points.append([x, y, z, class_id])

                        if segmented_points:
                            segmented_point_clouds.append(np.array(segmented_points, dtype=np.float32))

                    except Exception as e:
                        self.get_logger().warn(f"{namespace}: Error processing object {i}: {e}")
                        continue

                if not segmented_point_clouds:
                    self.get_logger().info(f"{namespace}: No valid segmented points")
                    continue

                # if self.debug_vis:
                #     # Create visualization of projected points
                #     debug_img = rgb_image.copy()
                    
                #     # Sample points to avoid overcrowding the visualization
                #     sample_rate = max(1, len(points) // 1000)  # Show at most ~1000 points
                    
                #     for i in range(0, len(points), sample_rate):
                #         x, y, z = points[i]
                #         if z <= 0 or np.isnan(z):
                #             continue
                            
                #         # Project 3D point to image
                #         u = int(fx * x / z + cx)
                #         v = int(fy * y / z + cy)
                        
                #         if 0 <= u < img_width and 0 <= v < img_height:
                #             # Color by depth (red=close, blue=far)
                #             depth_norm = min(1.0, z / 5.0)  # Normalize depth (0-5m)
                #             b = int(255 * (1 - depth_norm))
                #             r = int(255 * depth_norm)
                #             cv2.circle(debug_img, (u, v), 1, (b, 0, r), -1)
                    
                #     # Show image with projected points
                #     cv2.imshow(f"Point Cloud Projection - {namespace}", debug_img)
                #     cv2.waitKey(1)

                all_segmented = np.vstack(segmented_point_clouds)

                # Create ROS2 PointCloud2 message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = point_cloud_msg.header.frame_id

                fields = [
                    pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                    pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                    pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                    pc2.PointField(name='class_id', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
                ]

                pc_msg = pc2.create_cloud(header, fields, all_segmented.tolist())
                data['segmented_pc_publisher'].publish(pc_msg)

                self.get_logger().info(f"{namespace}: Published segmented point cloud with {len(all_segmented)} points")

            except Exception as e:
                self.get_logger().error(f"{namespace}: Error in processing loop: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = ObjectSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()