import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from zed_pose_estimation.pose_estimation_utils import visualize_point_cloud
import numpy as np


class PointCloudVisualizerNode(Node):
    def __init__(self):
        super().__init__('pointcloud_visualizer_node')

        # Declare and get parameters
        self.declare_parameter('pointcloud_topic', 'zed1/point_cloud/cloud_registered')
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value

        # Create a subscriber to the point cloud topic
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            10
        )

        self.get_logger().info(f"Subscribed to point cloud topic: {self.pointcloud_topic}")

    def pointcloud_callback(self, msg):
        """Callback function to process and visualize the point cloud."""
        try:
            # Convert PointCloud2 message to a list of points
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])

            # Convert to a NumPy array
            point_cloud_np = np.array(points, dtype=np.float32)

            # Visualize the point cloud
            self.get_logger().info(f"Visualizing point cloud with {len(point_cloud_np)} points")
            visualize_point_cloud(point_cloud_np, title="ZED Point Cloud")
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()