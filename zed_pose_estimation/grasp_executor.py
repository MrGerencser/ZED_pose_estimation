#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from franka_msgs.action import Grasp, Move, Homing
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time
from scipy.spatial.transform import Rotation


class GraspExecutor(Node):
    def __init__(self):
        super().__init__('grasp_executor')
        
        # Configuration constants
        self.config = {
            'workspace_limits': {
                'x_min': 0.1, 'x_max': 0.8,
                'y_min': -0.5, 'y_max': 0.5,
                'z_min': 0.0, 'z_max': 0.8
            },
            'gripper': {
                'max_width': 0.08,
                'goal_width': 0.0,
                'speed': 0.05,
                'force': 140.0,
                'epsilon_inner': 0.05,
                'epsilon_outer': 0.07
            },
            'offsets': {
                'pre_grasp': 0.15,
                'approach': 0.05,
                'lift': 0.1
            },
            'timing': {
                'home': 4, 'gripper': 2, 'pre_grasp': 3,
                'approach': 2, 'grasp': 2, 'close_gripper': 3,
                'lift': 3, 'safe': 3
            }
        }
        
        self.callback_group = ReentrantCallbackGroup()
        
        # Setup communication
        self._setup_publishers_subscribers()
        
        # Setup gripper
        self._setup_gripper()
        
        # Create home pose
        self.home_pose = self._create_home_pose()
        
        # State management
        self.latest_grasp_pose = None
        self.is_executing = False
        self.should_abort = False
        
        self.get_logger().info("Grasp Executor ready!")
    
    def _setup_publishers_subscribers(self):
        """Setup ROS2 communication"""
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/perception/object_pose',
            self.grasp_pose_callback, 1,
            callback_group=self.callback_group
        )
        
        self.cartesian_target_publisher = self.create_publisher(
            PoseStamped, '/cartesian_target_pose', 10
        )
        
        self.status_publisher = self.create_publisher(
            Bool, '/robot/grasp_status', 10
        )
    
    def _setup_gripper(self):
        """Setup gripper action clients"""
        self.homing_client = ActionClient(
            self, Homing, '/fr3_gripper/homing', 
            callback_group=self.callback_group
        )
        self.move_client = ActionClient(
            self, Move, '/fr3_gripper/move', 
            callback_group=self.callback_group
        )
        self.grasp_client = ActionClient(
            self, Grasp, '/fr3_gripper/grasp', 
            callback_group=self.callback_group
        )
        
        # Wait for servers and home gripper
        self._wait_for_gripper_servers()
        self._home_gripper()
    
    def _wait_for_gripper_servers(self):
        """Wait for all gripper action servers"""
        servers = [
            (self.homing_client, 'Homing'),
            (self.move_client, 'Move'),
            (self.grasp_client, 'Grasp')
        ]
        
        for client, name in servers:
            self.get_logger().info(f'Waiting for {name} action server...')
            while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
                self.get_logger().info(f'{name} not available, waiting...')
            
            if rclpy.ok():
                self.get_logger().info(f'{name} action server found.')
            else:
                raise SystemExit('ROS shutdown while waiting for servers')
    
    def _home_gripper(self):
        """Home the gripper"""
        self.get_logger().info("Homing gripper...")
        goal = Homing.Goal()
        self.homing_client.send_goal_async(goal)
    
    def _create_home_pose(self):
        """Create home pose"""
        home_rotation = Rotation.from_euler('xyz', [np.pi, 0.0, 0.0])
        home_quat = home_rotation.as_quat()  # x, y, z, w
        return self._create_pose(
            0.5, 0.0, 0.35, 
            home_quat[3], home_quat[0], home_quat[1], home_quat[2]
        )
    
    def _create_pose(self, x, y, z, qw, qx, qy, qz, frame_id="panda_link0"):
        """Create PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        
        pose.pose.orientation.w = qw
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        
        return pose
    
    def _create_offset_pose(self, base_pose, z_offset=0.0):
        """Create pose with Z offset from base pose"""
        new_pose = PoseStamped()
        new_pose.header = base_pose.header
        new_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Copy position with offset
        new_pose.pose.position.x = base_pose.pose.position.x
        new_pose.pose.position.y = base_pose.pose.position.y
        new_pose.pose.position.z = base_pose.pose.position.z + z_offset
        
        # Use detected orientation with validation
        new_pose.pose.orientation = base_pose.pose.orientation
        
        # Validate and normalize quaternion
        quat = base_pose.pose.orientation
        quat_norm = (quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)**0.5
        
        if abs(quat_norm - 1.0) > 0.01:
            self.get_logger().warn(f"Normalizing quaternion (norm: {quat_norm:.3f})")
            new_pose.pose.orientation.x /= quat_norm
            new_pose.pose.orientation.y /= quat_norm
            new_pose.pose.orientation.z /= quat_norm
            new_pose.pose.orientation.w /= quat_norm
        
        # Clamp orientation if needed
        self._clamp_orientation(new_pose)
        
        return new_pose
    
    def _clamp_orientation(self, pose):
        """Clamp orientation to safe limits"""
        quat = [pose.pose.orientation.x, pose.pose.orientation.y,
                pose.pose.orientation.z, pose.pose.orientation.w]
        
        euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        
        if euler[1] < -90 or euler[1] > 90:
            euler[1] = np.clip(euler[1], -90, 90)
            clamped_quat = Rotation.from_euler('xyz', euler, degrees=True).as_quat()
            
            pose.pose.orientation.x = float(clamped_quat[0])
            pose.pose.orientation.y = float(clamped_quat[1])
            pose.pose.orientation.z = float(clamped_quat[2])
            pose.pose.orientation.w = float(clamped_quat[3])
            
            self.get_logger().info(f"Clamped pitch to {euler[1]:.1f}°")
            
            
    def _check_table_collision(self, pose, table_height=0.0):
        """Check if gripper will collide with table
        
        Args:
            pose: Pose or PoseStamped object
            table_height: Height of table surface (default: 0.0m)
        
        Returns:
            bool: True if collision detected, False if safe
        
        The collision model calculates the minimum safe height based on:
        - Gripper orientation (alpha angle)
        - Gripper width and finger geometry
        - Safety margin
        """
        
        # Extract pose data
        actual_pose = pose.pose if hasattr(pose, 'pose') else pose
        
        # Get position and orientation
        pos = actual_pose.position
        quat = np.array([
            actual_pose.orientation.x,
            actual_pose.orientation.y, 
            actual_pose.orientation.z,
            actual_pose.orientation.w
        ])
        
        # Calculate gripper angle relative to table
        euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        alpha_deg = euler[0] - 180.0  # Gripper orientation adjustment
        alpha_rad = np.radians(alpha_deg)
        
        # Handle edge case for angle
        if abs(np.sin(alpha_rad)) < 0.01:  # gripper in default position
            self.get_logger().warn(f"Gripper in default position (alpha={alpha_deg:.1f}°)")
            return False  # assume no collision
        
        # Physical parameters
        FINGER_HEIGHT = 0.018  # meters
        FINGER_WIDTH = 0.005  # meters
        SAFETY_MARGIN = 0.00  # 5mm safety buffer
        half_gripper_width = self.config['gripper']['max_width'] / 2.0
        
        # Calculate collision geometry
        sin_alpha = np.sin(alpha_rad)
        
        # Distance from gripper center to finger tip projection
        finger_offset = FINGER_HEIGHT / (2 * abs(sin_alpha))
        
        # Calculate minimum safe height above table
        z_min_safe = (abs(sin_alpha) * (half_gripper_width + FINGER_WIDTH + finger_offset) + SAFETY_MARGIN + table_height)
        self.get_logger().debug(
            f"Collision model: alpha={alpha_deg:.1f}°, "
            f"z_min_safe={z_min_safe:.3f}m, "
            f"finger_offset={finger_offset:.3f}m, "
            f"half_gripper_width={half_gripper_width:.3f}m"
        )
        
        # Check collision
        clearance = pos.z - z_min_safe
        
        if clearance < 0:
            self.get_logger().warn(
                f"Table collision risk: pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}], "
                f"alpha={alpha_deg:.1f}°, clearance={clearance*1000:.1f}mm"
            )
            return True
        
        self.get_logger().debug(
            f"Pose safe: clearance={clearance*1000:.1f}mm, alpha={alpha_deg:.1f}°"
        )
        return False

    def _is_pose_reachable(self, pose):
        """Check if pose is within workspace"""
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        limits = self.config['workspace_limits']

        if not (limits['x_min'] <= x <= limits['x_max'] and
                limits['y_min'] <= y <= limits['y_max'] and
                limits['z_min'] <= z <= limits['z_max']):
            self.get_logger().warn(f"Pose [{x:.3f}, {y:.3f}, {z:.3f}] outside workspace")
            return False
    
        if self._check_table_collision(pose):
            self.get_logger().warn(f"Pose [{x:.3f}, {y:.3f}, {z:.3f}] would cause table collision")
            return False
        
        return True
    
    
    def _open_gripper(self):
        """Open gripper"""
        self.get_logger().info(f"Opening gripper to {self.config['gripper']['max_width']*1000:.0f}mm")
        
        goal = Move.Goal()
        goal.width = self.config['gripper']['max_width']
        goal.speed = self.config['gripper']['speed']
        
        self.move_client.send_goal_async(goal)
    
    def _close_gripper(self):
        """Close gripper"""
        self.get_logger().info(f"Grasping with {self.config['gripper']['force']:.0f}N force")
        
        goal = Grasp.Goal()
        goal.width = self.config['gripper']['goal_width']
        goal.speed = self.config['gripper']['speed']
        goal.force = self.config['gripper']['force']
        goal.epsilon.inner = self.config['gripper']['epsilon_inner']
        goal.epsilon.outer = self.config['gripper']['epsilon_outer']
        
        self.grasp_client.send_goal_async(goal)
    
    def _move_and_wait(self, pose, timing_key):
        """Move to pose and wait"""
        self.cartesian_target_publisher.publish(pose)
        self.get_logger().info(f"Moving to: [{pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f}]")
        time.sleep(self.config['timing'][timing_key])
    
    def _gripper_and_wait(self, open_gripper, timing_key):
        """Control gripper and wait"""
        if open_gripper:
            self._open_gripper()
        else:
            self._close_gripper()
        time.sleep(self.config['timing'][timing_key])
    
    def _publish_status(self, success):
        """Publish grasp status"""
        msg = Bool()
        msg.data = success
        self.status_publisher.publish(msg)
    
    def grasp_pose_callback(self, msg):
        """Handle new grasp pose from vision system"""
        self.get_logger().info(f"Received grasp pose: [{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]")
        
        self.latest_grasp_pose = msg
        
        if self.is_executing:
            self.get_logger().info("New pose received - will abort and restart")
            self.should_abort = True
            return
        
        if self._is_pose_reachable(msg.pose):
            self.execute_grasp_sequence()
        else:
            self.get_logger().error("Pose not reachable")
    
    def execute_grasp_sequence(self):
        """Execute complete grasp sequence"""
        if not self.latest_grasp_pose:
            self.get_logger().error("No grasp pose available")
            return
        
        try:
            self.is_executing = True
            self.should_abort = False
            
            self.get_logger().info("Starting grasp sequence...")
            
            # Define sequence steps
            steps = [
                ("Move to home", lambda: self._move_and_wait(self.home_pose, 'home')),
                ("Open gripper", lambda: self._gripper_and_wait(True, 'gripper')),
                ("Pre-grasp position", lambda: self._move_with_offset(self.config['offsets']['pre_grasp'], 'pre_grasp')),
                ("Approach position", lambda: self._move_with_offset(self.config['offsets']['approach'], 'approach')),
                ("Final grasp position", lambda: self._move_with_offset(0.0, 'grasp')),
                ("Close gripper", lambda: self._gripper_and_wait(False, 'close_gripper')),
                ("Lift object", lambda: self._move_with_offset(self.config['offsets']['lift'], 'lift')),
                ("Safe position", lambda: self._move_and_wait(self.home_pose, 'safe'))
            ]
            
            # Execute each step
            for step_name, action in steps:
                if self.should_abort:
                    return self._restart_with_new_pose()
                
                self.get_logger().info(f"Step: {step_name}")
                action()
            
            self._publish_status(True)
            self.get_logger().info("Grasp sequence completed successfully!")
            
        except Exception as e:
            self.get_logger().error(f"Grasp execution failed: {e}")
            self._publish_status(False)
        finally:
            self.is_executing = False
    
    def _move_with_offset(self, z_offset, timing_key):
        """Move to grasp pose with Z offset"""
        pose = self._create_offset_pose(self.latest_grasp_pose, z_offset)
        self._move_and_wait(pose, timing_key)
    
    def _restart_with_new_pose(self):
        """Restart sequence with latest pose"""
        self.get_logger().info("Restarting with updated pose")
        self.is_executing = False
        if (self.latest_grasp_pose and 
            self._is_pose_reachable(self.latest_grasp_pose.pose)):
            self.execute_grasp_sequence()


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()