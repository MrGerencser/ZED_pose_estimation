from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction
import os
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import yaml

# DEFINE CAMERA SERIAL NUMBERS HERE
CAMERA1_SERIAL = '33137761'  # Camera 1 serial number
CAMERA2_SERIAL = '36829049'  # Camera 2 serial number

def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('zed_pose_estimation')
    config_dir = os.path.join(pkg_share, 'config')
    
    # Launch arguments
    use_camera1 = LaunchConfiguration('use_camera1')
    use_camera2 = LaunchConfiguration('use_camera2')
    camera1_id = LaunchConfiguration('camera1_id')
    camera2_id = LaunchConfiguration('camera2_id')
    target_frame = LaunchConfiguration('target_frame')
    transform_yaml_path = LaunchConfiguration('transform_yaml_path')
    visualize = LaunchConfiguration('visualize')
    
    # Camera config file
    camera_config = os.path.join(config_dir, 'zed2i.yaml')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_camera1',
            default_value='true',
            description='Whether to use the first ZED camera'
        ),
        DeclareLaunchArgument(
            'use_camera2',
            default_value='true', 
            description='Whether to use the second ZED camera'
        ),
        DeclareLaunchArgument(
            'camera1_id',
            default_value='1',
            description='ID of the first ZED camera'
        ),
        DeclareLaunchArgument(
            'camera2_id',
            default_value='2',
            description='ID of the second ZED camera'
        ),
        DeclareLaunchArgument(
            'target_frame',
            default_value='panda_link0',
            description='The target frame for point cloud transformation'
        ),
        DeclareLaunchArgument(
            'transform_yaml_path',
            default_value=os.path.join(config_dir, 'transform.yaml'),
            description='Path to the transformation matrices YAML file'
        ),
        DeclareLaunchArgument(
            'visualize',
            default_value='true',
            description='Whether to visualize point clouds'
        ),
        
        # Launch first ZED camera node with hardcoded serial number
        Node(
            package='zed_pose_estimation',
            executable='camera_manager_node',
            name='zed_camera_1',
            namespace='zed1',
            parameters=[
                {'camera_id': camera1_id},
                {'serial_number': CAMERA1_SERIAL},  # Use constant defined at the top
                {'camera_name': 'zed1'},
                {'publish_rate': 15.0},
                {'config_file': camera_config},
                {'resolution': 'HD2K'},
                {'fps': 15.0}
            ],
            output='screen',
            condition=IfCondition(use_camera1)
        ),
        
        # Launch second ZED camera node after 3 seconds with hardcoded serial number
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='zed_pose_estimation',
                    executable='camera_manager_node',
                    name='zed_camera_2',
                    namespace='zed2',
                    parameters=[
                        {'camera_id': camera2_id},
                        {'serial_number': CAMERA2_SERIAL},  # Use constant defined at the top
                        {'camera_name': 'zed2'},
                        {'publish_rate': 15.0},
                        {'config_file': camera_config},
                        {'resolution': 'HD2K'},
                        {'fps': 15.0}
                    ],
                    output='screen',
                    condition=IfCondition(use_camera2)
                )
            ]
        ),

        # Launch point cloud transform and fusion node of the two cameras
        Node(
            package='zed_pose_estimation',
            executable='pointcloud_transform_fusion_node',
            name='pointcloud_fusion',
            parameters=[
                {'camera_topics': ['zed1/zed1/point_cloud/cloud_registered',
                                  'zed2/zed2/point_cloud/cloud_registered']},
                {'target_frame': target_frame},
                {'transform_yaml_path': transform_yaml_path},
                {'voxel_size': 0.01},
                {'max_points_per_cloud': 50000},
                {'use_workspace_crop': True},
                {'visualize': visualize},
                {'visualize_interval': 5.0},
                {'transform_and_fuse_of_full_clouds': True}
            ],
            output='screen'
        ),

        # Launch object segmentation node
        Node(
            package='zed_pose_estimation',
            executable='object_segmentation_node',
            name='object_segmentation',
            parameters=[
                {'camera_namespaces': ['zed1', 'zed2']},
                {'yolo_model_path': os.path.join(pkg_share, '/home/chris/franka_ros2_ws/src/zed_pose_estimation/models/yolo/best.pt')},
                {'confidence_threshold': 0.25},
                {'publish_rate': 1.0}
            ],
            output='screen'
        ),
        
        # Launch point cloud transform and fusion node
        Node(
            package='zed_pose_estimation',
            executable='pointcloud_transform_fusion_node',
            name='pointcloud_fusion_segmented',
            parameters=[
                # {'camera_topics': ['/zed1/zed1/point_cloud/cloud_registered', 
                #                    '/zed2/zed2/point_cloud/cloud_registered']},
                {'camera_topics': ['/zed1/segmented_point_cloud',
                                   '/zed2/segmented_point_cloud']},
                {'target_frame': target_frame},
                {'transform_yaml_path': transform_yaml_path},
                {'voxel_size': 0.001},
                {'max_points_per_cloud': 50000},
                {'use_workspace_crop': True},
                {'visualize': visualize},
                {'visualize_interval': 5.0},
                {'transform_and_fuse_of_full_clouds': False}
            ],
            output='screen'
        ),
    ])