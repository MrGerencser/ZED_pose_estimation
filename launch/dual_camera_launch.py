from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction

def generate_launch_description():
    # Launch arguments
    camera1_id = LaunchConfiguration('camera1_id')
    camera2_id = LaunchConfiguration('camera2_id')
    
    # Create launch description
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'camera1_id',
            default_value='0',
            description='ID of the first ZED camera'
        ),
        DeclareLaunchArgument(
            'camera2_id',
            default_value='1',
            description='ID of the second ZED camera'
        ),
        
        # Launch first ZED camera node
        Node(
            package='zed_pose_estimation',
            executable='camera_manager_node',
            name='zed_camera_1',
            namespace='zed1',
            parameters=[
                {'camera_id': camera1_id},
                {'camera_name': 'zed1'},
                {'publish_rate': 10.0}
            ],
            output='screen'
        ),
        
        # Launch second ZED camera node after 5 seconds
        TimerAction(
            period=5.0,  # Wait 5 seconds before launching second camera
            actions=[
                Node(
                    package='zed_pose_estimation',
                    executable='camera_manager_node',
                    name='zed_camera_2',
                    namespace='zed2',
                    parameters=[
                        {'camera_id': camera2_id},
                        {'camera_name': 'zed2'},
                        {'publish_rate': 10.0}
                    ],
                    output='screen'
                )
            ]
        )
    ])