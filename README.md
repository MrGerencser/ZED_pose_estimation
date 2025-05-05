# ZED Pose Estimation

A ROS2 package for pose estimation and point cloud fusion using ZED stereo cameras with a Franka robot.

## Prerequisites

1. Install ZED SDK from [stereolabs.com](https://www.stereolabs.com/en-ch/developers/). This package requires ZED SDK [4.2.5](https://www.stereolabs.com/en-ch/developers/release/4.2#82af3640d775). Follow the official [Linux Installation Guide](https://www.stereolabs.com/docs/installation/linux/)

2. Install the Python API:
   ```bash
   cd /usr/local/zed  # Or wherever ZED SDK is installed
   python get_python_api.py
   ```

3. ROS2 Humble (or compatible version)

## Setup

1. Clone this repository into your ROS2 workspace:
   ```bash
   cd ~/franka_ros2_ws/src/
   git clone git@github.com:MrGerencser/zed_pose_estimation.git
   ```

2. Build the workspace:
   ```bash
   cd ~/franka_ros2_ws
   colcon build --packages-select zed_pose_estimation
   source install/setup.bash
   ```

3. Configure your cameras by setting their serial numbers in the launch file:
   - Current configuration uses cameras with serial numbers:
     - Camera 1: 33137761
     - Camera 2: 36829049
   - Edit these values in `zed_pose_estimation/zed_gpu_node.py` if needed

## Usage

### Running Point Cloud Fusion

Launch the point cloud fusion node:
```bash
ros2 run zed_pose_estimation zed_gpu_node
```

To view the output of the node:
```bash
ros2 topic echo /perception/object_pose
```

### Running with Custom Parameters

You can pass parameters directly when launching the node:
```bash
ros2 run zed_pose_estimation zed_gpu_node --ros-args -p voxel_size:=0.003 -p visualize:=true
```

### Configuration Options

The pointcloud fusion node accepts the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `voxel_size` | Set the voxel size for point cloud downsampling | 0.002 |
| `max_points_per_cloud` | Maximum number of points per cloud | 100000 |
| `use_workspace_crop` | Enable workspace cropping | True |
| `visualize` | Enable visualization | False |
| `visualize_interval` | Visualization update interval in seconds | 5.0 |
| `transform_and_fuse_of_full_clouds` | Enable full cloud transformation and fusion | False |

## Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/perception/object_pose` | `geometry_msgs/PoseStamped` | Estimated pose of detected objects |
| `/perception/point_cloud` | `sensor_msgs/PointCloud2` | Fused point cloud data |

## Camera Transformation

The camera transformation matrix is defined in `config/transform.yaml`. Modify this file if you need to adjust the relative positioning of the cameras.

## Troubleshooting

### Common Issues

1. **Camera not found**: Ensure the camera serial numbers match those in your configuration.
   ```bash
   # Check connected ZED cameras
   ls /dev/video*
   ```

2. **CUDA errors**: Verify your NVIDIA GPU drivers are correctly installed:
   ```bash
   nvidia-smi
   ```

3. **Point cloud visualization issues**: Install and use RViz2 to visualize the point clouds:
   ```bash
   ros2 run rviz2 rviz2 -d <path_to_workspace>/src/zed_pose_estimation/config/visualization.rviz
   ```

## Development

This package follows ROS2 coding standards. To run the linters:

```bash
colcon test --packages-select zed_pose_estimation
```

## License

This project is licensed under the [MIT License](LICENSE).