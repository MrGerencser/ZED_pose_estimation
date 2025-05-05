# ZED Pose Estimation

A ROS2 package for pose estimation and point cloud fusion using ZED stereo cameras with a Franka robot.

## Prerequisites

1. Install ZED SDK from [stereolabs.com](https://www.stereolabs.com/developers/release/)

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
   git clone <repository-url> zed_pose_estimation
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
   - Edit these values in `launch/pointcloud_fusion.launch.py` if needed

## Usage

### Running Point Cloud Fusion

Launch the point cloud fusion node:
```bash
ros2 launch zed_pose_estimation pointcloud_fusion.launch.py
```

### Configuration Options

The pointcloud fusion node accepts the following parameters:

- `voxel_size`: Set the voxel size for point cloud downsampling (default: 0.002)
- `max_points_per_cloud`: Maximum number of points per cloud (default: 100000)
- `use_workspace_crop`: Enable workspace cropping (default: True)
- `visualize`: Enable visualization (default: False)
- `visualize_interval`: Visualization update interval in seconds (default: 5.0)
- `transform_and_fuse_of_full_clouds`: Enable full cloud transformation and fusion (default: False)

You can modify these parameters in the launch file or pass them as command-line arguments.

## Camera Transformation

The camera transformation matrix is defined in `config/transform.yaml`. Modify this file if you need to adjust the relative positioning of the cameras.

## Development

This package follows ROS2 coding standards. To run the linters:

```bash
colcon test --packages-select zed_pose_estimation
```