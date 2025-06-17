from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'zed_pose_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gejan',
    maintainer_email='gejan@student.ethz.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_transform_fusion_node = zed_pose_estimation.pointcloud_transform_fusion_node:main',
            'camera_manager_node = zed_pose_estimation.camera_manager_node:main',
            'camera_manager = zed_pose_estimation.camera_manager:main',
            'pose_estimation_utils = zed_pose_estimation.pose_estimation_utils:main',
            'pointcloud_visualizer_node = zed_pose_estimation.pointcloud_visualizer_node:main',
            'object_segmentation_node = zed_pose_estimation.object_segmentation_node:main',
            'icp_pose_estimation_node = zed_pose_estimation.icp_pose_estimation_node:main',
            'gpu_segmentation_node = zed_pose_estimation.gpu_segmentation_node:main',
            'zed_gpu_node = zed_pose_estimation.zed_gpu_node:main',
            'superquadric_grasp_node = zed_pose_estimation.superquadric_grasp_node:main',
            'grasp_executor = zed_pose_estimation.grasp_executor:main',
    },
)
