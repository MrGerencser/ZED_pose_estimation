import numpy as np
import open3d as o3d
import trimesh.transformations as tra
import os
from scipy.spatial.transform import Rotation

# Add this to the top of vis2.py
from zed_pose_estimation.superquadric_grasp_planner import Gripper

def get_gripper_control_points_o3d(
    grasp_transform,
    gripper: Gripper = None,
    show_sweep_volume=False,
    color=(0.2, 0.8, 0),
    finger_tip_to_origin=True):
    """
    Create gripper visualization using the actual Gripper class
    """
    if gripper is None:
        gripper = Gripper()  # Use default gripper
    
    # FIXED: Handle the new 4-mesh return
    gripper_meshes = gripper.make_open3d_meshes(colour=color)
    
    # Check how many meshes we got (for backward compatibility)
    if len(gripper_meshes) == 4:
        finger_L, finger_R, connector, back_Z = gripper_meshes
        all_gripper_parts = [finger_L, finger_R, back_Z, connector]
    else:
        print(f"Warning: Unexpected number of gripper meshes: {len(gripper_meshes)}")
        all_gripper_parts = gripper_meshes
    
    # Transform all meshes to world coordinates
    meshes = []
    for mesh in all_gripper_parts:
        # Apply the grasp transformation
        mesh_world = mesh.transform(grasp_transform)
        meshes.append(mesh_world)
    
    # Add coordinate frame at grasp pose
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    coord_frame.transform(grasp_transform)
    meshes.append(coord_frame)
    
    # Add approach direction arrow (blue)
    arrow_length = 0.001
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002,
        cone_radius=0.004,
        cylinder_height=arrow_length * 0.7,
        cone_height=arrow_length * 0.3
    )
    
    # Arrow points along gripper's approach direction (-Z in gripper frame)
    approach_dir = grasp_transform[:3, :3] @ gripper.approach_axis  # -Z axis
    arrow_pos = grasp_transform[:3, 3] + approach_dir * arrow_length
    
    # Orient arrow along approach direction
    z_axis = np.array([0, 0, 1])
    if not np.allclose(approach_dir, z_axis):
        if np.allclose(approach_dir, -z_axis):
            arrow_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            v = np.cross(z_axis, approach_dir)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, approach_dir)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            arrow_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    else:
        arrow_rotation = np.eye(3)
    
    arrow_transform = np.eye(4)
    arrow_transform[:3, :3] = arrow_rotation
    arrow_transform[:3, 3] = arrow_pos
    arrow.transform(arrow_transform)
    arrow.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    meshes.append(arrow)
    
    # Add closing direction arrow (red)
    closing_dir = grasp_transform[:3, :3] @ gripper.lambda_local  # Y axis
    closing_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002,
        cone_radius=0.004,
        cylinder_height=0.03,
        cone_height=0.008
    )
    
    # Orient closing arrow
    if not np.allclose(closing_dir, z_axis):
        if np.allclose(closing_dir, -z_axis):
            closing_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            v = np.cross(z_axis, closing_dir)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, closing_dir)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            closing_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    else:
        closing_rotation = np.eye(3)
    
    closing_arrow_transform = np.eye(4)
    closing_arrow_transform[:3, :3] = closing_rotation
    closing_arrow_transform[:3, 3] = grasp_transform[:3, 3] + closing_dir * 0.03
    closing_arrow.transform(closing_arrow_transform)
    closing_arrow.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    meshes.append(closing_arrow)
    
    # Add sweep volume if requested
    if show_sweep_volume:
        # Create sweep volume with correct dimensions
        # The sweep volume represents the space the gripper occupies during closing
        sweep_volume = o3d.geometry.TriangleMesh.create_box(
            width=gripper.thickness * 2,     # X: gripper thickness (finger width)
            height=gripper.max_open,         # Y: closing direction (finger separation)
            depth=gripper.jaw_len            # Z: finger length (approach direction)
        )
        
        # Center the sweep volume in gripper local coordinates
        # The box is created with one corner at origin, so we need to center it
        sweep_volume.translate([
            -gripper.thickness,              # Center in X (finger thickness)
            -gripper.max_open / 2,          # Center in Y (between fingers)
            -gripper.jaw_len                # Start at finger tips (Z=0), extend backward
        ])
        
        # Apply the grasp transformation to position in world coordinates
        sweep_transform = grasp_transform.copy()
        sweep_volume.transform(sweep_transform)
        
        # Make it semi-transparent blue
        sweep_volume.paint_uniform_color([0.2, 0.2, 0.8])
        meshes.append(sweep_volume)
    
    return meshes
        


def create_cubic_object(size=0.05, center=(0, 0, 0), color=(0.8, 0.2, 0.2)):
    """
    Create a cubic object (normally would use your own point cloud)
    """
    # Create a box/cube
    cube = o3d.geometry.TriangleMesh.create_box(
        width=size, 
        height=size, 
        depth=size
    )
    
    # Center the cube at the desired position
    cube.translate([-size/2, -size/2, -size/2])  # Center at origin first
    cube.translate(center)  # Then move to desired center
    
    cube.paint_uniform_color(color)
    cube.compute_vertex_normals()
    return cube


def superquadric_to_open3d_mesh(shape, scale, euler, translation, threshold=1e-2, num_limit=10000, arclength=0.2):
    """
    Convert superquadric parameters to Open3D mesh
    
    Args:
        shape: [ε1, ε2] shape parameters
        scale: [a1, a2, a3] scale parameters
        euler: [roll, pitch, yaw] rotation in radians
        translation: [tx, ty, tz] translation
        threshold: sampling threshold
        num_limit: maximum number of sampling points
        arclength: arc length for sampling
    
    Returns:
        Open3D triangle mesh or None if failed
    """
    try:
        from EMS.utilities import uniformSampledSuperellipse, create_mesh_from_grid
        
        # Avoid numerical instability in sampling
        shape_stable = shape.copy()
        if shape_stable[0] < 0.007:
            shape_stable[0] = 0.007
        if shape_stable[1] < 0.007:
            shape_stable[1] = 0.007
        
        # Sampling points in superellipse    
        point_eta = uniformSampledSuperellipse(shape_stable[0], [1, scale[2]], threshold, num_limit, arclength)
        point_omega = uniformSampledSuperellipse(shape_stable[1], [scale[0], scale[1]], threshold, num_limit, arclength)
        
        # Create rotation matrix from Euler angles
        R = Rotation.from_euler('xyz', euler).as_matrix()
        
        # Preallocate meshgrid
        x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
        y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
        z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))

        for m in range(np.shape(point_omega)[1]):
            for n in range(np.shape(point_eta)[1]):
                point_temp = np.zeros(3)
                point_temp[0:2] = point_omega[:, m] * point_eta[0, n]
                point_temp[2] = point_eta[1, n]
                point_temp = R @ point_temp + translation

                x_mesh[m, n] = point_temp[0]
                y_mesh[m, n] = point_temp[1]
                z_mesh[m, n] = point_temp[2]
        
        # Create Open3D mesh from the grid
        mesh = create_mesh_from_grid(x_mesh, y_mesh, z_mesh)
        return mesh
        
    except ImportError:
        print("Warning: EMS utilities not available. Creating approximation.")
        return create_superquadric_approximation(shape, scale, euler, translation)
    except Exception as e:
        print(f"Error creating superquadric mesh: {e}")
        return None


def create_superquadric_approximation(shape, scale, euler, translation):
    """
    Create an approximate superquadric mesh using basic primitives if EMS is not available
    """
    try:
        # Create ellipsoid as approximation
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        
        # Scale to match superquadric dimensions
        mesh.scale(scale[0], center=(0, 0, 0))
        mesh.scale([1, scale[1]/scale[0], scale[2]/scale[0]], center=(0, 0, 0))
        
        # Rotate
        R = Rotation.from_euler('xyz', euler).as_matrix()
        mesh.rotate(R, center=(0, 0, 0))
        
        # Translate
        mesh.translate(translation)
        
        return mesh
    except Exception as e:
        print(f"Error creating superquadric approximation: {e}")
        return None


def load_point_cloud(file_path):
    """
    Load point cloud from file
    
    Args:
        file_path: Path to point cloud file (.ply, .pcd, etc.)
    
    Returns:
        Open3D point cloud or None if failed
    """
    try:
        if not os.path.exists(file_path):
            print(f"Point cloud file not found: {file_path}")
            return None
        
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"Point cloud file is empty: {file_path}")
            return None
        
        print(f"Loaded point cloud with {len(pcd.points)} points from {file_path}")
        return pcd
    
    except Exception as e:
        print(f"Error loading point cloud from {file_path}: {e}")
        return None


def parse_grasp_pose(grasp_input):
    """
    Parse grasp pose from various input formats
    
    Args:
        grasp_input: Can be:
            - 4x4 numpy array (transformation matrix)
            - dict with 'position' and 'orientation' keys
            - tuple/list of (position, quaternion) where position is [x,y,z] and quaternion is [x,y,z,w]
    
    Returns:
        4x4 transformation matrix
    """
    try:
        if isinstance(grasp_input, np.ndarray) and grasp_input.shape == (4, 4):
            return grasp_input
        
        elif isinstance(grasp_input, dict):
            position = np.array(grasp_input['position'])
            if 'quaternion' in grasp_input:
                quat = np.array(grasp_input['quaternion'])  # [x, y, z, w]
                R = Rotation.from_quat(quat).as_matrix()
            elif 'rotation_matrix' in grasp_input:
                R = np.array(grasp_input['rotation_matrix'])
            elif 'euler' in grasp_input:
                euler = np.array(grasp_input['euler'])
                R = Rotation.from_euler('xyz', euler).as_matrix()
            else:
                R = np.eye(3)
            
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = position
            return transform
        
        elif isinstance(grasp_input, (tuple, list)) and len(grasp_input) == 2:
            position, quat = grasp_input
            position = np.array(position)
            quat = np.array(quat)  # [x, y, z, w]
            R = Rotation.from_quat(quat).as_matrix()
            
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = position
            return transform
        
        else:
            print("Invalid grasp pose format. Using identity matrix.")
            return np.eye(4)
    
    except Exception as e:
        print(f"Error parsing grasp pose: {e}")
        return np.eye(4)


def visualize_superquadric_grasps(
    point_cloud_path=None,
    point_cloud_data=None,
    superquadric_params=None,
    grasp_poses=None,
    show_sweep_volume=False,
    gripper_colors=None,
    window_name="Superquadric Grasp Visualization",
    align_finger_tips=True):  # 🔧 NEW parameter
    """
    Comprehensive visualization of point clouds, superquadrics, and grasp poses
    
    Args:
        point_cloud_path: Path to point cloud file
        point_cloud_data: Direct point cloud data as numpy array (Nx3)
        superquadric_params: Dict with keys 'shape', 'scale', 'euler', 'translation'
                           or list of such dicts for multiple superquadrics
        grasp_poses: Single grasp pose or list of grasp poses (4x4 matrices)
        show_sweep_volume: Whether to show gripper sweep volumes
        gripper_colors: Colors for grippers (list of RGB tuples)
        window_name: Name for the visualization window
        align_finger_tips: If True, finger tips align with coordinate frame origin
    """
    geometries = []
    
    # Load or create point cloud
    if point_cloud_path is not None:
        pcd = load_point_cloud(point_cloud_path)
        if pcd is not None:
            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for point cloud
            geometries.append(pcd)
    elif point_cloud_data is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for point cloud
        geometries.append(pcd)
    
    # Add superquadric meshes
    if superquadric_params is not None:
        if isinstance(superquadric_params, dict):
            superquadric_params = [superquadric_params]  # Convert single to list
        
        sq_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]  # Different colors
        
        for i, sq_params in enumerate(superquadric_params):
            try:
                sq_mesh = superquadric_to_open3d_mesh(
                    shape=sq_params['shape'],
                    scale=sq_params['scale'],
                    euler=sq_params['euler'],
                    translation=sq_params['translation']
                )
                
                if sq_mesh is not None:
                    color = sq_colors[i % len(sq_colors)]
                    sq_mesh.paint_uniform_color(color)
                    geometries.append(sq_mesh)
                    # print(f"Added superquadric {i+1} with color {color}")
                else:
                    print(f"Failed to create superquadric mesh {i+1}")
            
            except Exception as e:
                print(f"Error creating superquadric {i+1}: {e}")
    
    # Add grasp visualizations
    if grasp_poses is not None:
        if not isinstance(grasp_poses, list):
            grasp_poses = [grasp_poses]  # Convert single to list
        
        # Default colors for grippers
        if gripper_colors is None:
            gripper_colors = [(0.2, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.2, 0.8), 
                            (0.8, 0.8, 0.2), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8)]
        
        for i, grasp_pose in enumerate(grasp_poses):
            try:
                # Parse grasp pose to transformation matrix
                transform_matrix = parse_grasp_pose(grasp_pose)
                
                # Get color for this gripper
                color = gripper_colors[i % len(gripper_colors)]
                
                # Create gripper visualization with aligned finger tips
                gripper_meshes = get_gripper_control_points_o3d(
                    transform_matrix,
                    show_sweep_volume=show_sweep_volume,
                    color=color,
                    finger_tip_to_origin=align_finger_tips  # 🔧 NEW parameter
                )
                
                geometries.extend(gripper_meshes)
                
                # Add coordinate frame at grasp pose
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                coord_frame.transform(transform_matrix)
                geometries.append(coord_frame)
                
            except Exception as e:
                print(f"Error creating grasp visualization {i+1}: {e}")
    
    # Add main coordinate frame
    main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(main_coord_frame)
    
    # Visualize all geometries
    if len(geometries) > 0:       
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            zoom=0.7,
            front=[0, -1, 0],
            lookat=[0, 0, 0],
            up=[0, 0, 1]
        )
    else:
        print("No geometries to visualize!")


def demo_superquadric_visualization():
    """
    Demo showing how to use the enhanced visualization functions
    """
    # Create cube point cloud (shared for all demos)
    cube_points = []
    for x in np.linspace(-0.03, 0.03, 10):
        for y in np.linspace(-0.03, 0.03, 10):
            for z in np.linspace(-0.03, 0.03, 10):
                if abs(x) > 0.025 or abs(y) > 0.025 or abs(z) > 0.025:  # Hollow cube
                    cube_points.append([x, y, z])
    
    cube_points = np.array(cube_points)
    
    # Define superquadric parameters (box-like)
    sq_params = {
        'shape': np.array([0.1, 0.1]),      # ε1, ε2 (box-like)
        'scale': np.array([0.03, 0.03, 0.03]),  # a1, a2, a3
        'euler': np.array([0, 0, 0]),           # roll, pitch, yaw
        'translation': np.array([0, 0, 0])      # tx, ty, tz
    }
    
    # Example 1: Single grasp using transformation matrix
    print("Demo 1: Basic cube with single grasp (transformation matrix)")
    
    # GRASP FROM LEFT: gripper at (-0.08, 0, 0) pointing toward +X
    grasp_single = np.array([
        [0, 0, 1, -0.08],   # X-axis of gripper points in world +Z  
        [0, 1, 0, 0],       # Y-axis of gripper points in world +Y (closing)
        [-1, 0, 0, 0],      # Z-axis of gripper points in world -X (approach)
        [0, 0, 0, 1]
    ])
    
    visualize_superquadric_grasps(
        point_cloud_data=cube_points,
        superquadric_params=sq_params,
        grasp_poses=grasp_single,  # Single transformation matrix
        show_sweep_volume=True,
        window_name="Demo 1: Basic Visualization (Transformation Matrix)"
    )
    
    # Example 2: Multiple grasps using transformation matrices
    print("\nDemo 2: Multiple grasp poses (transformation matrices)")
    
    # GRASP FROM LEFT: gripper at (-0.08, 0, 0) pointing toward +X
    grasp_left = np.array([
        [0, 0, 1, -0.08],   # X-axis of gripper points in world +Z  
        [0, 1, 0, 0],       # Y-axis of gripper points in world +Y (closing)
        [-1, 0, 0, 0],      # Z-axis of gripper points in world -X (approach)
        [0, 0, 0, 1]
    ])
    
    # GRASP FROM RIGHT: gripper at (+0.08, 0, 0) pointing toward -X
    grasp_right = np.array([
        [0, 0, -1, 0.08],   # X-axis of gripper points in world -Z
        [0, 1, 0, 0],       # Y-axis of gripper points in world +Y (closing)  
        [1, 0, 0, 0],       # Z-axis of gripper points in world +X (approach)
        [0, 0, 0, 1]
    ])
    
    # GRASP FROM FRONT: gripper at (0, -0.08, 0) pointing toward +Y
    grasp_front = np.array([
        [1, 0, 0, 0],       # X-axis of gripper points in world +X
        [0, 0, 1, -0.08],   # Y-axis of gripper points in world +Z (closing)
        [0, -1, 0, 0],      # Z-axis of gripper points in world -Y (approach)  
        [0, 0, 0, 1]
    ])
    
    # GRASP FROM BACK: gripper at (0, +0.08, 0) pointing toward -Y  
    grasp_back = np.array([
        [1, 0, 0, 0],       # X-axis of gripper points in world +X
        [0, 0, -1, 0.08],   # Y-axis of gripper points in world -Z (closing)
        [0, 1, 0, 0],       # Z-axis of gripper points in world +Y (approach)
        [0, 0, 0, 1]
    ])
    
    grasp_poses = [grasp_left, grasp_right, grasp_front, grasp_back]
    
    visualize_superquadric_grasps(
        point_cloud_data=cube_points,
        superquadric_params=sq_params,
        grasp_poses=grasp_poses,
        show_sweep_volume=True,
        window_name="Demo 2: Multiple Grasps (Transformation Matrices)"
    )
    
    # Example 3: Multiple grasps using quaternions (for comparison)
    print("\nDemo 3: Multiple grasp poses (quaternions)")
    
    # Grasp from left (-X direction)
    grasp_left_quat = Rotation.from_euler('xyz', [np.pi, 0, np.pi/2])
    
    # Grasp from right (+X direction) 
    grasp_right_quat = Rotation.from_euler('xyz', [np.pi, 0, -np.pi/2])
    
    # Grasp from front (-Y direction)
    grasp_front_quat = Rotation.from_euler('xyz', [np.pi, 0, 0])
    
    # Grasp from back (+Y direction)
    grasp_back_quat = Rotation.from_euler('xyz', [np.pi, 0, np.pi])
    
    grasp_poses_quat = [
        {'position': [-0.08, 0, 0], 'quaternion': grasp_left_quat.as_quat()},   # From left
        {'position': [0.08, 0, 0], 'quaternion': grasp_right_quat.as_quat()},   # From right  
        {'position': [0, -0.08, 0], 'quaternion': grasp_front_quat.as_quat()},  # From front
        {'position': [0, 0.08, 0], 'quaternion': grasp_back_quat.as_quat()}     # From back
    ]
    
    visualize_superquadric_grasps(
        point_cloud_data=cube_points,
        superquadric_params=sq_params,
        grasp_poses=grasp_poses_quat,
        show_sweep_volume=True,
        window_name="Demo 3: Multiple Grasps (Quaternions)"
    )


def demo_file_visualization():
    """
    Demo showing how to load from files (requires actual files)
    """
    # Example file paths (modify these to match your actual files)
    point_cloud_file = "pointclouds/observed_cloud.ply"
    
    if os.path.exists(point_cloud_file):
        print(f"Demo 3: Loading from file {point_cloud_file}")
        
        # Example superquadric parameters (you would get these from your fitting)
        sq_params = {
            'shape': np.array([0.5, 0.5]),      # More rounded
            'scale': np.array([0.04, 0.04, 0.08]),  # Taller object
            'euler': np.array([0, 0, 0.2]),         # Slight rotation
            'translation': np.array([0, 0, 0.04])   # Raised above ground
        }
        
        # Example grasp from superquadric fitting
        grasp_pose = np.array([
            [1, 0, 0, -0.08],
            [0, 1, 0, 0],
            [0, 0, 1, 0.04],
            [0, 0, 0, 1]
        ])
        
        visualize_superquadric_grasps(
            point_cloud_path=point_cloud_file,
            superquadric_params=sq_params,
            grasp_poses=grasp_pose,
            window_name="Demo 3: File-based Visualization"
        )
    else:
        print(f"File {point_cloud_file} not found. Skipping file demo.")


if __name__ == '__main__':
    print("Running visualization demos...")
    demo_superquadric_visualization()
    demo_file_visualization()