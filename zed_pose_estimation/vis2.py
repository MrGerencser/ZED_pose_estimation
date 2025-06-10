import numpy as np
import open3d as o3d
import trimesh.transformations as tra
import os
from scipy.spatial.transform import Rotation


def get_gripper_control_points_o3d(
    grasp,
    show_sweep_volume=False,
    color=(0.2, 0.8, 0),
    finger_tip_to_origin=True):
    """
    Simple, clear gripper visualization
    
    Gripper coordinate system (Visualizer's internal convention for construction):
    - X-axis: approach direction (gripper moves along this)
    - Y-axis: side direction  
    - Z-axis: closing direction (fingers open/close along this)
    
    Args:
        grasp: [4, 4] transformation matrix from planner (defines planner's gripper frame in world)
        show_sweep_volume: bool, whether to show sweep volume
        color: RGB color tuple
        finger_tip_to_origin: bool, if True, finger tips at coordinate frame origin
    """
    meshes = []
    
    # Extract position and rotation from grasp matrix (which is T_world_plannerFrame)
    grasp_pos = grasp[:3, 3].copy()
    grasp_rot_planner_frame = grasp[:3, :3].copy() # This is R_world_plannerFrame

    # Define the rotation from the Visualizer's conventional local frame 
    # (X=approach, Y=side, Z=closing) to the Planner's local gripper frame
    # Planner's local frame: X_p=side, Y_p=closing, Z_p=anti-approach (so -Z_p is approach)
    # R_planner_from_visualizer maps v_visualizer to v_planner
    # Column 1 (X_visualizer in Planner coords): [0,0,-1] (Viz Approach -> Planner -Z_p)
    # Column 2 (Y_visualizer in Planner coords): [1,0,0]  (Viz Side     -> Planner  X_p)
    # Column 3 (Z_visualizer in Planner coords): [0,1,0]  (Viz Closing  -> Planner  Y_p)
    R_planner_from_visualizer = np.array([
        [0, -1, 0],
        [0, 0, 1],
        [-1, 0, 0]
    ], dtype=float)

    # Effective rotation matrix for visualization: R_world_visualizerFrame
    # This matrix's columns will be [approach_world, side_world, closing_world]
    # as expected by the visualizer's geometry construction logic.
    grasp_rot = grasp_rot_planner_frame @ R_planner_from_visualizer
    
    # Gripper dimensions
    finger_length = 0.041      # Length of each finger
    finger_width = 0.008      # Thickness of fingers
    jaw_width = 0.08          # Distance between fingers (fully open)
    palm_size = 0.02          # Size of gripper palm/base
    
    if finger_tip_to_origin:
        # Finger tips should be at grasp_pos (the coordinate frame origin)
        finger_tip_pos = grasp_pos
        # Palm is offset back along approach direction (-X)
        palm_pos = grasp_pos + grasp_rot @ np.array([-finger_length, 0, 0])
    else:
        # Palm at grasp_pos, fingers extend forward
        palm_pos = grasp_pos
        finger_tip_pos = grasp_pos + grasp_rot @ np.array([finger_length, 0, 0])
    
    # 1. Create gripper palm/base (green box)
    palm = o3d.geometry.TriangleMesh.create_box(
        width=palm_size,    # Along X (approach)
        height=palm_size,   # Along Y (side)  
        depth=jaw_width     # Along Z (closing direction)
    )
    # Center the palm box
    palm.translate([-palm_size/2, -palm_size/2, -jaw_width/2])
    # Orient and position the palm
    palm_transform = np.eye(4)
    palm_transform[:3, :3] = grasp_rot
    palm_transform[:3, 3] = palm_pos
    palm.transform(palm_transform)
    palm.paint_uniform_color(color)
    palm.compute_vertex_normals()
    meshes.append(palm)
    
    # 2. Create right finger (positive Z direction)
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_length,   # Along X (extends forward)
        height=finger_width,   # Along Y (thin)
        depth=finger_width     # Along Z (thin)
    )
    # Position finger: extends from palm to tip, offset in +Z direction
    right_finger_center = (palm_pos + finger_tip_pos) / 2 + grasp_rot @ np.array([0, 0, jaw_width/2 - finger_width/2])
    right_finger.translate([-finger_length/2, -finger_width/2, -finger_width/2])
    right_finger_transform = np.eye(4)
    right_finger_transform[:3, :3] = grasp_rot
    right_finger_transform[:3, 3] = right_finger_center
    right_finger.transform(right_finger_transform)
    right_finger.paint_uniform_color(color)
    right_finger.compute_vertex_normals()
    meshes.append(right_finger)
    
    # 3. Create left finger (negative Z direction)
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_length,   # Along X (extends forward)
        height=finger_width,   # Along Y (thin)
        depth=finger_width     # Along Z (thin)
    )
    # Position finger: extends from palm to tip, offset in -Z direction
    left_finger_center = (palm_pos + finger_tip_pos) / 2 + grasp_rot @ np.array([0, 0, -jaw_width/2 + finger_width/2])
    left_finger.translate([-finger_length/2, -finger_width/2, -finger_width/2])
    left_finger_transform = np.eye(4)
    left_finger_transform[:3, :3] = grasp_rot
    left_finger_transform[:3, 3] = left_finger_center
    left_finger.transform(left_finger_transform)
    left_finger.paint_uniform_color(color)
    left_finger.compute_vertex_normals()
    meshes.append(left_finger)
    
    # 4. Add finger tip markers (red spheres/boxes)
    if finger_tip_to_origin:
        dark_color = np.array(color) * 0.5  # Slightly darker 
        # Right finger tip
        right_tip_pos = finger_tip_pos + grasp_rot @ np.array([0, 0, jaw_width/2 - finger_width/2])
        
        # Option 1: Small box
        right_tip = o3d.geometry.TriangleMesh.create_box(width=0.009, height=0.009, depth=0.009)
        right_tip.translate([-0.0045, -0.0045, -0.0045])  # Center the box
        right_tip.translate(right_tip_pos)
        
        # Option 2: Small sphere (better for finger tips)
        # right_tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        # right_tip.translate(right_tip_pos)
        
        right_tip.paint_uniform_color(dark_color)
        right_tip.compute_vertex_normals()
        meshes.append(right_tip)
        
        # Left finger tip  
        left_tip_pos = finger_tip_pos + grasp_rot @ np.array([0, 0, -jaw_width/2 + finger_width/2])
        
        # Option 1: Small box
        left_tip = o3d.geometry.TriangleMesh.create_box(width=0.009, height=0.009, depth=0.009)
        left_tip.translate([-0.0045, -0.0045, -0.0045])  # Center the box
        left_tip.translate(left_tip_pos)
        
        # Option 2: Small sphere (better for finger tips)
        # left_tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        # left_tip.translate(left_tip_pos)
        
        left_tip.paint_uniform_color(dark_color)
        left_tip.compute_vertex_normals()
        meshes.append(left_tip)
    
    # 5. Add approach direction indicator (blue arrow)
    arrow_length = 0.04
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002,
        cone_radius=0.004, 
        cylinder_height=arrow_length * 0.7,
        cone_height=arrow_length * 0.3
    )
    # Arrow points along +X (approach direction)
    arrow_transform = np.eye(4)
    arrow_transform[:3, :3] = grasp_rot
    if finger_tip_to_origin:
        arrow_start = finger_tip_pos - grasp_rot @ np.array([arrow_length, 0, 0])
    else:
        arrow_start = palm_pos - grasp_rot @ np.array([arrow_length, 0, 0])
    arrow_transform[:3, 3] = arrow_start
    arrow.transform(arrow_transform)
    arrow.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    arrow.compute_vertex_normals()
    meshes.append(arrow)
    
    # 6. Add sweep volume if requested
    if show_sweep_volume:
        sweep_volume = o3d.geometry.TriangleMesh.create_box(
            width=finger_length,
            height=0.02,
            depth=jaw_width
        )
        sweep_volume.translate([-finger_length/2, -0.01, -jaw_width/2])
        sweep_transform = np.eye(4)
        sweep_transform[:3, :3] = grasp_rot
        if finger_tip_to_origin:
            sweep_transform[:3, 3] = finger_tip_pos + grasp_rot @ np.array([-finger_length/2, 0, 0])
        else:
            sweep_transform[:3, 3] = palm_pos + grasp_rot @ np.array([finger_length/2, 0, 0])
        sweep_volume.transform(sweep_transform)
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
    # Example 1: Create some demo data
    print("Demo 1: Basic cube with single grasp")
    
    # Create cube point cloud
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
    
    # Define grasp pose
    grasp_pose = {
        'position': [-0.08, 0, 0],
        'quaternion': [0, 0, 0, 1]  # [x, y, z, w]
    }
    
    visualize_superquadric_grasps(
        point_cloud_data=cube_points,
        superquadric_params=sq_params,
        grasp_poses=grasp_pose,
        window_name="Demo 1: Basic Visualization"
    )
    
    # Example 2: Multiple grasps
    print("\nDemo 2: Multiple grasp poses")
    
    grasp_poses = [
        {'position': [-0.08, 0, 0], 'quaternion': [0, 0, 0, 1]},
        {'position': [0.08, 0, 0], 'quaternion': [0, 0, 1, 0]},  # 180° rotation
        {'position': [0, -0.08, 0], 'quaternion': [0, 0, 0.707, 0.707]}  # 90° rotation
    ]
    
    visualize_superquadric_grasps(
        point_cloud_data=cube_points,
        superquadric_params=sq_params,
        grasp_poses=grasp_poses,
        show_sweep_volume=False,
        window_name="Demo 2: Multiple Grasps"
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