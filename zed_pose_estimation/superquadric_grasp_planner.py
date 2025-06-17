#!/usr/bin/env python3
"""
Superquadric-based grasp planner for ROS2 integration
Based on 'Learning‐Free Grasping of Unknown Objects Using Hidden Superquadrics'
Uses exact logic from generate_grasps.py
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from zed_pose_estimation.vis2 import get_gripper_control_points_o3d

# ------------------------------------------------------------
# 1. Superquadric & Gripper definitions  
# ------------------------------------------------------------

class Superquadric:
    def __init__(self, ε, a, euler, t):
        # ε = [ε1, ε2]; a = [ax, ay, az]; euler = [roll, pitch, yaw]; t = [tx, ty, tz]
        self.ε1, self.ε2 = ε
        self.ax, self.ay, self.az = a
        self.T = np.asarray(t, dtype=float)        # Center in world frame
        # 3×3 rotation matrix from SQ‐local → world
        self.R = R.from_euler('xyz', euler).as_matrix()

    @property
    def axes_world(self):
        """
        Return the three principal axes (λ_x, λ_y, λ_z) as columns in world frame.
        """
        return self.R  # each column is a unit vector

class Gripper:
    """
    Parallel-jaw gripper description **in its own local frame**.

    • +Y  – closing line  (jaws move ±Y)   ← matches paper
    • –Z  – approach axis (tool moves –Z) ← matches paper
    •  X  – completes the RH frame
    """

    def __init__(self,
                 jaw_len   = 0.041,   # finger length  (m)
                 max_open  = 0.080,   # maximum jaw separation (m)
                 thickness = 0.004,   # finger thickness l_j (m)
                 palm_depth= 0.010,   # distance from jaw mid-line to tool flange (m)
                 palm_width= 0.020):  # width of the aluminium bracket (m)

        # --- geometry used by the paper’s tests --------------------
        self.jaw_len      = float(jaw_len)
        self.max_open     = float(max_open)
        self.thickness    = float(thickness)   # l_j  (support & collision radius)

        # --- local coordinate conventions --------------------------
        self.lambda_local = np.array([0., 1., 0.])   # closing line
        self.approach_axis= np.array([0., 0., -1.])  # approach

        # --- extras for viz or URDF generation ---------------------
        self.palm_depth   = float(palm_depth)
        self.palm_width   = float(palm_width)

    # ---------- helper: analytic Open3D meshes ---------------------
    def make_open3d_meshes(self, colour=(0.2, 0.8, 0.2)):
        """
        Returns four cylinders in gripper-local coordinates:
            finger_L, finger_R, cross_bar_Y, back_Z
        All share the same radius = thickness and specified colour.
        """
        meshes = []

        # 1. LEFT FINGER (+Y) - create with correct dimensions directly
        finger_L = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,      # Keep original thickness
            height=self.jaw_len         # Set correct height directly
        )
        finger_L.paint_uniform_color(colour)
        finger_L.compute_vertex_normals()
        
        # Position left finger
        T = np.eye(4)
        T[1, 3] = +self.max_open / 2 + self.thickness   # y-offset
        T[2, 3] = -self.jaw_len / 2     # so tip sits at Z = 0
        finger_L.transform(T)
        meshes.append(finger_L)

        # 2. RIGHT FINGER (-Y) - create with correct dimensions directly
        finger_R = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,      # Keep original thickness
            height=self.jaw_len         # Set correct height directly
        )
        finger_R.paint_uniform_color(colour)
        finger_R.compute_vertex_normals()
        
        # Position right finger
        T = np.eye(4)
        T[1, 3] = -self.max_open / 2 - self.thickness  # y-offset
        T[2, 3] = -self.jaw_len / 2
        finger_R.transform(T)
        meshes.append(finger_R)

        # 3. CROSS-BAR (axis = Y) - create with correct dimensions directly
        cross_Y = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,                      # Keep original thickness
            height=self.max_open + 4 * self.thickness       # Span across fingers
        )
        cross_Y.paint_uniform_color(colour)
        cross_Y.compute_vertex_normals()
        
        # Rotate so cylinder's axis (default +Z) becomes +Y
        R_x_neg90 = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])
        T = np.eye(4)
        T[:3, :3] = R_x_neg90
        T[2, 3] = -self.jaw_len  # Place at finger tips (Z = -jaw_len)
        cross_Y.transform(T)
        meshes.append(cross_Y)

        # 4. BACK CYLINDER (axis = Z) - create with correct dimensions directly
        back_Z = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,      # Keep original thickness
            height=self.jaw_len      # Set correct height directly
        )
        back_Z.paint_uniform_color(colour)
        back_Z.compute_vertex_normals()
        
        # Position back cylinder
        T = np.eye(4)
        T[1, 3] = 0  # Y = 0 (centered between fingers)
        T[2, 3] = -3/2*self.jaw_len  # Behind fingers
        back_Z.transform(T)
        meshes.append(back_Z)

        return meshes



# ------------------------------------------------------------
# 2. Utility: rotation from vector u → v  
# ------------------------------------------------------------

def rotation_from_u_to_v(u, v):
    """
    Compute the shortest‐arc rotation matrix that sends unit‐vector u → unit‐vector v.
    Handles the case u ≈ -v by picking an arbitrary perpendicular axis for a 180° spin.
    """
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    dot = np.dot(u, v)
    if dot > 1 - 1e-8:
        return np.eye(3)
    if dot < -1 + 1e-8:
        # u ≈ -v: pick arbitrary perpendicular axis
        if abs(u[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, perp)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi).as_matrix()
    axis = np.cross(u, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle).as_matrix()

# ------------------------------------------------------------
# 3. Candidate Generation  
# ------------------------------------------------------------

DEG = np.pi / 180.0

def principal_axis_sweeps(S: Superquadric, G: Gripper, step_deg=10):
    """
    §2.1 – CORRECTED: Exactly implement the paper's method
    """
    candidate_R = []
    
    print(f"[INFO] Implementing paper's method: align λ with each principal axis")
    print(f"[DEBUG] Principal axes in world frame:")
    print(f"  λ_x: {S.axes_world[:, 0]}")
    print(f"  λ_y: {S.axes_world[:, 1]}")  
    print(f"  λ_z: {S.axes_world[:, 2]}")
    print(f"[DEBUG] Gripper closing line λ: {G.lambda_local}")  # FIXED: Use lambda_local
    
    # PAPER METHOD: For each principal axis λi (i = x,y,z)
    for i, axis_world in enumerate(S.axes_world.T):  # iterate over λ_x, λ_y, λ_z
        axis_names = ["λ_x", "λ_y", "λ_z"]  # FIXED: Create proper list
        axis_name = axis_names[i]  # FIXED: Use index to get correct name

        print(f"[INFO] Creating set Δ{axis_name}:")
        print(f"  Step 1: Align gripper closing line λ with {axis_name} = {axis_world}")
        
        # Step 1: Compute Rotation that aligns gripper's closing line λ → principal axis
        R_align = rotation_from_u_to_v(G.lambda_local, axis_world)  # FIXED: Use lambda_local
        
        print(f"  Step 2: Rotate gripper around {axis_name} every {step_deg}°")
        
        # Step 2: Rotate the gripper around the aligned axis in step_deg increments  
        rotations_in_set = 0
        for theta_deg in range(0, 360, step_deg):
            theta_rad = theta_deg * DEG
            
            # Create rotation around the principal axis
            R_spin = R.from_rotvec(axis_world * theta_rad).as_matrix()
            
            # Combined rotation: first align, then spin
            R_final = R_spin @ R_align
            
            candidate_R.append(R_final)
            rotations_in_set += 1
            
            # Debug first few rotations
            if theta_deg < 30:  # Show first 3 rotations for verification
                closing_dir_after = R_final @ G.lambda_local  # FIXED: Use lambda_local
                alignment_check = np.dot(closing_dir_after, axis_world)
                print(f"    θ={theta_deg:3d}°: closing_dir={closing_dir_after}, alignment={alignment_check:.3f}")
        
        print(f"  → Created set Δ{axis_name} with {rotations_in_set} poses")
    
    expected_total = 3 * (360 // step_deg)  # 3 axes × (360/step_deg) rotations
    print(f"[INFO] Paper method complete: {len(candidate_R)} candidates (expected: {expected_total})")
    
    # VERIFICATION: Check that closing line is always aligned with one of the principal axes
    print(f"[DEBUG] Verification - checking alignment of first few candidates:")
    for i in range(min(9, len(candidate_R))):  # Check first 3 from each axis
        R_candidate = candidate_R[i]
        closing_dir = R_candidate @ G.lambda_local  # FIXED: Use lambda_local
        
        # Check alignment with each principal axis
        alignments = []
        for j, axis in enumerate(S.axes_world.T):
            alignment = abs(np.dot(closing_dir, axis))
            alignments.append(alignment)
        
        best_axis_idx = np.argmax(alignments)
        best_alignment = alignments[best_axis_idx]
        axis_name = ["λx", "λy", "λz"][best_axis_idx]
        
        print(f"  Candidate {i+1}: best aligned with {axis_name} (alignment={best_alignment:.6f})")
        
        if best_alignment < 0.99:  # Should be nearly 1.0 for perfect alignment
            print(f"WARNING: Poor alignment detected!")
    
    return candidate_R
    

def extra_sweeps_special_shapes(S: Superquadric, base_R_list, G: Gripper):
    """
    §2.2 – For prism-like (ε1→0) or cuboid-like (ε2→0): slide gripper along SQ-local axes in 15mm grid.
            For cylinder-like (ε1→0, ε2=1, ax=ay): rotate about λ_z in π/8 increments.
    """
    R_list_full = list(base_R_list)
    poses_offsets = []
    return R_list_full, poses_offsets

    # Heuristic thresholds (from the paper)
    prism_like   = (S.ε1 < 0.3)  # ε1 → 0
    cuboid_like  = (S.ε2 < 0.3)  # ε2 → 0
    
    #   FIX: Complete cylinder condition from paper
    cylinder_like = (S.ε1 < 0.3) and (abs(S.ε2 - 1.0) < 0.1) and (abs(S.ax - S.ay) < 0.01)
    
    print(f"[DEBUG] Shape analysis: ε1={S.ε1:.3f}, ε2={S.ε2:.3f}, ax={S.ax:.3f}, ay={S.ay:.3f}")
    print(f"[DEBUG] prism_like={prism_like}, cuboid_like={cuboid_like}, cylinder_like={cylinder_like}")

    # SQ-local axes in world
    λx_w, λy_w, λz_w = S.axes_world[:, 0], S.axes_world[:, 1], S.axes_world[:, 2]

    # Grid step (15 mm)
    step = 0.015

    def linspace_clamped(extent):
        if extent < 1e-6:
            return np.array([0.0])
        n = int(np.floor(extent / step))
        vals = np.arange(-n, n+1) * step
        return vals

    #   Handle cylinder case FIRST (most specific)
    if cylinder_like:
        print(f"[INFO] Detected cylinder shape - adding π/8 rotations around z-axis")
        
        # For cylinders: only rotate around z-axis (no translation)
        seen = set()
        axis_world = λz_w
        R_align = rotation_from_u_to_v(G.lambda_local, axis_world)
        
        # π/8 = 22.5 degrees intervals as per paper
        for extra_angle in np.arange(0, 360, 22.5):
            R_extra = R.from_rotvec(axis_world * (extra_angle * DEG)).as_matrix()
            R_new = R_extra @ R_align
            key = tuple(np.round(R_new, 6).ravel())
            if key not in seen:
                seen.add(key)
                R_list_full.append(R_new)
                poses_offsets.append((R_new, np.zeros(3)))
        
        print(f"[INFO] Added {len(poses_offsets)} cylinder rotations")
    
    # Handle prism/cuboid cases (translation grids)
    elif prism_like or cuboid_like:
        print(f"[INFO] Detected prism/cuboid shape - adding translation grid")
        
        z_vals = linspace_clamped(S.az) if prism_like else np.array([0.0])
        x_vals = linspace_clamped(S.ax) if (cuboid_like or prism_like) else np.array([0.0])
        y_vals = linspace_clamped(S.ay) if (cuboid_like or prism_like) else np.array([0.0])

        for Rg in base_R_list:
            for dz in z_vals:
                for dx in x_vals:
                    for dy in y_vals:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # Skip origin (already in base_R_list)
                        Δt_world = dx * λx_w + dy * λy_w + dz * λz_w
                        poses_offsets.append((Rg, Δt_world))
        
        print(f"[INFO] Added {len(poses_offsets)} grid translations")

    return R_list_full, poses_offsets

def make_world_pose(S: Superquadric, Rg, Δt=np.zeros(3)):
    """
    Place gripper closing point P_G at SQ center + Δt (world).
    Return (R_world, t_world).
    """
    return (Rg, S.T + Δt)

# ------------------------------------------------------------
# 4. Candidate Filtering
# ------------------------------------------------------------

def support_test(R, t, S, G, kdtree: KDTree, κ=12, r_support=None, h_support=0.02, debug_mode=False, max_debug_calls=5):
    """
    True cylinder support test with optional visualization:
    - r_support: cylinder radius (defaults to half jaw width)
    - h_support: half cylinder height
    """
    # Closing direction in world
    closing_dir = R @ G.lambda_local
    closing_dir = closing_dir / np.linalg.norm(closing_dir)

    # Default support radius: half the jaw opening
    if r_support is None:
        r_support = 3 * 0.003      # e.g. 3 × voxel_size (≈9 mm)
    if h_support is None:
        h_support = G.jaw_len      # full finger length

    # two finger-contact points on the SQ surface, along closing_dir
    tip1 = t + closing_dir * h_support
    tip2 = t - closing_dir * h_support

    # Get all points
    X = kdtree.data
    
    # Compute point projections onto closing axis for tip1
    rel1 = X - tip1
    proj1 = np.dot(rel1, closing_dir)
    radial1 = np.linalg.norm(rel1 - np.outer(proj1, closing_dir), axis=1)
    mask1 = (np.abs(proj1) <= h_support) & (radial1 <= r_support)
    cnt1 = np.count_nonzero(mask1)

    # Compute point projections onto closing axis for tip2
    rel2 = X - tip2
    proj2 = np.dot(rel2, closing_dir)
    radial2 = np.linalg.norm(rel2 - np.outer(proj2, closing_dir), axis=1)
    mask2 = (np.abs(proj2) <= h_support) & (radial2 <= r_support)
    cnt2 = np.count_nonzero(mask2)

    support_result = (cnt1 >= κ) and (cnt2 >= κ)

    # --- DEBUG VISUALIZATION ---
    if debug_mode:
        if not hasattr(support_test, 'debug_call_count'):
            support_test.debug_call_count = 0
        
        support_test.debug_call_count += 1
        
        if support_test.debug_call_count <= max_debug_calls:
            try:
                print(f"\n[SUPPORT DEBUG #{support_test.debug_call_count}]")
                print(f"  Closing direction: {closing_dir}")
                print(f"  Grasp center: {t}")
                print(f"  Tip1 center: {tip1}")
                print(f"  Tip2 center: {tip2}")
                print(f"  Support cylinder radius: {r_support:.4f}m")
                print(f"  Support cylinder half-height: {h_support:.4f}m")
                print(f"  Required points per tip: {κ}")
                print(f"  Tip1 support points: {cnt1}/{len(X)}")
                print(f"  Tip2 support points: {cnt2}/{len(X)}")
                print(f"  Support test result: {'PASS' if support_result else 'FAIL'}")
                
                # Visualization
                _visualize_support_test(
                    R, t, S, G, kdtree, 
                    closing_dir, tip1, tip2, r_support, h_support,
                    mask1, mask2, cnt1, cnt2, κ,
                    f"Support Test #{support_test.debug_call_count}"
                )
                
            except Exception as viz_error:
                print(f"    [ERROR] Support visualization failed: {viz_error}")

    return support_result

def _visualize_support_test(R_world, t_world, S, G, kdtree, 
                           closing_dir, tip1, tip2, r_support, h_support,
                           mask1, mask2, cnt1, cnt2, required_points,
                           window_name="Support Test"):
    """
    Visualization showing support test logic with color coding
    """   
    try:
        geometries = []
        X = kdtree.data
        
        # 1. Background points (not supporting either tip)
        background_mask = ~(mask1 | mask2)
        if np.any(background_mask):
            pcd_background = o3d.geometry.PointCloud()
            pcd_background.points = o3d.utility.Vector3dVector(X[background_mask])
            pcd_background.paint_uniform_color([0.7, 0.7, 0.7])  # Gray background
            geometries.append(pcd_background)

        # 2. Support points for tip1
        if np.any(mask1):
            tip1_support_points = X[mask1]
            pcd_tip1 = o3d.geometry.PointCloud()
            pcd_tip1.points = o3d.utility.Vector3dVector(tip1_support_points)
            # Color based on whether tip1 has enough support
            if cnt1 >= required_points:
                pcd_tip1.paint_uniform_color([0.0, 1.0, 0.0])  # Green = sufficient support
            else:
                pcd_tip1.paint_uniform_color([1.0, 0.5, 0.0])  # Orange = insufficient support
            geometries.append(pcd_tip1)

        # 3. Support points for tip2
        if np.any(mask2):
            tip2_support_points = X[mask2]
            pcd_tip2 = o3d.geometry.PointCloud()
            pcd_tip2.points = o3d.utility.Vector3dVector(tip2_support_points)
            # Color based on whether tip2 has enough support
            if cnt2 >= required_points:
                pcd_tip2.paint_uniform_color([0.0, 0.8, 0.0])  # Slightly different green
            else:
                pcd_tip2.paint_uniform_color([1.0, 0.3, 0.0])  # Slightly different orange
            geometries.append(pcd_tip2)

        # 4. Support points for BOTH tips (overlap)
        overlap_mask = mask1 & mask2
        if np.any(overlap_mask):
            overlap_points = X[overlap_mask]
            pcd_overlap = o3d.geometry.PointCloud()
            pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
            pcd_overlap.paint_uniform_color([0.0, 0.0, 1.0])  # Blue = supporting both tips
            geometries.append(pcd_overlap)

        # 5. Support cylinder visualization for tip1
        cylinder1 = o3d.geometry.TriangleMesh.create_cylinder(
            radius=r_support, 
            height=2 * h_support,
            resolution=20
        )
        
        # Orient and position cylinder1
        z_axis = np.array([0, 0, 1])
        if not np.allclose(closing_dir, z_axis):
            if np.allclose(closing_dir, -z_axis):
                cyl_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                v = np.cross(z_axis, closing_dir)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, closing_dir)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                cyl_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            cyl_rotation = np.eye(3)

        cylinder1_transform = np.eye(4)
        cylinder1_transform[:3, :3] = cyl_rotation
        cylinder1_transform[:3, 3] = tip1
        cylinder1.transform(cylinder1_transform)

        # Create wireframe version for tip1
        cylinder1_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder1)
        if cnt1 >= required_points:
            cylinder1_wireframe.paint_uniform_color([0.0, 0.8, 0.0])  # Green wireframe
        else:
            cylinder1_wireframe.paint_uniform_color([1.0, 0.0, 0.0])  # Red wireframe
        geometries.append(cylinder1_wireframe)

        # 6. Support cylinder visualization for tip2
        cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(
            radius=r_support, 
            height=2 * h_support,
            resolution=20
        )
        
        cylinder2_transform = np.eye(4)
        cylinder2_transform[:3, :3] = cyl_rotation
        cylinder2_transform[:3, 3] = tip2
        cylinder2.transform(cylinder2_transform)

        # Create wireframe version for tip2
        cylinder2_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder2)
        if cnt2 >= required_points:
            cylinder2_wireframe.paint_uniform_color([0.0, 0.6, 0.0])  # Slightly different green
        else:
            cylinder2_wireframe.paint_uniform_color([0.8, 0.0, 0.0])  # Slightly different red
        geometries.append(cylinder2_wireframe)

        # 7. Grasp center as sphere
        grasp_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        grasp_center_sphere.translate(t_world)
        grasp_center_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
        geometries.append(grasp_center_sphere)

        # 8. Closing direction arrow
        # Create arrow from grasp center along closing direction
        arrow_length = h_support * 2.5
        arrow_end = t_world + closing_dir * arrow_length
        
        # Create line for arrow shaft
        line_points = [t_world, arrow_end]
        line_lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_lines)
        line_set.paint_uniform_color([1.0, 1.0, 1.0])  # White
        geometries.append(line_set)

        # 9. Gripper geometry (optional)
        try:
            gripper_meshes = G.make_open3d_meshes(colour=(0.2, 0.8, 0.2))
            gripper_transform = np.eye(4)
            gripper_transform[:3, :3] = R_world
            gripper_transform[:3, 3] = t_world
            
            for mesh in gripper_meshes:
                mesh.transform(gripper_transform)
                geometries.append(mesh)
                
        except Exception as gripper_error:
            print(f"    [ERROR] Could not add gripper visualization: {gripper_error}")
        
        # 10. Main coordinate frame
        main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        geometries.append(main_coord_frame)
        
        # Print legend
        print(f"\n{'='*60}")
        print(f"SUPPORT TEST VISUALIZATION:")
        print(f"{'='*60}")
        print(f"  🟫 Gray:    Background points (not supporting)")
        tip1_status = "✅ PASS" if cnt1 >= required_points else "❌ FAIL"
        tip2_status = "✅ PASS" if cnt2 >= required_points else "❌ FAIL"
        print(f"  🟢 Green:   Tip1 support points ({cnt1}/{required_points}) {tip1_status}")
        print(f"  🟢 Lt.Green: Tip2 support points ({cnt2}/{required_points}) {tip2_status}")
        print(f"  🟣 Magenta: Grasp center")
        print(f"  ⚪ White:   Closing direction")
        print(f"  🤖 Robot:   Gripper geometry")
        print(f"  🟢 Green cylinders: Sufficient support")
        print(f"  🔴 Red cylinders:   Insufficient support")
        print(f"{'='*60}")
        print(f"  Support cylinder radius: {r_support:.3f}m")
        print(f"  Support cylinder height: {2*h_support:.3f}m")
        print(f"  Required points per tip: {required_points}")
        overall_result = "✅ PASS" if (cnt1 >= required_points and cnt2 >= required_points) else "❌ FAIL"
        print(f"  OVERALL SUPPORT TEST: {overall_result}")
        
        # Show visualization
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            zoom=0.6,
            front=[0, -1, 0],
            lookat=t_world,
            up=[0, 0, 1]
        )
        
    except Exception as e:
        print(f"Error in support test visualization: {e}")
        import traceback
        traceback.print_exc()

def collision_test(R_world, t_world, S, G, kdtree: KDTree,
                   debug_mode=False, max_debug_calls=5) -> bool:
    """
    CORRECTED IMPLEMENTATION with two-slab logic:
    
    1. Small slab (cylinder region): for safe grasping area
    2. Large slab (includes fingers): for collision detection
    
    Returns:
        True  → NO collision (pose is valid)
        False → collision detected (reject this pose)
    """

    # --- 1. closing line in world coordinates -----------------------
    λ_dir = R_world @ G.lambda_local
    λ_dir /= np.linalg.norm(λ_dir)

    # --- 2. pick the SQ semi-axis most aligned with λ --------------
    λ_local = S.R.T @ λ_dir
    axis_idx = np.argmax(np.abs(λ_local))
    a_axis = [S.ax, S.ay, S.az][axis_idx]

    # --- 3. TWO DIFFERENT SLAB DEFINITIONS -------------------------
    half_open = G.max_open / 2.0
    
    # SMALL SLAB: Just the cylinder region (for safe grasping)
    half_height_cylinder = half_open
    radius = G.jaw_len
    
    # LARGE SLAB: Includes finger extent (for collision detection)
    half_height_finger = half_open + 2 * G.thickness  # Add finger length to slab
    
    # --- 4. GET POINTS FOR BOTH SLABS ------------------------------
    X = kdtree.data
    rel = X - t_world
    proj = rel @ λ_dir
    
    # Small slab mask (cylinder region only)
    small_slab_mask = np.abs(proj) <= half_height_cylinder
    
    # Large slab mask (includes finger extent)
    large_slab_mask = np.abs(proj) <= half_height_finger
    
    if not np.any(large_slab_mask):
        collision_result = True  # No points in either slab → no collision
        if debug_mode:
            print(f"    [COLLISION] NO POINTS IN LARGE SLAB - Safe grasp")
        return collision_result

    # --- 5. CYLINDER LOGIC: Use SMALL slab for safe grasping ------
    if np.any(small_slab_mask):
        # Points inside small slab
        small_slab_points = X[small_slab_mask]
        rel_small_slab = rel[small_slab_mask]
        proj_small_slab = proj[small_slab_mask]
        
        # Calculate radial distance from closing line for points in small slab
        radial_vec_small = rel_small_slab - np.outer(proj_small_slab, λ_dir)
        radial_len_small = np.linalg.norm(radial_vec_small, axis=1)
        
        # Points OUTSIDE cylinder in small slab = potential collision points
        outside_cylinder_mask_small = radial_len_small > radius
        potential_collision_points_from_cylinder = small_slab_points[outside_cylinder_mask_small]
    else:
        potential_collision_points_from_cylinder = np.array([]).reshape(0, 3)
        outside_cylinder_mask_small = np.array([], dtype=bool)

    # --- 6. FINGER COLLISION: Use LARGE slab ----------------------
    # All points in large slab that are NOT in small slab = finger collision candidates
    large_slab_points = X[large_slab_mask]
    
    # Find points that are in large slab but NOT in small slab
    # These are the points in the finger extension region
    finger_region_mask = large_slab_mask.copy()
    finger_region_mask[small_slab_mask] = False  # Remove small slab points
    finger_region_points = X[finger_region_mask]

    # Combine all potential collision points
    if len(potential_collision_points_from_cylinder) > 0 and len(finger_region_points) > 0:
        all_potential_collision_points = np.vstack([
            potential_collision_points_from_cylinder,
            finger_region_points
        ])
    elif len(potential_collision_points_from_cylinder) > 0:
        all_potential_collision_points = potential_collision_points_from_cylinder
    elif len(finger_region_points) > 0:
        all_potential_collision_points = finger_region_points
    else:
        collision_result = True  # No potential collision points → safe
        if debug_mode:
            print(f"    [COLLISION] No potential collision points - Safe grasp")
        return collision_result

    # --- 7. CHECK GRIPPER BODY COLLISIONS -------------------------
    # Check collision with gripper palm/back
    palm_collisions = check_palm_collision(
        all_potential_collision_points, t_world, R_world, G
    )
    
    # Check collision with gripper fingers
    finger_collisions = check_finger_collision(
        all_potential_collision_points, t_world, R_world, G
    )
    
    # Combine both collision types
    has_palm_collision = np.any(palm_collisions) if len(palm_collisions) > 0 else False
    has_finger_collision = np.any(finger_collisions) if len(finger_collisions) > 0 else False
    has_collision = has_palm_collision or has_finger_collision
    
    collision_result = not has_collision  # True = no collision

    # --- 8. DEBUG OUTPUT -------------------------------------------
    if debug_mode:
        if not hasattr(collision_test, 'debug_call_count'):
            collision_test.debug_call_count = 0
        
        collision_test.debug_call_count += 1
        
        if collision_test.debug_call_count <= max_debug_calls:
            try:
                small_slab_count = np.sum(small_slab_mask)
                large_slab_count = np.sum(large_slab_mask)
                finger_region_count = len(finger_region_points)
                cylinder_collision_count = len(potential_collision_points_from_cylinder)
                
                print(f"\n[COLLISION DEBUG #{collision_test.debug_call_count}] - TWO-SLAB LOGIC")
                print(f"  Small slab (cylinder): half_height={half_height_cylinder:.4f}m, radius={radius:.4f}m")
                print(f"  Large slab (w/fingers): half_height={half_height_finger:.4f}m")
                print(f"  Closing direction λ: {λ_dir}")
                print(f"  Grasp center: {t_world}")
                print(f"  Points in small slab: {small_slab_count}/{len(X)}")
                print(f"  Points in large slab: {large_slab_count}/{len(X)}")
                print(f"  Points in finger region: {finger_region_count}")
                print(f"  Cylinder collision candidates: {cylinder_collision_count}")
                print(f"  Total collision candidates: {len(all_potential_collision_points)}")
                print(f"  Palm collisions: {np.sum(palm_collisions) if len(palm_collisions) > 0 else 0}")
                print(f"  Finger collisions: {np.sum(finger_collisions) if len(finger_collisions) > 0 else 0}")
                print(f"  Total collisions: {has_collision}")
                print(f"  Result: {'NO COLLISION' if collision_result else 'COLLISION DETECTED'}")
                
                # Visualization with two-slab logic
                _visualize_collision_two_slab(
                    R_world, t_world, S, G, kdtree, 
                    λ_dir, radius, half_height_cylinder, half_height_finger,
                    small_slab_mask, large_slab_mask, outside_cylinder_mask_small,
                    palm_collisions, finger_collisions,
                    f"Two-Slab Collision Test #{collision_test.debug_call_count}"
                )
                
            except Exception as viz_error:
                print(f"    [ERROR] Visualization failed: {viz_error}")

    return collision_result

def check_finger_collision(points, gripper_center, R_world, gripper):
    """Check collision with gripper fingers modeled as cylinders in world frame"""
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # Work entirely in world coordinates
    rel_points = points - gripper_center
    
    # Get gripper axes in world frame
    approach_world = R_world @ gripper.approach_axis  # -Z in local becomes approach direction in world
    closing_world = R_world @ gripper.lambda_local    # +Y in local becomes closing direction in world
    width_world = R_world @ np.array([1, 0, 0])       # +X in local becomes width direction in world
    
    # Project points onto gripper axes
    approach_proj = rel_points @ approach_world
    closing_proj = rel_points @ closing_world
    width_proj = rel_points @ width_world
    
    # Finger cylinder parameters
    finger_radius = gripper.thickness
    finger_half_width = gripper.max_open / 2.0
    
    # CORRECTED: Fingers extend TOWARD the object (in POSITIVE approach direction)
    # Since approach_axis = [0, 0, -1] and gripper moves toward object in -Z direction,
    # the approach_world vector points toward the object
    # Fingers extend from gripper_center in the approach_world direction
    in_finger_length = (approach_proj >= 0) & (approach_proj <= gripper.jaw_len)
    
    # Calculate distance from each finger's center line
    # Left finger center line: closing_proj = +finger_half_width, any width_proj, approach_proj ∈ [0, +jaw_len]
    left_finger_closing_dist = np.abs(closing_proj - finger_half_width)
    left_finger_width_dist = np.abs(width_proj)
    left_finger_radial_dist = np.sqrt(left_finger_closing_dist**2 + left_finger_width_dist**2)
    
    # Right finger center line: closing_proj = -finger_half_width, any width_proj, approach_proj ∈ [0, +jaw_len]  
    right_finger_closing_dist = np.abs(closing_proj + finger_half_width)
    right_finger_width_dist = np.abs(width_proj)
    right_finger_radial_dist = np.sqrt(right_finger_closing_dist**2 + right_finger_width_dist**2)
    
    # Check if points are within cylinder radius of either finger
    near_left_finger = left_finger_radial_dist <= finger_radius
    near_right_finger = right_finger_radial_dist <= finger_radius
    
    # Collision occurs if point is in finger length AND near either finger cylinder
    finger_collisions = in_finger_length & (near_left_finger | near_right_finger)
    
    return finger_collisions

def check_palm_collision(points, gripper_center, R_world, gripper):
    """Check collision in world frame directly"""
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # Work entirely in world coordinates
    rel_points = points - gripper_center
    
    # Get gripper axes in world frame
    approach_world = R_world @ gripper.approach_axis  # -Z in local becomes approach direction in world
    closing_world = R_world @ gripper.lambda_local    # +Y in local becomes closing direction in world
    width_world = R_world @ np.array([1, 0, 0])       # +X in local becomes width direction in world
    
    # Project points onto gripper axes
    approach_proj = rel_points @ approach_world
    closing_proj = rel_points @ closing_world
    width_proj = rel_points @ width_world
    
    # Palm is in the approach direction (where gripper comes from)
    # Gripper approaches from positive approach direction, so palm is at positive approach_proj
    in_approach_region = (approach_proj >= gripper.jaw_len) & (approach_proj <= (gripper.jaw_len + gripper.palm_depth))
    within_palm_width = np.abs(width_proj) <= gripper.palm_width / 2
    within_palm_height = np.abs(closing_proj) <= gripper.max_open / 2
    
    collisions = in_approach_region & within_palm_width & within_palm_height
    return collisions

def _visualize_collision_two_slab(R_world, t_world, S, G, kdtree, 
                                 lambda_dir, radius, half_height_cylinder, half_height_finger,
                                 small_slab_mask, large_slab_mask, outside_cylinder_mask_small,
                                 palm_collisions, finger_collisions,
                                 window_name="Two-Slab Collision Test"):
    """
    Visualization showing two-slab logic with proper color coding
    """   
    try:
        geometries = []
        X = kdtree.data
        
        # 1. Background points (not in any slab)
        background_mask = ~large_slab_mask
        if np.any(background_mask):
            pcd_background = o3d.geometry.PointCloud()
            pcd_background.points = o3d.utility.Vector3dVector(X[background_mask])
            pcd_background.paint_uniform_color([0.7, 0.7, 0.7])  # Gray background
            geometries.append(pcd_background)

        # 2. Small slab points (cylinder region)
        if np.any(small_slab_mask):
            small_slab_points = X[small_slab_mask]
            
            # Points INSIDE cylinder = GREEN (safe for grasping)
            inside_cylinder_points = small_slab_points[~outside_cylinder_mask_small]
            if len(inside_cylinder_points) > 0:
                pcd_safe = o3d.geometry.PointCloud()
                pcd_safe.points = o3d.utility.Vector3dVector(inside_cylinder_points)
                pcd_safe.paint_uniform_color([0.0, 1.0, 0.0])  # Green = safe
                geometries.append(pcd_safe)
            
            # Points OUTSIDE cylinder in small slab = POTENTIAL COLLISION (orange)
            if np.any(outside_cylinder_mask_small):
                potential_collision_points = small_slab_points[outside_cylinder_mask_small]
                pcd_potential = o3d.geometry.PointCloud()
                pcd_potential.points = o3d.utility.Vector3dVector(potential_collision_points)
                pcd_potential.paint_uniform_color([1.0, 0.5, 0.0])  # Orange = potential collision
                geometries.append(pcd_potential)

        # 3. Finger region points (large slab minus small slab)
        finger_region_mask = large_slab_mask.copy()
        finger_region_mask[small_slab_mask] = False
        if np.any(finger_region_mask):
            finger_region_points = X[finger_region_mask]
            pcd_finger_region = o3d.geometry.PointCloud()
            pcd_finger_region.points = o3d.utility.Vector3dVector(finger_region_points)
            pcd_finger_region.paint_uniform_color([0.0, 0.0, 1.0])  # Blue = finger region
            geometries.append(pcd_finger_region)

        # 4. Collision points (from both regions)
        all_potential_points = []
        if np.any(small_slab_mask) and np.any(outside_cylinder_mask_small):
            all_potential_points.extend(X[small_slab_mask][outside_cylinder_mask_small])
        if np.any(finger_region_mask):
            all_potential_points.extend(X[finger_region_mask])
        
        if len(all_potential_points) > 0:
            all_potential_points = np.array(all_potential_points)
            
            # Separate collision types for visualization
            palm_collision_mask = palm_collisions if len(palm_collisions) > 0 else np.zeros(len(all_potential_points), dtype=bool)
            finger_collision_mask = finger_collisions if len(finger_collisions) > 0 else np.zeros(len(all_potential_points), dtype=bool)
            
            # PALM COLLISION POINTS (red spheres)
            if np.any(palm_collision_mask):
                palm_collision_points = all_potential_points[palm_collision_mask]
                for point in palm_collision_points:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                    sphere.translate(point)
                    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                    geometries.append(sphere)
            
            # FINGER COLLISION POINTS (magenta spheres)
            if np.any(finger_collision_mask):
                finger_collision_points = all_potential_points[finger_collision_mask]
                for point in finger_collision_points:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                    sphere.translate(point)
                    sphere.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
                    geometries.append(sphere)

        # 5. Cylinder visualization (small slab)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, 
            height=2 * half_height_cylinder,
            resolution=20
        )
        
        # Orient and position cylinder
        z_axis = np.array([0, 0, 1])
        if not np.allclose(lambda_dir, z_axis):
            if np.allclose(lambda_dir, -z_axis):
                cyl_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                v = np.cross(z_axis, lambda_dir)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, lambda_dir)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                cyl_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            cyl_rotation = np.eye(3)

        cylinder_transform = np.eye(4)
        cylinder_transform[:3, :3] = cyl_rotation
        cylinder_transform[:3, 3] = t_world
        cylinder.transform(cylinder_transform)

        # Create wireframe version
        cylinder_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
        cylinder_wireframe.paint_uniform_color([0.2, 0.2, 0.8])  # Blue wireframe
        geometries.append(cylinder_wireframe)

        # 6. Large slab boundaries (optional - as planes)
        # You could add plane visualizations here to show the large slab extent

        # 7. Gripper geometry
        try:           
            gripper_transform = np.eye(4)
            gripper_transform[:3, :3] = R_world
            gripper_transform[:3, 3] = t_world
            
            gripper_meshes = get_gripper_control_points_o3d(
                gripper_transform,
                gripper=G,
                show_sweep_volume=False,
                color=(0.2, 0.8, 0.2),
                finger_tip_to_origin=True
            )
            geometries.extend(gripper_meshes)
            
        except ImportError:
            print(f"    [ERROR] Could not import gripper visualization")
        
        # 8. Legend
        print(f"\n{'='*60}")
        print(f"TWO-SLAB COLLISION TEST VISUALIZATION:")
        print(f"{'='*60}")
        print(f"  🟫 Gray:    Background points (outside large slab)")
        print(f"  🟢 Green:   Safe grasping points (inside cylinder)")
        print(f"  🟠 Orange:  Potential collision (outside cylinder in small slab)")
        print(f"  🟨 Yellow:  Finger region points (large slab - small slab)")
        print(f"  🔴 Red:     PALM collision points")
        print(f"  🟣 Magenta: FINGER collision points")
        print(f"  🔵 Blue:    Collision cylinder (small slab)")
        print(f"  🤖 Robot:   Gripper geometry")
        print(f"{'='*60}")
        print(f"  Small slab height: {2*half_height_cylinder:.3f}m")
        print(f"  Large slab height: {2*half_height_finger:.3f}m")
        print(f"  Cylinder radius: {radius:.3f}m")
        
        # Show visualization
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            zoom=0.6,
            front=[0, -1, 0],
            lookat=t_world,
            up=[0, 0, 1]
        )
        
    except Exception as e:
        print(f"Error in two-slab collision visualization: {e}")
        import traceback
        traceback.print_exc()

# ------------------------------------------------------------
# 5. Scoring
# ------------------------------------------------------------


def score_grasp(R, t, S, G, kdtree, surface_tol=0.005):
    """
      FIXED: Corrected superquadric implicit function
    """
    X = kdtree.data
    N = X.shape[0]
    
    # Transform to SQ-local coordinates
    rel = X - S.T
    pts_local = rel @ S.R.T
    
    surface_tol = 0.05 * min(S.ax, S.ay, S.az)
    
    #   DEBUG: Add extensive debugging
    # print(f"    [DEBUG] SQ params: ε1={S.ε1:.3f}, ε2={S.ε2:.3f}")
    # print(f"    [DEBUG] SQ scale: ax={S.ax:.3f}, ay={S.ay:.3f}, az={S.az:.3f}")
    # print(f"    [DEBUG] SQ center: {S.T}")
    # print(f"    [DEBUG] Points range: x=[{pts_local[:, 0].min():.3f}, {pts_local[:, 0].max():.3f}]")
    # print(f"    [DEBUG] Points range: y=[{pts_local[:, 1].min():.3f}, {pts_local[:, 1].max():.3f}]")
    # print(f"    [DEBUG] Points range: z=[{pts_local[:, 2].min():.3f}, {pts_local[:, 2].max():.3f}]")
    
    #   CRITICAL FIX: Correct superquadric implicit function
    # Standard form: ((|x/a1|^(2/ε2) + |y/a2|^(2/ε2))^(ε2/ε1) + |z/a3|^(2/ε1))^(1/1) = 1
    
    # Avoid division by zero
    safe_ax = max(S.ax, 1e-6)
    safe_ay = max(S.ay, 1e-6) 
    safe_az = max(S.az, 1e-6)
    safe_eps1 = max(S.ε1, 0.1)
    safe_eps2 = max(S.ε2, 0.1)
    
    # Normalized coordinates
    x_norm = np.abs(pts_local[:, 0]) / safe_ax
    y_norm = np.abs(pts_local[:, 1]) / safe_ay
    z_norm = np.abs(pts_local[:, 2]) / safe_az
    
    #   CORRECT EQUATION: F(x,y,z) = ((x/a)^(2/ε2) + (y/b)^(2/ε2))^(ε2/ε1) + (z/c)^(2/ε1)
    try:
        # Handle potential numerical issues with small exponents
        eps_ratio_xy = 2.0 / safe_eps2
        eps_ratio_z = 2.0 / safe_eps1
        eps_power = safe_eps2 / safe_eps1
        
        # Compute terms with clamping to avoid overflow
        x_term = np.power(np.clip(x_norm, 1e-10, 1e10), eps_ratio_xy)
        y_term = np.power(np.clip(y_norm, 1e-10, 1e10), eps_ratio_xy)
        z_term = np.power(np.clip(z_norm, 1e-10, 1e10), eps_ratio_z)
        
        # Combine xy terms
        xy_combined = np.power(np.clip(x_term + y_term, 1e-10, 1e10), eps_power)
        
        # Final implicit function value
        implicit_values = xy_combined + z_term
        
        # Distance from surface (F = 1)
        surface_distances = np.abs(implicit_values - 1.0)

        # print(f"    [DEBUG] Implicit values range: [{implicit_values.min():.6f}, {implicit_values.max():.6f}]")
        # print(f"    [DEBUG] Surface distances range: [{surface_distances.min():.6f}, {surface_distances.max():.6f}]")

    except Exception as eq_error:
        print(f"    [ERROR] Equation computation failed: {eq_error}")
        # Fallback: assume no surface points
        surface_distances = np.full(len(pts_local), 1000.0)
    
    #   ADAPTIVE SURFACE TOLERANCE: Scale with object size
    char_size = (safe_ax * safe_ay * safe_az) ** (1/3)
    base_tolerance = 0.1  # Base tolerance for implicit function
    size_scaled_tolerance = base_tolerance * max(0.5, char_size / 0.05)  # Scale with object size
    
    # print(f"    [DEBUG] Char size: {char_size:.6f}, tolerance: {size_scaled_tolerance:.6f}")
    
    # Surface mask with adaptive tolerance
    surface_mask = surface_distances < size_scaled_tolerance
    Y = X[surface_mask]
    
    # print(f"    [DEBUG] Surface points found: {len(Y)}/{N} with tolerance {size_scaled_tolerance:.6f}")
    
    if len(Y) == 0:
        #   FALLBACK: If no surface points, use proximity-based approach
        # print(f"    [FALLBACK] No surface points found, using proximity-based approach")
        
        # Find points within reasonable distance of SQ center
        distances_from_center = np.linalg.norm(rel, axis=1)
        max_reasonable_distance = np.max([safe_ax, safe_ay, safe_az]) * 2.0
        
        proximity_mask = distances_from_center < max_reasonable_distance
        Y_fallback = X[proximity_mask]
        
        if len(Y_fallback) > 10:
            Y = Y_fallback[:min(100, len(Y_fallback))]  # Limit to avoid computation issues
            # print(f"    [FALLBACK] Using {len(Y)} proximity points instead")
        else:
            # print(f"    [FALLBACK] Still no good points, using h_α=h_β=0")
            h_α = 0.0
            h_β = 0.0
            α = float('inf')
            β = 0.0
    
    if len(Y) > 0:
        # h_α: Point-to-surface distance (using actual superquadric surface)
        try:
            S_surface = sample_superquadric_surface(S, n_samples=500)  # Reduced samples for performance
            if len(S_surface) > 0:
                tree_surface = KDTree(S_surface)
                distances_Y_to_S, _ = tree_surface.query(Y)
                α = np.mean(distances_Y_to_S)
                
                # print(f"    [DEBUG] α (mean distance to surface): {α:.6f}")
                
                # Scale q_α with object size
                q_α = 0.001 * (char_size / 0.05)  # Scale with object size
                h_α = np.exp(- (α**2) / q_α)
            else:
                α = float('inf')
                h_α = 0.0
        except Exception as alpha_error:
            print(f"    [ERROR] h_α computation failed: {alpha_error}")
            α = float('inf')
            h_α = 0.0

        # h_β: Coverage
        try:
            if len(S_surface) > 0:
                tree_Y = KDTree(Y)
                d_th = char_size * 0.2  # Scale coverage threshold with object size
                distances_S_to_Y, _ = tree_Y.query(S_surface)
                T_mask = distances_S_to_Y <= d_th
                T_count = np.sum(T_mask)
                β = T_count / len(S_surface)
                h_β = β**2
                
                # print(f"    [DEBUG] β (coverage): {β:.6f}, d_th: {d_th:.6f}")
            else:
                β = 0.0
                h_β = 0.0
        except Exception as beta_error:
            print(f"    [ERROR] h_β computation failed: {beta_error}")
            β = 0.0
            h_β = 0.0
    
    # Rest of the function remains the same...
    #   h_γ and h_δ calculations unchanged
    closing_dir = R @ G.lambda_local
    half_open = G.max_open / 2.0
    
    tip1_world = t + closing_dir * half_open
    tip2_world = t - closing_dir * half_open
    
    curv1 = gaussian_curvature_at_point(tip1_world, S)
    curv2 = gaussian_curvature_at_point(tip2_world, S)
    γ = (curv1 + curv2) / 2.0
    
    q_γ = 0.5
    h_γ = np.exp(- (γ**2) / q_γ)

    centroid = np.mean(X, axis=0)
    δ = np.linalg.norm(t - centroid)
    q_δ = 0.005
    h_δ = np.exp(- (δ**2) / q_δ)

    final_score = h_α * h_β * h_γ * h_δ

    # Enhanced debug output
    # print(f"    Detailed scoring breakdown:")
    # print(f"      Surface points: {len(Y)}/{N}")
    # print(f"      α={α:.8f} → h_α={h_α:.8f}")
    # print(f"      β={β:.8f} → h_β={h_β:.8f}")
    # print(f"      γ={γ:.8f} → h_γ={h_γ:.8f}")
    # print(f"      δ={δ:.8f} → h_δ={h_δ:.8f}")
    print(f"      FINAL SCORE: {final_score:.12f}")

    return final_score

def gaussian_curvature_at_point(point_world, S: Superquadric):
    """
    Compute Gaussian curvature of superquadric surface at a given world point.
      FIXED: Scale-aware curvature for small objects
    """
    # Transform point to SQ-local coordinates
    point_local = S.R.T @ (point_world - S.T)
    x, y, z = point_local
    
    eps1, eps2 = S.ε1, S.ε2
    ax, ay, az = S.ax, S.ay, S.az
    
    #   DEBUG: Show what's causing the high scale factor
    # print(f"      Scale params: ax={ax:.6f}, ay={ay:.6f}, az={az:.6f}")
    # print(f"      Shape params: ε1={eps1:.6f}, ε2={eps2:.6f}")
    
    #   CRITICAL: Scale-aware curvature calculation for small objects
    
    # Calculate characteristic size (geometric mean of scales)
    char_size = (ax * ay * az) ** (1/3)  # Geometric mean
    
    #   NEW: Normalize curvature by object size
    # For the paper's method, typical objects are 10-20cm
    # Your objects are 2-4cm, so we need to scale accordingly
    
    reference_size = 0.10  # 10cm reference size (typical for paper)
    size_ratio = char_size / reference_size
    
    # Base curvature from shape parameters (conservative)
    if eps1 < 0.5 or eps2 < 0.5:
        shape_factor = 2.0  # Box-like, higher curvature at edges
    elif eps1 > 1.5 and eps2 > 1.5:
        shape_factor = 0.3  # Sphere-like, lower curvature
    else:
        shape_factor = 1.0  # Ellipsoid-like, moderate
    
    #   FIXED: Size-normalized scale factor
    # Smaller objects get LOWER curvature penalty (counter-intuitive but needed for small objects)
    size_normalized_factor = 1.0 / (size_ratio + 0.5)  # Add 0.5 to prevent division issues
    
    # Position factor: very conservative for small objects
    r_local = np.sqrt((x/ax)**2 + (y/ay)**2 + (z/az)**2)
    position_factor = 1.0 + 0.2 * r_local  # Much more conservative
    
    #   FINAL: Combine all factors with small object compensation
    base_curvature = shape_factor * size_normalized_factor * position_factor
    
    #   CRITICAL: For very small objects, clamp to much lower range
    if char_size < 0.05:  # Objects smaller than 5cm
        max_curvature = 1.5  # Much lower max for small objects
    elif char_size < 0.10:  # Objects smaller than 10cm
        max_curvature = 2.5
    else:
        max_curvature = 5.0  # Original max for larger objects
    
    gaussian_curv = np.clip(base_curvature, 0.1, max_curvature)
    
    #   DEBUG: Show the scale-aware calculation
    # print(f"      char_size={char_size:.6f}m, size_ratio={size_ratio:.3f}")
    # print(f"      shape_factor={shape_factor:.3f}")
    # print(f"      size_normalized_factor={size_normalized_factor:.6f}")
    # print(f"      position_factor={position_factor:.3f}")
    # print(f"      max_curvature={max_curvature:.1f} (size-dependent)")
    # print(f"      final_curv={gaussian_curv:.6f} (SCALE-AWARE)")
    
    return gaussian_curv

def sample_superquadric_surface(S: Superquadric, n_samples=1000):
    """
    Sample points densely on superquadric surface for α computation.
    """
    # Create parameter grid
    u = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(n_samples)))
    v = np.linspace(-np.pi, np.pi, int(np.sqrt(n_samples)))
    U, V = np.meshgrid(u, v)
    U, V = U.flatten(), V.flatten()
    
    # Parametric surface equations
    eps1, eps2 = S.ε1, S.ε2
    x = S.ax * np.sign(np.cos(U)) * (np.abs(np.cos(U))**eps1) * np.sign(np.cos(V)) * (np.abs(np.cos(V))**eps2)
    y = S.ay * np.sign(np.cos(U)) * (np.abs(np.cos(U))**eps1) * np.sign(np.sin(V)) * (np.abs(np.sin(V))**eps2)
    z = S.az * np.sign(np.sin(U)) * (np.abs(np.sin(U))**eps1)
    
    # Transform to world coordinates
    points_local = np.column_stack([x, y, z])
    points_world = (S.R @ points_local.T).T + S.T
    
    return points_world

# ------------------------------------------------------------
# 6. SuperquadricGraspPlanner class for ROS2 integration
# ------------------------------------------------------------

class SuperquadricGraspPlanner:
    def __init__(self, jaw_len=0.054, max_open=0.08):
        """
        Initialize the superquadric-based grasp planner
        
        Args:
            jaw_len: gripper jaw length in meters
            max_open: maximum gripper opening in meters
        """
        self.gripper = Gripper(jaw_len=jaw_len, max_open=max_open)
        # Store last valid grasps for visualization
        self.last_valid_grasps = []
        
    def get_all_grasps(self, point_cloud_path, shape, scale, euler, translation):
        """
        Get ALL grasps (raw candidates) without filtering
        Returns:
            List of dictionaries with 'pose' and 'score' keys for consistency
        """
        try:
            # Create superquadric object
            S = Superquadric(ε=shape, a=scale, euler=euler, t=translation)
            G = self.gripper
            
            # Load point cloud & build KD-tree
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            if not pcd.has_points():
                return []
            X = np.asarray(pcd.points)
            kdtree = KDTree(X)
            
            # Generate raw candidates
            base_R = principal_axis_sweeps(S, G, step_deg=10)
            all_R, extra_offsets = extra_sweeps_special_shapes(S, base_R, G)

            # Build G_raw
            G_raw = []
            for Rg, Δt in extra_offsets:
                G_raw.append(make_world_pose(S, Rg, Δt))
            seen_rots = set()
            for Rg, _ in extra_offsets:
                key = tuple(np.round(Rg, 6).ravel())
                seen_rots.add(key)
            for Rg in all_R:
                key = tuple(np.round(Rg, 6).ravel())
                if key not in seen_rots:
                    G_raw.append(make_world_pose(S, Rg, np.zeros(3)))
                    seen_rots.add(key)

            print(f"[INFO] Found {len(G_raw)} raw grasp candidates")
            
            # Convert to consistent dictionary format
            grasp_data = []
            for Rg, tg in G_raw:
                T = np.eye(4)
                T[:3, :3] = Rg
                T[:3, 3] = tg
                
                grasp_data.append({
                    'pose': T,
                    'score': 0.0,  # Raw candidates get default score
                    'rotation': Rg,
                    'translation': tg
                })
            
            # Store for future reference
            self.last_valid_grasps = [data['pose'] for data in grasp_data]
            
            return grasp_data  # Now returns list of dictionaries
            
        except Exception as e:
            print(f"Error getting all grasps: {e}")
            return []

    def plan_grasps(self, point_cloud_path, shape, scale, euler, translation, max_grasps=5):
        """
        Plan grasps for an object using superquadric fitting (returns poses with scores)
        
        Args:
            point_cloud_path: path to object point cloud file
            shape: [ε1, ε2] shape parameters
            scale: [ax, ay, az] scale parameters  
            euler: [roll, pitch, yaw] orientation parameters
            translation: [tx, ty, tz] position parameters
            max_grasps: maximum number of grasps to return
            
        Returns:
            List of dictionaries with 'pose', 'score', 'rotation', 'translation' keys
        """
        try:
            # Create superquadric object using exact parameter order from generate_grasps.py
            S = Superquadric(ε=shape, a=scale, euler=euler, t=translation)
            G = self.gripper
            
            # 6.1 Load point cloud & build KD-tree
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            if not pcd.has_points():
                print(f"Point cloud '{point_cloud_path}' is empty or invalid.")
                return []
            X = np.asarray(pcd.points)
            kdtree = KDTree(X)
            
            print(f"[INFO] Loaded {len(X)} points from point cloud")

            # 6.2 Generate raw candidates (Steps 2.1 & 2.2)
            base_R = principal_axis_sweeps(S, G, step_deg=10)
            all_R, extra_offsets = extra_sweeps_special_shapes(S, base_R, G)

            # Build G_raw: include every (R, Δt) from extra_offsets, and base_R with Δt=0
            G_raw = []
            # Poses from extra_offsets
            for Rg, Δt in extra_offsets:
                G_raw.append(make_world_pose(S, Rg, Δt))
            # Poses from all_R not already in extra_offsets (zero translation)
            seen_rots = set()
            for Rg, _ in extra_offsets:
                key = tuple(np.round(Rg, 6).ravel())
                seen_rots.add(key)
            for Rg in all_R:
                key = tuple(np.round(Rg, 6).ravel())
                if key not in seen_rots:
                    G_raw.append(make_world_pose(S, Rg, np.zeros(3)))
                    seen_rots.add(key)


            # 6.3 Cull invalid candidates (Step 3) - SEPARATED FOR DEBUGGING
            G_after_support = []
            G_valid = []

            # First: Support test only
            for Rg, tg in G_raw:
                if support_test(Rg, tg, S, G, kdtree, debug_mode=True, max_debug_calls=3):  # Show first 3
                    G_after_support.append((Rg, tg))

            print(f"[INFO] {len(G_after_support)}/{len(G_raw)} grasps remain after support filtering.")

            # Second: Collision test on support-passing grasps
            for Rg, tg in G_after_support:
                if collision_test(Rg, tg, S, G, kdtree, debug_mode=False):
                    G_valid.append((Rg, tg))

            print(f"[INFO] {len(G_valid)}/{len(G_after_support)} grasps remain after collision filtering.")
            print(f"[INFO] TOTAL: {len(G_valid)}/{len(G_raw)} grasps remain after all filtering.")

            if len(G_valid) == 0:
                print("[WARNING] No valid grasps found after filtering.")
                return []

            # 6.4 Score remaining candidates (Step 4)
            scores = []
            for i, (Rg, tg) in enumerate(G_valid):
                score = score_grasp(Rg, tg, S, G, kdtree)
                scores.append(score)
                print(f"  Grasp {i+1}: score={score:.8f}")
            
            # Sort by score and take best ones
            scored_grasps = list(zip(scores, G_valid))
            scored_grasps.sort(key=lambda x: x[0], reverse=True)
            
            
            # Return both poses and their scores
            grasp_data = []
            for i, (score, (Rg, tg)) in enumerate(scored_grasps):
                T = np.eye(4)
                T[:3, :3] = Rg
                T[:3, 3] = tg
                
                grasp_data.append({
                    'pose': T,
                    'score': score,
                    'rotation': Rg,
                    'translation': tg
                })
                
                # Debug output to verify diversity
                approach_vector = -Rg[:, 2]  # Gripper approach direction
                print(f"Diverse Grasp {i+1}: score={score:.8f}, pos={tg}, approach_dir={approach_vector}")
            
            print(f"Returning {len(grasp_data)} diverse grasps with scores from {len(scored_grasps)} candidates")
            return grasp_data  # Return enhanced data instead of just poses

        except Exception as e:
            print(f"Error in grasp planning: {e}")
            import traceback
            traceback.print_exc()
            return []

        except Exception as e:
            print(f"Error in grasp planning: {e}")
            import traceback
            traceback.print_exc()
            return []

    def select_best_grasp_with_criteria(self, grasp_data_list, object_center=None):
        """
        Select the best grasp using multi-criteria optimization
        
        Args:
            grasp_data_list: List of dicts with 'pose', 'score', 'rotation', 'translation'
            object_center: Center of the object (optional, computed if None)
            
        Returns:
            Dict: Best grasp data with additional 'selection_info' key
        """
        if not grasp_data_list:
            return None
        
        if len(grasp_data_list) == 1:
            return grasp_data_list[0]
        
        # Compute object center if not provided
        if object_center is None:
            positions = [data['translation'] for data in grasp_data_list]
            object_center = np.mean(positions, axis=0)
        
        # Robot base origin (your robot starts at x=0.5, y=0.0, z=0.4)
        robot_origin = np.array([0.5, 0.0, 0.4])
        
        # Preferred approach direction (gripper approaches from +Z direction, not -Z)
        preferred_approach = np.array([0.0, 0.0, 1.0])  # Upward approach (from above)

        # Table height constraint
        table_height = 0.0  # Assuming table at z=0
        
        best_energy = float('inf')
        best_grasp = None
        
        print(f"[SELECTION] Robot base at: {robot_origin}")
        print(f"[SELECTION] Preferred approach direction: {preferred_approach}")
        print(f"[SELECTION] Object center: {object_center}")
        print(f"[SELECTION] Table height: {table_height}")
        
        # NEW: Pre-filter grasps to remove physically impossible ones
        valid_grasps = []
        for i, grasp_data in enumerate(grasp_data_list):
            pose = grasp_data['pose']
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            
            # Check 1: Gripper position must be above table
            if position[2] <= table_height + 0.01:  # 1cm safety margin above table
                print(f"  [FILTER] Rejected grasp {i+1}: position too low (z={position[2]:.3f}m)")
                continue
            
            # Check 2: Approach direction must not be from below table
            gripper_z_world = rotation_matrix[:, 2]
            actual_approach_direction = -gripper_z_world  # Gripper approaches in -Z direction
            
            # If approach direction has negative Z component, gripper is approaching from below
            if actual_approach_direction[2] < -0.5:  # Strong downward approach = approaching from below
                print(f"  [FILTER] Rejected grasp {i+1}: approaching from below table (approach_z={actual_approach_direction[2]:.3f})")
                continue
            
            # Check 3: Ensure gripper fingers won't go through table
            # Calculate finger tip positions
            closing_dir = rotation_matrix @ self.gripper.lambda_local
            half_open = self.gripper.max_open / 2.0
            tip1 = position + closing_dir * half_open
            tip2 = position - closing_dir * half_open
            
            # Both finger tips must be above table
            if tip1[2] <= table_height + 0.005 or tip2[2] <= table_height + 0.005:  # 5mm safety margin
                print(f"  [FILTER] Rejected grasp {i+1}: finger tips below table (tip1_z={tip1[2]:.3f}m, tip2_z={tip2[2]:.3f}m)")
                continue
            
            # Check 4: Gripper body collision with table
            # The gripper extends in the approach direction, check if any part goes below table
            gripper_extent_in_approach = self.gripper.jaw_len + self.gripper.palm_depth
            furthest_point = position + actual_approach_direction * gripper_extent_in_approach
            
            if furthest_point[2] <= table_height + 0.01:  # 1cm safety margin
                print(f"  [FILTER] Rejected grasp {i+1}: gripper body below table (extent_z={furthest_point[2]:.3f}m)")
                continue
            
            # If all checks pass, add to valid grasps
            valid_grasps.append((i, grasp_data))
        
        print(f"[SELECTION] After table collision filtering: {len(valid_grasps)}/{len(grasp_data_list)} grasps remain")
        
        if not valid_grasps:
            print("[WARNING] No grasps remain after table collision filtering!")
            return None
        
        for original_idx, grasp_data in valid_grasps:
            try:
                pose = grasp_data['pose']
                base_score = grasp_data['score']
                position = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                
                # FIXED: Correct approach vector interpretation
                gripper_z_world = rotation_matrix[:, 2]  # Z-axis of gripper in world
                actual_approach_direction = -gripper_z_world  # Gripper approaches in -Z direction
                
                # === CRITERIA EVALUATION ===
                
                # 1. Distance to robot base (CORRECTED)
                distance_to_base = np.linalg.norm(position - robot_origin)
                distance_score = 1.0 / (1.0 + distance_to_base)
                
                # 2. Reachability penalty (ADJUSTED for robot's reach from new base position)
                max_reach = 0.855  # Panda's maximum reach from base
                reach_penalty = max(0.0, (distance_to_base - max_reach) * 3.0)
                
                # 3. Height preference (ENHANCED - stronger penalty for low grasps)
                height_score = 1.0
                if position[2] < table_height + 0.02:  # Too close to table
                    height_score = 0.05  # REDUCED: Strong penalty for very low grasps
                elif position[2] < table_height + 0.05:  # Slightly low
                    height_score = 0.3   # REDUCED: Medium penalty for low grasps
                elif position[2] > robot_origin[2] + 0.2:  # Too high above robot
                    height_score = 0.7
                
                # 4. ENHANCED: Approach direction alignment with stronger penalty for downward approaches
                approach_alignment = np.dot(actual_approach_direction, preferred_approach)
                approach_score = max(0.0, approach_alignment)
                
                # NEW: Additional penalty for approaches that are too horizontal or downward
                if actual_approach_direction[2] < 0.3:  # Not sufficiently upward
                    approach_penalty = 2.0 * (0.3 - actual_approach_direction[2])  # Penalty increases as approach becomes more horizontal/downward
                else:
                    approach_penalty = 0.0
                
                # 5. Orientation stability (unchanged)
                euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
                roll, pitch, yaw = euler_angles
                orientation_penalty = (
                    (abs(roll) / np.pi) * 0.5 +
                    (abs(pitch) / np.pi) * 0.3 +
                    (abs(yaw) / np.pi) * 0.2
                )
                
                # 6. Object proximity (unchanged)
                distance_to_object = np.linalg.norm(position - object_center)
                object_proximity_score = 1.0 / (1.0 + distance_to_object * 10.0)
                
                # 7. Enhanced collision avoidance (stricter table height check)
                collision_score = 0.01 if position[2] < table_height + 0.02 else 1.0  # ENHANCED: Stricter penalty
                
                # 8. Workspace preference (unchanged)
                workspace_distance = np.sqrt((position[0] - robot_origin[0])**2 + 
                                        (position[1] - robot_origin[1])**2)
                workspace_score = 1.0 / (1.0 + workspace_distance * 2.0)
                
                # === ENHANCED ENERGY FUNCTION (lower is better) ===
                energy = (
                    -2.5 * base_score +                    # Grasp quality (most important)
                    -1.0 * distance_score +                # Distance to robot
                    3.0 * reach_penalty +                  # Reachability penalty
                    -2.0 * height_score +                  # INCREASED: Height preference (more important)
                    -2.5 * approach_score +                # Approach direction (very important)
                    2.0 * approach_penalty +               # NEW: Penalty for non-upward approaches
                    0.8 * orientation_penalty +            # Orientation stability
                    -0.5 * object_proximity_score +        # Object proximity
                    -3.0 * collision_score +               # INCREASED: Collision avoidance (more important)
                    -1.0 * workspace_score                 # Workspace preference
                )
                
                # Debug output for first few grasps
                if original_idx < 3:
                    print(f"  [EVAL] Grasp {original_idx+1}:")
                    print(f"    Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                    print(f"    Actual approach dir: {actual_approach_direction}")
                    print(f"    Approach alignment: {approach_alignment:.3f} (score: {approach_score:.3f})")
                    print(f"    Approach penalty: {approach_penalty:.3f}")
                    print(f"    Distance to base: {distance_to_base:.3f}m (score: {distance_score:.3f})")
                    print(f"    Height: {position[2]:.3f}m (score: {height_score:.3f})")
                    print(f"    Base score: {base_score:.6f}")
                    print(f"    Total energy: {energy:.4f}")
                
                if energy < best_energy:
                    best_energy = energy
                    best_grasp = grasp_data.copy()
                    best_grasp['selection_info'] = {
                        'total_energy': energy,
                        'base_score': base_score,
                        'distance_to_base': distance_to_base,
                        'height': position[2],
                        'approach_alignment': approach_alignment,
                        'approach_penalty': approach_penalty,
                        'actual_approach_direction': actual_approach_direction.tolist(),
                        'orientation_penalty': orientation_penalty,
                        'workspace_score': workspace_score,
                        'rank': original_idx + 1
                    }
            
            except Exception as e:
                print(f"Error evaluating grasp {original_idx+1}: {e}")
                continue
        
        if best_grasp:
            info = best_grasp['selection_info']
            print(f"\n🎯 SELECTED BEST GRASP:")
            print(f"   Original rank: {info['rank']}")
            print(f"   Total energy: {info['total_energy']:.4f}")
            print(f"   Base score: {info['base_score']:.4f}")
            print(f"   Distance to robot base: {info['distance_to_base']:.3f}m")
            print(f"   Height: {info['height']:.3f}m")
            print(f"   Approach alignment: {info['approach_alignment']:.3f}")
            print(f"   Approach penalty: {info.get('approach_penalty', 0.0):.3f}")
            print(f"   Actual approach direction: {info['actual_approach_direction']}")
            print(f"   Workspace score: {info['workspace_score']:.3f}")
            
            # Show final pose
            best_pose = best_grasp['pose']
            pos = best_pose[:3, 3]
            euler = R.from_matrix(best_pose[:3, :3]).as_euler('xyz')
            print(f"   FINAL POSITION: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"   FINAL ORIENTATION: [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}] rad")
            
            # NEW: Verify the selected grasp is safe
            approach_dir = np.array(info['actual_approach_direction'])
            if pos[2] > table_height + 0.02 and approach_dir[2] >= -0.5:
                print(f"   ✅ SAFETY CHECK PASSED: Grasp is above table and not approaching from below")
            else:
                print(f"   ⚠️  WARNING: Selected grasp may have safety issues!")
        
        return best_grasp

    # Also add this method to the SuperquadricGraspPlanner class
    def plan_grasps_with_best_selection(self, point_cloud_path, shape, scale, euler, translation, max_grasps=5):
        """
        Enhanced plan_grasps that includes intelligent best grasp selection
        
        Returns:
            Tuple: (all_grasp_data, best_grasp_data)
        """
        # Get all diverse grasps using existing method
        all_grasp_data = self.plan_grasps(point_cloud_path, shape, scale, euler, translation, max_grasps)
        
        if not all_grasp_data:
            return [], None
        
        # Select the best grasp using multi-criteria
        object_center = np.mean([data['translation'] for data in all_grasp_data], axis=0)
        best_grasp_data = self.select_best_grasp_with_criteria(all_grasp_data, object_center)
        
        return all_grasp_data, best_grasp_data

    def visualize_multi_sq_grasps(self, points, superquadrics, all_grasps_data, 
                                 window_name="Multi-Superquadric Grasps", 
                                 highlight_final=True, final_count=15):
        """
        Visualize multiple superquadrics with their grasps using existing gripper visualization
        
        Args:
            points: object point cloud (Nx3)
            superquadrics: list of Superquadric objects (our S objects)
            all_grasps_data: list of dicts with keys ['pose', 'sq_index', 'score', 'is_final']
            window_name: visualization window name
            highlight_final: whether to highlight final selected grasps
            final_count: number of final grasps to highlight
        """
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
            geometries = [pcd]
            
            # Colors for superquadrics and grippers
            colors = [
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0]
            ]
            
            # Add superquadric surface points - NOW WE CAN USE S OBJECTS DIRECTLY!
            for i, S in enumerate(superquadrics):
                try:
                    # Use existing surface sampling with S objects (has ε1, ε2, etc.)
                    sq_surface_points = sample_superquadric_surface(S, n_samples=2000)
                    if len(sq_surface_points) > 0:
                        sq_pcd = o3d.geometry.PointCloud()
                        sq_pcd.points = o3d.utility.Vector3dVector(sq_surface_points)
                        color = colors[i % len(colors)]
                        sq_pcd.paint_uniform_color(color)
                        geometries.append(sq_pcd)
                        print(f"Added superquadric {i+1} visualization with {len(sq_surface_points)} surface points")
                    else:
                        print(f"No surface points generated for superquadric {i+1}")
                except Exception as e:
                    print(f"Failed to visualize superquadric {i+1}: {e}")
                    # Add a simple marker for failed superquadrics
                    try:
                        center = S.T  # S objects have .T attribute
                        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                        marker.translate(center)
                        color = colors[i % len(colors)]
                        marker.paint_uniform_color(color)
                        geometries.append(marker)
                        print(f"Added marker for superquadric {i+1} at {center}")
                    except Exception as marker_error:
                        print(f"Failed to add marker for superquadric {i+1}: {marker_error}")
            
            # Add grasps using EXISTING gripper mesh approach
            for i, grasp_data in enumerate(all_grasps_data):
                try:
                    grasp_pose = grasp_data['pose']
                    sq_index = grasp_data.get('sq_index', 0)
                    is_final = grasp_data.get('is_final', False)
                    score = grasp_data.get('score', 0.0)
                    
                    # Use color based on superquadric
                    base_color = colors[sq_index % len(colors)]
                    
                    # FIXED: Only highlight final grasps if highlight_final is True
                    if highlight_final and is_final and i < final_count:
                        if i == 0:  # Best grasp
                            gripper_color = (1.0, 0.0, 0.0)  # Bright red for best
                        else:
                            gripper_color = tuple(base_color)  # Keep SQ color for other finals
                    elif highlight_final and not is_final:
                        # Make non-final grasps more transparent/muted when highlighting is enabled
                        gripper_color = tuple(c * 0.6 for c in base_color)
                    else:
                        # FIXED: When highlight_final=False, use full base color for ALL grasps
                        gripper_color = tuple(base_color)
                    
                    # Use EXISTING gripper mesh generation from THIS class
                    gripper_meshes = self.gripper.make_open3d_meshes(colour=gripper_color)
                    
                    # Transform all gripper meshes to the grasp pose
                    for mesh in gripper_meshes:
                        mesh.transform(grasp_pose)
                        geometries.append(mesh)
                    
                    # FIXED: Coordinate frame size logic
                    if highlight_final and is_final and i == 0:
                        frame_size = 0.03  # Large frame for best grasp when highlighting
                    elif highlight_final and is_final:
                        frame_size = 0.02  # Medium frame for other final grasps when highlighting
                    else:
                        frame_size = 0.015  # Small frame for all others (or when not highlighting)
                    
                    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
                    coord_frame.transform(grasp_pose)
                    geometries.append(coord_frame)
                    
                    # FIXED: Logging logic
                    pos = grasp_pose[:3, 3]
                    if highlight_final and is_final and i < 5:  # Only log final grasps when highlighting
                        marker = "★" if i == 0 else "→"
                        print(f"  {marker} Final grasp {i+1}: SQ{sq_index+1}, score={score:.6f}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    elif not highlight_final and i < 5:  # Log first few grasps when not highlighting
                        print(f"  → Grasp {i+1}: SQ{sq_index+1}, score={score:.6f}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], rotation={grasp_pose[:3, :3].tolist()}")
                    
                except Exception as e:
                    print(f"Failed to visualize grasp {i+1}: {e}")
            
            # Add main coordinate frame
            main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geometries.append(main_coord_frame)
            
            # Create informative window title
            final_grasps = sum(1 for g in all_grasps_data if g.get('is_final', False))
            total_grasps = len(all_grasps_data)
            title = f"{window_name} ({len(superquadrics)} SQs, {final_grasps}/{total_grasps} grasps)"
            
            print(f"Showing visualization: {len(superquadrics)} superquadrics, {total_grasps} total grasps, {final_grasps} final")
            
            # Show visualization
            o3d.visualization.draw_geometries(
                geometries,
                window_name=title,
                zoom=0.7,
                front=[0, -1, 0],
                lookat=np.mean(points, axis=0),
                up=[0, 0, 1]
            )
            
            # Print legend
            print(f"\n{'='*60}")
            print(f"MULTI-SUPERQUADRIC GRASPS VISUALIZATION:")
            print(f"{'='*60}")
            print(f"  🟫 Gray points: Object point cloud")
            for i, S in enumerate(superquadrics):
                color = colors[i % len(colors)]
                print(f"  🟦 SQ {i+1}: RGB{color} - Surface points")
            if highlight_final:
                print(f"  🔴 Red gripper: BEST final grasp")
                print(f"  🟨 Colored grippers: Other final grasps (colored by source SQ)")
                print(f"  🌫️  Muted grippers: Non-final grasps")
            else:
                print(f"  🤖 Colored grippers: All grasps (colored by source SQ)")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Multi-SQ visualization error: {e}")
            import traceback
            traceback.print_exc()
   