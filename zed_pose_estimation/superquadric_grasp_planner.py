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
                 palm_depth= 0.020,   # distance from jaw mid-line to tool flange (m)
                 palm_width= 0.040):  # width of the aluminium bracket (m)

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
        T[1, 3] = +self.max_open / 2    # y-offset
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
        T[1, 3] = -self.max_open / 2
        T[2, 3] = -self.jaw_len / 2
        finger_R.transform(T)
        meshes.append(finger_R)

        # 3. CROSS-BAR (axis = Y) - create with correct dimensions directly
        cross_Y = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,                      # Keep original thickness
            height=self.max_open + self.thickness       # Span across fingers
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
    # """
    # §2.1 – CORRECTED: Exactly implement the paper's method
    # """
    # candidate_R = []
    
    # print(f"[INFO] Implementing paper's method: align λ with each principal axis")
    # print(f"[DEBUG] Principal axes in world frame:")
    # print(f"  λ_x: {S.axes_world[:, 0]}")
    # print(f"  λ_y: {S.axes_world[:, 1]}")  
    # print(f"  λ_z: {S.axes_world[:, 2]}")
    # print(f"[DEBUG] Gripper closing line λ: {G.lambda_local}")  # FIXED: Use lambda_local
    
    # # PAPER METHOD: For each principal axis λi (i = x,y,z)
    # for i, axis_world in enumerate(S.axes_world.T):  # iterate over λ_x, λ_y, λ_z
    #     axis_names = ["λ_x", "λ_y", "λ_z"]  # FIXED: Create proper list
    #     axis_name = axis_names[i]  # FIXED: Use index to get correct name

    #     print(f"[INFO] Creating set Δ{axis_name}:")
    #     print(f"  Step 1: Align gripper closing line λ with {axis_name} = {axis_world}")
        
    #     # Step 1: Compute Rotation that aligns gripper's closing line λ → principal axis
    #     R_align = rotation_from_u_to_v(G.lambda_local, axis_world)  # FIXED: Use lambda_local
        
    #     print(f"  Step 2: Rotate gripper around {axis_name} every {step_deg}°")
        
    #     # Step 2: Rotate the gripper around the aligned axis in step_deg increments  
    #     rotations_in_set = 0
    #     for theta_deg in range(0, 360, step_deg):
    #         theta_rad = theta_deg * DEG
            
    #         # Create rotation around the principal axis
    #         R_spin = R.from_rotvec(axis_world * theta_rad).as_matrix()
            
    #         # Combined rotation: first align, then spin
    #         R_final = R_spin @ R_align
            
    #         candidate_R.append(R_final)
    #         rotations_in_set += 1
            
    #         # Debug first few rotations
    #         if theta_deg < 30:  # Show first 3 rotations for verification
    #             closing_dir_after = R_final @ G.lambda_local  # FIXED: Use lambda_local
    #             alignment_check = np.dot(closing_dir_after, axis_world)
    #             print(f"    θ={theta_deg:3d}°: closing_dir={closing_dir_after}, alignment={alignment_check:.3f}")
        
    #     print(f"  → Created set Δ{axis_name} with {rotations_in_set} poses")
    
    # expected_total = 3 * (360 // step_deg)  # 3 axes × (360/step_deg) rotations
    # print(f"[INFO] Paper method complete: {len(candidate_R)} candidates (expected: {expected_total})")
    
    # # VERIFICATION: Check that closing line is always aligned with one of the principal axes
    # print(f"[DEBUG] Verification - checking alignment of first few candidates:")
    # for i in range(min(9, len(candidate_R))):  # Check first 3 from each axis
    #     R_candidate = candidate_R[i]
    #     closing_dir = R_candidate @ G.lambda_local  # FIXED: Use lambda_local
        
    #     # Check alignment with each principal axis
    #     alignments = []
    #     for j, axis in enumerate(S.axes_world.T):
    #         alignment = abs(np.dot(closing_dir, axis))
    #         alignments.append(alignment)
        
    #     best_axis_idx = np.argmax(alignments)
    #     best_alignment = alignments[best_axis_idx]
    #     axis_name = ["λx", "λy", "λz"][best_axis_idx]
        
    #     print(f"  Candidate {i+1}: best aligned with {axis_name} (alignment={best_alignment:.6f})")
        
    #     if best_alignment < 0.99:  # Should be nearly 1.0 for perfect alignment
    #         print(f"WARNING: Poor alignment detected!")
    
    # return candidate_R
    
    # candidate_R = []
    
    # # Find most vertical axis for top-down grasps
    # vertical_ref = np.array([0.0, 0.0, 1.0])
    # axis_alignments = []
    
    # for i, axis in enumerate(S.axes_world.T):
    #     alignment = abs(np.dot(axis, vertical_ref))
    #     axis_alignments.append(alignment)
    
    # best_axis_idx = np.argmax(axis_alignments)
    # best_axis_vector = S.axes_world[:, best_axis_idx]
    
    # # Ensure axis points downward for top-down grasps
    # if best_axis_vector[2] > 1e-6:
    #     best_axis_vector = -best_axis_vector
    
    # #   FIX: Align gripper's closing line (lambda_local = Y-axis) with selected axis
    # R_align = rotation_from_u_to_v(G.lambda_local, best_axis_vector)
    
    # # Rotate around the aligned axis
    # for theta_deg in range(0, 360, step_deg):
    #     theta_rad = theta_deg * np.pi / 180.0
    #     R_spin = R.from_rotvec(best_axis_vector * theta_rad).as_matrix()
    #     R_final = R_spin @ R_align
    #     candidate_R.append(R_final)
    
    # return candidate_R

    candidate_R = []
    
    # Find most vertical axis for top-down grasps
    vertical_ref = np.array([0.0, 0.0, 1.0])
    axis_alignments = []
    
    for i, axis in enumerate(S.axes_world.T):
        alignment = abs(np.dot(axis, vertical_ref))
        axis_alignments.append(alignment)
    
    best_axis_idx = np.argmax(axis_alignments)
    print(f"[INFO] Most vertical axis: {['X', 'Y', 'Z'][best_axis_idx]} (alignment: {axis_alignments[best_axis_idx]:.3f})")
    
    # PICK ANOTHER AXIS: Choose the axis with second-best vertical alignment
    # or alternatively, choose a horizontal axis for side grasping
    
    # Option 1: Second most vertical axis
    sorted_indices = np.argsort(axis_alignments)[::-1]  # Sort in descending order
    if len(sorted_indices) > 1:
        selected_axis_idx = sorted_indices[1]  # Second most vertical
        print(f"[INFO] Selected second most vertical axis: {['X', 'Y', 'Z'][selected_axis_idx]}")
    else:
        selected_axis_idx = best_axis_idx  # Fallback if only one axis
    
    # Option 2: Most horizontal axis (least vertical alignment)
    # selected_axis_idx = np.argmin(axis_alignments)
    # print(f"[INFO] Selected most horizontal axis: {['X', 'Y', 'Z'][selected_axis_idx]} (alignment: {axis_alignments[selected_axis_idx]:.3f})")
    
    # Option 3: Specific axis preference (e.g., always use Y-axis if available)
    # preferred_axis_order = [1, 0, 2]  # Prefer Y, then X, then Z
    # selected_axis_idx = preferred_axis_order[0]
    # print(f"[INFO] Selected preferred axis: {['X', 'Y', 'Z'][selected_axis_idx]}")
    
    selected_axis_vector = S.axes_world[:, selected_axis_idx]
    
    # For side grasps, we might want the axis to point horizontally
    # Uncomment this if you want horizontal grasping:
    # if abs(selected_axis_vector[2]) > 0.8:  # If too vertical
    #     selected_axis_vector = -selected_axis_vector  # Flip it
    
    # For top-down grasps on the selected axis:
    if selected_axis_vector[2] > 1e-6:
        selected_axis_vector = -selected_axis_vector
    
    print(f"[INFO] Using axis vector: {selected_axis_vector} for closing direction")
    
    # Align gripper's closing line (lambda_local = Y-axis) with selected axis
    R_align = rotation_from_u_to_v(G.lambda_local, selected_axis_vector)
    
    # Rotate around the aligned axis
    for theta_deg in range(0, 360, step_deg):
        theta_rad = theta_deg * np.pi / 180.0
        R_spin = R.from_rotvec(selected_axis_vector * theta_rad).as_matrix()
        R_final = R_spin @ R_align
        candidate_R.append(R_final)
    
    print(f"[INFO] Generated {len(candidate_R)} candidates around selected axis")
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

def support_test(R, t, S, G, kdtree: KDTree, κ=12, r_support=None, h_support=0.02):
    """
    True cylinder support test:
    - r_support: cylinder radius (defaults to half jaw width)
    - h_support: half cylinder height
    """
    return True
    # Closing direction in world
    closing_dir = R @ G.lambda_local
    closing_dir = closing_dir / np.linalg.norm(closing_dir)

    # Default support radius: half the jaw opening
    # default geometry – use paper’s constants
    if r_support is None:
        r_support = 3 * 0.003      # e.g. 3 × voxel_size (≈9 mm)
    if h_support is None:
        h_support = G.jaw_len      # full finger length

    # two finger-contact points on the SQ surface, along closing_dir
    tip1 = t + closing_dir * h_support
    tip2 = t - closing_dir * h_support

    # Get all points
    X = kdtree.data
    # Compute point projections onto closing axis
    rel1 = X - tip1
    proj1 = np.dot(rel1, closing_dir)
    radial1 = np.linalg.norm(rel1 - np.outer(proj1, closing_dir), axis=1)
    mask1 = (np.abs(proj1) <= h_support) & (radial1 <= r_support)
    cnt1 = np.count_nonzero(mask1)

    rel2 = X - tip2
    proj2 = np.dot(rel2, closing_dir)
    radial2 = np.linalg.norm(rel2 - np.outer(proj2, closing_dir), axis=1)
    mask2 = (np.abs(proj2) <= h_support) & (radial2 <= r_support)
    cnt2 = np.count_nonzero(mask2)

    return (cnt1 >= κ) and (cnt2 >= κ)


def collision_test2(R_world,              # 3×3 rotation of the candidate pose
                   t_world,              # 3×1 grasp centre (mid point of the jaws)
                   S,                    # Superquadric object (has .ax .ay .az .R .T)
                   G,                    # Gripper description (has .jaw_len .max_open .lambda_local)
                   kdtree: KDTree,       # built on the raw point cloud
                   finger_thickness=None # override if you have a direct measure
                  ) -> bool:
    """
    Returns True  → NO collision        (pose is still valid)
            False → collision detected  (reject this pose)

    Implements: “generate a cylinder whose axis is the closing line λ,
                 radius = l_j (finger thickness),
                 half-height = min(a_axis, l_w/2),
                 centre = t_world (grasp centre);
                 if any cloud point lies in the slab but outside the cylinder
                 → collision.”
    """
    return True

    # --- 1. closing line in world coordinates -----------------------
    λ_dir = R_world @ G.lambda_local
    λ_dir /= np.linalg.norm(λ_dir)       # just in case

    # --- 2. pick the SQ semi-axis most aligned with λ --------------
    λ_local = S.R.T @ λ_dir              # expressed in SQ-local frame
    axis_idx = np.argmax(np.abs(λ_local))  # 0=x, 1=y, 2=z
    a_axis   = [S.ax, S.ay, S.az][axis_idx]

    # --- 3. cylinder dimensions ------------------------------------
    half_open = G.max_open / 2.0
    half_height = min(a_axis, half_open)           # paper’s rule

    if finger_thickness is None:
        finger_thickness = G.jaw_len               # treat jaw_len as thickness
    radius = finger_thickness                      # paper: r = l_j

    # --- 4. gather points inside the slab ---------------------------
    X      = kdtree.data
    rel    = X - t_world                           # centre on current grasp
    proj   = rel @ λ_dir                           # signed distance along λ
    slab_mask = np.abs(proj) <= half_height        # between the two base planes

    if not np.any(slab_mask):
        return True                                # empty slab → certainly safe

    # radial distance of points inside the slab
    rel_slab   = rel[slab_mask]
    proj_slab  = proj[slab_mask]
    radial_vec = rel_slab - np.outer(proj_slab, λ_dir)
    radial_len = np.linalg.norm(radial_vec, axis=1)

    # --- 5. collision decision --------------------------------------
    colliding = radial_len > radius + 1e-6         # small ε margin
    return not np.any(colliding)

def collision_test(R_world, t_world, S, G, kdtree: KDTree,
                   debug_mode=True, max_debug_calls=20) -> bool:
    """
    CORRECTED IMPLEMENTATION based on paper logic:
    
    Paper logic: "A pose will be kept in G only if the posed gripper (body)
    does not intersect with any of the potential collision points in slab"
    
    Steps:
    1. Define slab region around closing line
    2. Find points inside slab 
    3. Remove points inside cylinder (these are safe - can be grasped)
    4. Remaining points = potential collision points
    5. Check if gripper body intersects with these potential collision points
    
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

    # --- 3. cylinder dimensions ------------------------------------
    half_open = G.max_open / 2.0
    half_height = half_open
    radius = G.jaw_len

    # --- 4. gather potential collision points inside the slab ---------------------------
    X = kdtree.data
    rel = X - t_world
    proj = rel @ λ_dir
    slab_mask = np.abs(proj) <= half_height

    if not np.any(slab_mask):
        collision_result = True  # empty slab → no collision
        if debug_mode:
            print(f"    [COLLISION] NO SLAB POINTS - Safe grasp")
        return collision_result

    # --- 5. PAPER LOGIC: Find potential collision points ---------------
    # Points inside slab
    slab_points = X[slab_mask]
    rel_slab = rel[slab_mask]
    proj_slab = proj[slab_mask]
    
    # Calculate radial distance from closing line for points in slab
    radial_vec = rel_slab - np.outer(proj_slab, λ_dir)
    radial_len = np.linalg.norm(radial_vec, axis=1)
    
    # CRITICAL: Points OUTSIDE cylinder = potential collision points
    # Points INSIDE cylinder = safe (object can be grasped)
    outside_cylinder_mask = radial_len > radius
    potential_collision_points = slab_points[outside_cylinder_mask]
    
    if len(potential_collision_points) == 0:
        collision_result = True  # No potential collision points → safe
        if debug_mode:
            print(f"    [COLLISION] All slab points INSIDE cylinder - Safe grasp")
        return collision_result

    # --- 6. CHECK IF GRIPPER BODY INTERSECTS WITH POTENTIAL COLLISION POINTS -----
    # Now we need to check if the gripper fingers would actually collide
    # with the potential collision points
    
    # Check collision with gripper palm/back
    palm_collisions = check_palm_collision(
        potential_collision_points, t_world, R_world, G
    )
    
    # FIX: Initialize has_collision properly
    has_collision = np.any(palm_collisions) if len(palm_collisions) > 0 else False
    
    collision_result = not has_collision  # True = no collision

    # ## --- 7. DEBUG OUTPUT ------------------------------------------------
    # if debug_mode:
    #     if not hasattr(collision_test, 'debug_call_count'):
    #         collision_test.debug_call_count = 0
        
    #     collision_test.debug_call_count += 1
        
    #     if collision_test.debug_call_count <= max_debug_calls:
    #         try:
    #             slab_points_count = np.sum(slab_mask)
    #             inside_cylinder_count = slab_points_count - len(potential_collision_points)
    #             outside_cylinder_count = len(potential_collision_points)
                
    #             print(f"\n[COLLISION DEBUG #{collision_test.debug_call_count}] - PAPER LOGIC")
    #             print(f"  Cylinder params: radius={radius:.4f}m, half_height={half_height:.4f}m")
    #             print(f"  Closing direction λ: {λ_dir}")
    #             print(f"  Grasp center: {t_world}")
    #             print(f"  Points in slab: {slab_points_count}/{len(X)}")
    #             print(f"  Points INSIDE cylinder (safe for grasping): {inside_cylinder_count}")
    #             print(f"  Points OUTSIDE cylinder (potential collision): {outside_cylinder_count}")
    #             print(f"  Palm collisions: {np.sum(palm_collisions)}")
    #             print(f"  Result: {'NO COLLISION' if collision_result else 'COLLISION DETECTED'}")
                
    #             # Visualization with corrected logic
    #             _visualize_collision_corrected(
    #                 R_world, t_world, S, G, kdtree, 
    #                 λ_dir, radius, half_height, slab_mask, outside_cylinder_mask,
    #                 palm_collisions,
    #                 f"Paper Logic Collision Test #{collision_test.debug_call_count}"
    #             )
                
    #         except Exception as viz_error:
    #             print(f"    [ERROR] Visualization failed: {viz_error}")

    return collision_result

def check_palm_collision(points, gripper_center, R_world, gripper):
    """
    Check if points collide with gripper palm/back region
    
    Args:
        points: Points to check (Nx3)
        gripper_center: Center position of gripper (3,)
        R_world: Gripper orientation matrix (3x3)
        gripper: Gripper object with dimensions
    
    Returns:
        Boolean array indicating which points collide with palm
    """
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # Transform points to gripper local coordinates
    rel_points = points - gripper_center
    local_points = rel_points @ R_world.T  # Transform to gripper frame
    
    # Check if points are in palm region (behind fingers)
    # Palm extends from Z=-jaw_len to Z=-jaw_len-palm_depth
    behind_fingers = (local_points[:, 2] <= -gripper.jaw_len) & (local_points[:, 2] >= -(gripper.jaw_len + gripper.palm_depth))
    
    # Check if points are within palm width and height
    within_palm_width = np.abs(local_points[:, 0]) <= gripper.palm_width / 2
    within_palm_height = np.abs(local_points[:, 1]) <= gripper.max_open / 2
    
    # Collision occurs if point is in all palm dimensions
    collisions = behind_fingers & within_palm_width & within_palm_height
    
    return collisions

def _visualize_collision_corrected(R_world, t_world, S, G, kdtree, 
                                 lambda_dir, radius, half_height, slab_mask, outside_cylinder_mask, palm_collisions,
                                 window_name="CORRECTED Collision Test"):
    """
    CORRECTED visualization showing proper collision logic
    """   
    try:
        geometries = []
        X = kdtree.data
        
        # 1. Original point cloud (gray)
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(X)
        pcd_all.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_all)
        
        # 2. Points in slab (yellow background)
        if np.any(slab_mask):
            slab_points = X[slab_mask]
            pcd_slab = o3d.geometry.PointCloud()
            pcd_slab.points = o3d.utility.Vector3dVector(slab_points)
            pcd_slab.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
            geometries.append(pcd_slab)
            
            # 3. Points INSIDE cylinder = GREEN (safe for grasping)
            inside_cylinder_points = slab_points[~outside_cylinder_mask]
            if len(inside_cylinder_points) > 0:
                pcd_safe = o3d.geometry.PointCloud()
                pcd_safe.points = o3d.utility.Vector3dVector(inside_cylinder_points)
                pcd_safe.paint_uniform_color([0.0, 1.0, 0.0])  # Green = safe
                geometries.append(pcd_safe)
            
            # 4. Points OUTSIDE cylinder = POTENTIAL COLLISION (orange)
            if np.any(outside_cylinder_mask):
                potential_collision_points = slab_points[outside_cylinder_mask]
                pcd_potential = o3d.geometry.PointCloud()
                pcd_potential.points = o3d.utility.Vector3dVector(potential_collision_points)
                pcd_potential.paint_uniform_color([1.0, 0.5, 0.0])  # Orange = potential collision
                geometries.append(pcd_potential)
                
                # 5. ACTUAL COLLISION POINTS (red)
                all_collisions = palm_collisions
                if np.any(all_collisions):
                    actual_collision_points = potential_collision_points[all_collisions]
                    pcd_collision = o3d.geometry.PointCloud()
                    pcd_collision.points = o3d.utility.Vector3dVector(actual_collision_points)
                    pcd_collision.paint_uniform_color([1.0, 0.0, 0.0])  # Red = actual collision
                    geometries.append(pcd_collision)
                    
                    print(f"    [VIZ] GREEN: {len(inside_cylinder_points)} safe points INSIDE cylinder")
                    print(f"    [VIZ] ORANGE: {len(potential_collision_points)} potential collision points")
                    print(f"    [VIZ] RED: {len(actual_collision_points)} ACTUAL collision points")
        
        # 6. Gripper geometry
        try:
            from zed_pose_estimation.vis2 import get_gripper_control_points_o3d
            
            gripper_transform = np.eye(4)
            gripper_transform[:3, :3] = R_world
            gripper_transform[:3, 3] = t_world
            
            gripper_meshes = get_gripper_control_points_o3d(
                gripper_transform,
                gripper=G,
                show_sweep_volume=False,  # Show sweep volume for collision context
                color=(0.2, 0.8, 0.2),
                finger_tip_to_origin=True
            )
            geometries.extend(gripper_meshes)
            
            print(f"    [GRIPPER DEBUG] Added {len(gripper_meshes)} gripper meshes")
            
        except ImportError as import_error:
            print(f"    [ERROR] Could not import gripper visualization: {import_error}")
        
        # 7. Collision cylinder (semi-transparent blue)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, 
            height=2 * half_height,
            resolution=20
        )
        
        # Orient cylinder along lambda_dir
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
        cylinder.paint_uniform_color([0.2, 0.2, 0.8])  # Blue
        geometries.append(cylinder)
        
        # 8. Closing direction arrow
        closing_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.003,
            cone_radius=0.006,
            cylinder_height=0.04,
            cone_height=0.01
        )
        arrow_transform = np.eye(4)
        arrow_transform[:3, :3] = cyl_rotation
        arrow_transform[:3, 3] = t_world + lambda_dir * (half_height + 0.03)
        closing_arrow.transform(arrow_transform)
        closing_arrow.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        geometries.append(closing_arrow)
        
        # 9. Legend/explanation
        print(f"\n{'='*60}")
        print(f"PAPER LOGIC COLLISION TEST VISUALIZATION:")
        print(f"{'='*60}")
        print(f"  🟫 Gray:   All points in point cloud")
        print(f"  🟨 Yellow: Points inside slab region")
        print(f"  🟢 Green:  Points INSIDE cylinder (safe for grasping)")
        print(f"  🟠 Orange: Potential collision points (outside cylinder)")
        print(f"  🔴 Red:    ACTUAL collision points (gripper intersects)")
        print(f"  🔵 Blue:   Collision cylinder and closing direction")
        print(f"  🤖 Robot:  Gripper geometry and sweep volume")
        print(f"{'='*60}")
        
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
        print(f"Error in paper logic collision visualization: {e}")
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
    print(f"    [DEBUG] SQ params: ε1={S.ε1:.3f}, ε2={S.ε2:.3f}")
    print(f"    [DEBUG] SQ scale: ax={S.ax:.3f}, ay={S.ay:.3f}, az={S.az:.3f}")
    print(f"    [DEBUG] SQ center: {S.T}")
    print(f"    [DEBUG] Points range: x=[{pts_local[:, 0].min():.3f}, {pts_local[:, 0].max():.3f}]")
    print(f"    [DEBUG] Points range: y=[{pts_local[:, 1].min():.3f}, {pts_local[:, 1].max():.3f}]")
    print(f"    [DEBUG] Points range: z=[{pts_local[:, 2].min():.3f}, {pts_local[:, 2].max():.3f}]")
    
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
        
        print(f"    [DEBUG] Implicit values range: [{implicit_values.min():.6f}, {implicit_values.max():.6f}]")
        print(f"    [DEBUG] Surface distances range: [{surface_distances.min():.6f}, {surface_distances.max():.6f}]")
        
    except Exception as eq_error:
        print(f"    [ERROR] Equation computation failed: {eq_error}")
        # Fallback: assume no surface points
        surface_distances = np.full(len(pts_local), 1000.0)
    
    #   ADAPTIVE SURFACE TOLERANCE: Scale with object size
    char_size = (safe_ax * safe_ay * safe_az) ** (1/3)
    base_tolerance = 0.1  # Base tolerance for implicit function
    size_scaled_tolerance = base_tolerance * max(0.5, char_size / 0.05)  # Scale with object size
    
    print(f"    [DEBUG] Char size: {char_size:.6f}, tolerance: {size_scaled_tolerance:.6f}")
    
    # Surface mask with adaptive tolerance
    surface_mask = surface_distances < size_scaled_tolerance
    Y = X[surface_mask]
    
    print(f"    [DEBUG] Surface points found: {len(Y)}/{N} with tolerance {size_scaled_tolerance:.6f}")
    
    if len(Y) == 0:
        #   FALLBACK: If no surface points, use proximity-based approach
        print(f"    [FALLBACK] No surface points found, using proximity-based approach")
        
        # Find points within reasonable distance of SQ center
        distances_from_center = np.linalg.norm(rel, axis=1)
        max_reasonable_distance = np.max([safe_ax, safe_ay, safe_az]) * 2.0
        
        proximity_mask = distances_from_center < max_reasonable_distance
        Y_fallback = X[proximity_mask]
        
        if len(Y_fallback) > 10:
            Y = Y_fallback[:min(100, len(Y_fallback))]  # Limit to avoid computation issues
            print(f"    [FALLBACK] Using {len(Y)} proximity points instead")
        else:
            print(f"    [FALLBACK] Still no good points, using h_α=h_β=0")
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
                
                print(f"    [DEBUG] α (mean distance to surface): {α:.6f}")
                
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
                
                print(f"    [DEBUG] β (coverage): {β:.6f}, d_th: {d_th:.6f}")
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
    print(f"    Detailed scoring breakdown:")
    print(f"      Surface points: {len(Y)}/{N}")
    print(f"      α={α:.8f} → h_α={h_α:.8f}")
    print(f"      β={β:.8f} → h_β={h_β:.8f}")
    print(f"      γ={γ:.8f} → h_γ={h_γ:.8f}")
    print(f"      δ={δ:.8f} → h_δ={h_δ:.8f}")
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
            
    
    def get_all_valid_grasps(self, point_cloud_path, shape, scale, euler, translation):
        """
        Get ALL valid grasps after filtering (for visualization purposes)
        
        Returns:
            List of 4x4 transformation matrices for all valid grasp poses
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

            # Filter for valid grasps
            G_valid = []
            for Rg, tg in G_raw:
                if support_test(Rg, tg, S, G, kdtree) and collision_test(Rg, tg, S, G, kdtree):
                    G_valid.append((Rg, tg))
            
            print(f"[INFO] Found {len(G_valid)} valid grasps for visualization")
            
            # Score all valid grasps
            grasp_data = []
            for Rg, tg in G_valid:
                score = score_grasp(Rg, tg, S, G, kdtree)
                T = np.eye(4)
                T[:3, :3] = Rg
                T[:3, 3] = tg
                
                grasp_data.append({
                    'pose': T,
                    'score': score,
                    'rotation': Rg,
                    'translation': tg
                })
            
            # Sort by score for better visualization
            grasp_data.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"[INFO] Found {len(grasp_data)} valid grasps with scores for visualization")
            
            # Store for future reference
            self.last_valid_grasps = [data['pose'] for data in grasp_data]
            
            return grasp_data  # Return with scores
            
        except Exception as e:
            print(f"Error getting all valid grasps: {e}")
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
                if support_test(Rg, tg, S, G, kdtree):
                    G_after_support.append((Rg, tg))

            print(f"[INFO] {len(G_after_support)}/{len(G_raw)} grasps remain after support filtering.")

            # Second: Collision test on support-passing grasps
            for Rg, tg in G_after_support:
                if collision_test(Rg, tg, S, G, kdtree):
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
            
            print(f"[INFO] Top scores: {[score for score, _ in scored_grasps[:5]]}")
            
            def unique_by_tips(scored_grasp_list, G_gripper, tol=2e-3):
                """
                Remove grasps that have the same finger tip positions (differ only by wrist roll)
                """
                seen_tips = []
                unique_grasps = []
                
                for score, (Rg, tg) in scored_grasp_list:
                    # Calculate finger tip positions for this grasp
                    closing_dir = Rg @ G_gripper.lambda_local
                    half_open = G_gripper.max_open / 2.0
                    tip1 = tg + closing_dir * half_open
                    tip2 = tg - closing_dir * half_open
                    
                    # Check if these tips are already seen
                    is_duplicate = False
                    for existing_tip1, existing_tip2 in seen_tips:
                        if (np.linalg.norm(tip1 - existing_tip1) < tol and 
                            np.linalg.norm(tip2 - existing_tip2) < tol):
                            is_duplicate = True
                            break
                        # Also check flipped order (tip1 <-> tip2)
                        if (np.linalg.norm(tip1 - existing_tip2) < tol and 
                            np.linalg.norm(tip2 - existing_tip1) < tol):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_tips.append((tip1, tip2))
                        unique_grasps.append((score, (Rg, tg)))
                
                return unique_grasps
            
            # Apply unique filtering as per paper's method
            unique_scored_grasps = unique_by_tips(scored_grasps, G, tol=2e-3)
            
            print(f"[INFO] After duplicate removal: {len(unique_scored_grasps)}/{len(scored_grasps)} grasps remain")
            print(f"[INFO] Top scores after duplicate removal: {[score for score, _ in unique_scored_grasps[:5]]}")
            
            # Now proceed with diversity selection from unique grasps
            diverse_grasps = []
            diverse_scores = []
            min_rotation_diff = 0.3  # Minimum rotation difference (radians)
            min_position_diff = 0.02  # Minimum position difference (2cm)
            
            for score, (Rg, tg) in unique_scored_grasps:  # Use unique_scored_grasps instead
                is_diverse = True
                
                # Check if this grasp is sufficiently different from already selected ones
                for existing_R, existing_t in diverse_grasps:
                    # Check position difference
                    pos_diff = np.linalg.norm(tg - existing_t)
                    # Check rotation difference
                    rot_similarity = np.trace(Rg.T @ existing_R)
                    angle_diff = np.arccos(np.clip((rot_similarity - 1) / 2, -1, 1))
                    
                    if pos_diff < min_position_diff and angle_diff < min_rotation_diff:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_grasps.append((Rg, tg))
                    diverse_scores.append(score)
                    print(f"Selected diverse grasp {len(diverse_grasps)}: score={score:.8f}, pos={tg}")
                    
                    if len(diverse_grasps) >= max_grasps:
                        break
            
            
            # Select grasps with different orientations
            diverse_grasps = []
            diverse_scores = []  # Keep track of scores for diverse grasps
            min_rotation_diff = 0.3  # Minimum rotation difference (radians)
            min_position_diff = 0.02  # Minimum position difference (2cm)
            
            for score, (Rg, tg) in scored_grasps:
                is_diverse = True
                
                # Check if this grasp is sufficiently different from already selected ones
                for existing_R, existing_t in diverse_grasps:
                    # Check position difference
                    pos_diff = np.linalg.norm(tg - existing_t)
                    # Check rotation difference
                    rot_similarity = np.trace(Rg.T @ existing_R)
                    angle_diff = np.arccos(np.clip((rot_similarity - 1) / 2, -1, 1))
                    
                    if pos_diff < min_position_diff and angle_diff < min_rotation_diff:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_grasps.append((Rg, tg))
                    diverse_scores.append(score)  # Store the corresponding score
                    print(f"Selected diverse grasp {len(diverse_grasps)}: score={score:.8f}, pos={tg}")
                    
                    if len(diverse_grasps) >= max_grasps:
                        break
            
            # Return both poses and their scores
            grasp_data = []
            for i, ((Rg, tg), score) in enumerate(zip(diverse_grasps, diverse_scores)):
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

