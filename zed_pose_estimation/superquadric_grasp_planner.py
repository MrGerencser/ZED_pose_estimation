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
# 1. Superquadric & Gripper definitions (exact from generate_grasps.py)
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
    def __init__(self, jaw_len=0.054, max_open=0.08):
        """
        jaw_len : length of each finger (m)
        max_open: maximum opening width (m)
        In gripper‐local frame: λ is +X, tip positions are at ±(max_open/2, 0, 0).
        """
        self.jaw_len = jaw_len
        self.max_open = max_open
        self.lambda_local = np.array([0.0, 1.0, 0.0])  # closing line direction in gripper‐local
        self.approach_local = np.array([0.0, 0.0, -1.0])  # approach direction (toward object)

        # (tip_offset_local unused, origin P_G is at mid‐point)
        self.tip_offset_local = np.zeros(3)

# ------------------------------------------------------------
# 2. Utility: rotation from vector u → v (exact from generate_grasps.py)
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
# 3. Candidate Generation (exact from generate_grasps.py)
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
    # print(f"[DEBUG] Gripper closing line λ: {G.lambda_local}")
    
    # # 🔧 PAPER METHOD: For each principal axis λi (i = x,y,z)
    # # for i, axis_world in enumerate(S.axes_world.T):  # iterate over λ_x, λ_y, λ_z
    # # 🔧 FOR GRASPS FROM ABOVE: Use λ₃ (Z-axis) sweeps
    # for i, axis_world in enumerate(S.axes_world.T[2:3]):  # Only λ₃ (Z-axis)
    #     axis_name = "λz"

    #     print(f"[INFO] Creating set Δ{axis_name}:")
    #     print(f"  Step 1: Align gripper closing line λ with {axis_name} = {axis_world}")
        
    #     # Step 1: Compute rotation that aligns gripper's closing line λ → principal axis
    #     R_align = rotation_from_u_to_v(G.lambda_local, axis_world)
        
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
    #             closing_dir_after = R_final @ G.lambda_local
    #             alignment_check = np.dot(closing_dir_after, axis_world)
    #             print(f"    θ={theta_deg:3d}°: closing_dir={closing_dir_after}, alignment={alignment_check:.3f}")
        
    #     print(f"  → Created set Δ{axis_name} with {rotations_in_set} poses")
    
    # expected_total = 3 * (360 // step_deg)  # 3 axes × (360/step_deg) rotations
    # print(f"[INFO] Paper method complete: {len(candidate_R)} candidates (expected: {expected_total})")
    
    # # 🔧 VERIFICATION: Check that closing line is always aligned with one of the principal axes
    # print(f"[DEBUG] Verification - checking alignment of first few candidates:")
    # for i in range(min(9, len(candidate_R))):  # Check first 3 from each axis
    #     R_candidate = candidate_R[i]
    #     closing_dir = R_candidate @ G.lambda_local
        
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
    #         print(f"    ⚠️ WARNING: Poor alignment detected!")
    
    # return candidate_R
    
    """
    §2.1 – CORRECTED: Automatically select the most vertical axis for top-down grasps
    """
    candidate_R = []
    
    print(f"[INFO] Implementing paper's method: align λ with most vertical principal axis")
    print(f"[DEBUG] Principal axes in world frame:")
    print(f"  λ_x: {S.axes_world[:, 0]}")
    print(f"  λ_y: {S.axes_world[:, 1]}")  
    print(f"  λ_z: {S.axes_world[:, 2]}")
    print(f"[DEBUG] Gripper closing line λ: {G.lambda_local}")
    
    # 🔧 AUTOMATIC AXIS SELECTION: Find axis closest to [0, 0, 1]
    vertical_ref = np.array([0.0, 0.0, 1.0])  # World Z-axis (up)
    
    axis_alignments = []
    axis_names = ["λx", "λy", "λz"]
    
    for i, axis in enumerate(S.axes_world.T):
        # Calculate how aligned this axis is with vertical (absolute value for up/down)
        alignment = abs(np.dot(axis, vertical_ref))
        axis_alignments.append(alignment)
        print(f"  {axis_names[i]}: {axis} → vertical alignment = {alignment:.6f}")
    
    # Select the most vertical axis
    best_axis_idx = np.argmax(axis_alignments)
    best_alignment = axis_alignments[best_axis_idx]
    best_axis_name = axis_names[best_axis_idx]
    best_axis_vector = S.axes_world[:, best_axis_idx]
    
    print(f"[INFO] Selected {best_axis_name} as most vertical axis (alignment={best_alignment:.6f})")
    
    # For "top-down" grasps where the closing action is along the most vertical SQ axis,
    # ensure this axis effectively points downwards or is horizontal.
    # If the selected most vertical SQ axis points upwards (positive Z),
    # flip it so the gripper closing line aligns with a downward-pointing or horizontal axis.
    if best_axis_vector[2] > 1e-6:  # If Z component is significantly positive (points upwards)
        print(f"[INFO] Original {best_axis_name} ({best_axis_vector}) points upwards (Z={best_axis_vector[2]:.6f}).")
        best_axis_vector = -best_axis_vector
        print(f"[INFO] Flipped {best_axis_name} to point downwards/horizontally: {best_axis_vector}")
    
    print(f"[INFO] Using {best_axis_name} = {best_axis_vector} for top-down grasps (this vector defines closing direction)")
    
    # 🔧 SINGLE AXIS SWEEP: Use only the most vertical axis
    axis_world = best_axis_vector
    axis_name = best_axis_name

    print(f"[INFO] Creating set Δ{axis_name} (TOP-DOWN GRASPS):")
    print(f"  Step 1: Align gripper closing line λ with {axis_name} = {axis_world}")
    
    # Step 1: Compute rotation that aligns gripper's closing line λ → principal axis
    R_align = rotation_from_u_to_v(G.approach_local, axis_world)
    
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
            closing_dir_after = R_final @ G.approach_local
            alignment_check = np.dot(closing_dir_after, axis_world)
            print(f"    θ={theta_deg:3d}°: closing_dir={closing_dir_after}, alignment={alignment_check:.3f}")
    
    print(f"  → Created set Δ{axis_name} with {rotations_in_set} poses")
    
    print(f"[INFO] Auto-axis selection complete: {len(candidate_R)} candidates using most vertical axis")
    
    return candidate_R

def extra_sweeps_special_shapes(S: Superquadric, base_R_list, G: Gripper):
    """
    §2.2 – For prism-like (ε1→0) or cuboid-like (ε2→0): slide gripper along SQ-local axes in 15mm grid.
            For cylinder-like (ε1→0, ε2=1, ax=ay): rotate about λ_z in π/8 increments.
    """
    R_list_full = list(base_R_list)
    poses_offsets = []

    # Heuristic thresholds (from the paper)
    prism_like   = (S.ε1 < 0.3)  # ε1 → 0
    cuboid_like  = (S.ε2 < 0.3)  # ε2 → 0
    
    # 🔧 FIX: Complete cylinder condition from paper
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

    # 🔧 Handle cylinder case FIRST (most specific)
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
    
    # 🔧 Handle prism/cuboid cases (translation grids)
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
# 4. Candidate Filtering (exact from generate_grasps.py)
# ------------------------------------------------------------

# Support threshold κ (paper default: 12)
κ_support = 3

def support_test(R, t, G: Gripper, kdtree: KDTree, κ=κ_support):
    """
    Place small cylinders at each jaw tip:
      tip1 = t + R @ ( +max_open/2, 0, 0 )
      tip2 = t + R @ ( -max_open/2, 0, 0 )
    Count points within radius = jaw_len/2 around each tip.
    If both counts ≥ κ, pass; else fail.
    """
    half_open = G.max_open / 2.0
    tip1 = t + R @ np.array([ half_open, 0.0, 0.0 ])
    tip2 = t + R @ np.array([-half_open, 0.0, 0.0 ])

    r_support = G.jaw_len / 2.0
    cnt1 = len(kdtree.query_ball_point(tip1, r=r_support))
    cnt2 = len(kdtree.query_ball_point(tip2, r=r_support))
    return (cnt1 >= κ) and (cnt2 >= κ)

def collision_test(R, t, S: Superquadric, G: Gripper, kdtree: KDTree):
    """
    🔧 CORRECTED: Exactly implement paper's collision detection
    
    Paper method:
    1. Create cylinder: radius=lj, height=min(ax, lw/2), center=SQ_center, direction=grasp_axis
    2. Find points within base planes but OUTSIDE cylinder interior
    3. If any such points exist → collision
    """
    
    # Gripper closing direction (grasping axis)
    λ_dir = R[:, 0]  # X-axis is closing direction
    
    # 🔧 FIX 1: Use the grasping axis directly for height calculation
    # The paper uses the SQ axis that corresponds to the current grasping direction
    # For grasp from Δλx: use ax, for Δλy: use ay, for Δλz: use az
    
    # Transform grasping direction to SQ-local coordinates
    λ_local = S.R.T @ λ_dir  # SQ-local coordinates
    
    # Find which SQ axis is most aligned with grasping direction
    axis_alignments = np.abs(λ_local)
    dominant_axis_idx = np.argmax(axis_alignments)
    
    # Get corresponding SQ semi-axis length
    axis_lengths = [S.ax, S.ay, S.az]
    a_axis = axis_lengths[dominant_axis_idx]
    
    # 🔧 PAPER FORMULA: height = min(a_axis, lw/2)
    # where lw = max_open (gripper width)
    cylinder_height = min(a_axis, G.max_open / 2.0)
    cylinder_radius = G.jaw_len  # lj in paper
    
    # 🔧 FIX 2: Cylinder center at SUPERQUADRIC center (not grasp center)
    cylinder_center = S.T  # Paper: "centroid locates at the centroid of the superquadric"
    
    # print(f"    Collision cylinder: center={cylinder_center}, radius={cylinder_radius:.3f}, height={cylinder_height:.3f}")
    # print(f"    Grasping axis: {λ_dir}, aligned with SQ axis {dominant_axis_idx} (length={a_axis:.3f})")
    
    # Get all object points
    X = kdtree.data  # (N,3) object points
    
    # 🔧 FIX 3: Project points onto grasping axis from SQ center
    relative_positions = X - cylinder_center  # Relative to SQ center
    projections = relative_positions @ λ_dir  # Projection onto grasping axis
    
    # 🔧 STEP 1: Find points within the two base planes
    half_height = cylinder_height
    within_base_planes = np.abs(projections) <= half_height
    
    if not np.any(within_base_planes):
        return True  # No points in cylinder region → no collision
    
    # Get points within the base planes
    slab_points = X[within_base_planes]
    slab_projections = projections[within_base_planes]
    
    print(f"    {len(slab_points)} points within base planes")
    
    # 🔧 STEP 2: Check which points are OUTSIDE the cylinder interior
    # Calculate radial distance from cylinder axis for each point in slab
    
    # Vector from cylinder center to each slab point
    slab_relative = slab_points - cylinder_center
    
    # Remove component along cylinder axis to get radial component
    axial_components = np.outer(slab_projections, λ_dir)  # (N, 3)
    radial_vectors = slab_relative - axial_components  # (N, 3)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)  # (N,)
    
    # 🔧 PAPER LOGIC: Points that are within planes but OUTSIDE cylinder
    outside_cylinder = radial_distances > cylinder_radius
    collision_points = slab_points[outside_cylinder]
    
    print(f"    {len(collision_points)} collision points (outside cylinder, within planes)")
    
    # 🔧 PAPER RESULT: If any collision points exist → reject grasp
    has_collision = len(collision_points) > 0
    
    if has_collision:
        print(f"    ❌ COLLISION: {len(collision_points)} points would collide")
        # Show first few collision points for debugging
        for i in range(min(3, len(collision_points))):
            pt = collision_points[i]
            dist = radial_distances[outside_cylinder][i]
            print(f"      Collision point {i+1}: {pt} (radial_dist={dist:.3f} > {cylinder_radius:.3f})")
    else:
        # print(f"NO COLLISION: All points either outside planes or inside cylinder")
        pass
    
    return not has_collision  # Return True if NO collision

# ------------------------------------------------------------
# 5. Scoring (exact from generate_grasps.py)
# ------------------------------------------------------------

def score_grasp(R, t, S: Superquadric, G: Gripper, kdtree: KDTree):
    """
    Compute composite score h(g) = h_α · h_β · h_γ · h_δ ∈ (0,1].
    """
    X = kdtree.data  # (N,3)
    N = X.shape[0]

    # -------- First, identify points on/near superquadric surface --------
    R_inv = S.R.T
    rel_pts = X - S.T  # (N,3) in world
    pts_slocal = rel_pts @ R_inv  # (N,3) in SQ-local
    eps1, eps2 = S.ε1, S.ε2
    x_n = np.abs(pts_slocal[:, 0]) / S.ax
    y_n = np.abs(pts_slocal[:, 1]) / S.ay
    z_n = np.abs(pts_slocal[:, 2]) / S.az
    term_xy = (x_n**(2.0/eps1) + y_n**(2.0/eps1))**(eps1/eps2)
    implicit_vals = np.abs(term_xy + z_n**(2.0/eps2) - 1.0)
    
    # Identify points Y that correspond to superquadric (within tolerance)
    surface_tol = 0.05  # 5cm tolerance for identifying surface points
    surface_mask = implicit_vals < surface_tol
    Y = X[surface_mask]
    
    if len(Y) == 0:
        # No points near surface - bad fit
        h_α = 0.0
        h_β = 0.0
        α = float('inf')
        β = 0.0
    else:
        # 🔧 CORRECTED h_α: Point-to-surface distance (Equation 6)
        # Sample points densely on superquadric surface
        S_surface = sample_superquadric_surface(S, n_samples=1000)
        
        # Build KDTree for SURFACE points (not Y points)
        tree_surface = KDTree(S_surface)
        
        # 🔧 CORRECT: For each point yi in Y, find minimum distance to surface S
        # This implements: α = (1/N) * Σ min ||yi - sj||₂
        distances_Y_to_S, _ = tree_surface.query(Y)  # For each Y point, closest S point
        α = np.mean(distances_Y_to_S)  # Average of minimum distances
        
        # 🔧 DEBUG: Show what we're actually calculating
        # print(f"    |Y| = {len(Y)} surface points identified")
        # print(f"    |S| = {len(S_surface)} sampled surface points")
        # print(f"    α = mean(min_dist_Y_to_S) = {α:.6f}")
        
        q_α = 0.002  # Adjust based on expected point-to-surface distances
        h_α = np.exp(- (α**2) / q_α)

        # -------- h_β: Coverage (Equation 8) --------
        # 🔧 CORRECTED h_β: Surface coverage by Y points
        # Build KDTree for Y points to find coverage
        tree_Y = KDTree(Y)
        d_th = 0.01  # 1cm threshold for coverage
        
        # For each sampled surface point, check if it's close to any Y point
        distances_S_to_Y, _ = tree_Y.query(S_surface)
        T_mask = distances_S_to_Y <= d_th  # Surface points covered by Y
        T_count = np.sum(T_mask)
        
        β = T_count / len(S_surface)  # |T| / |S|
        h_β = β**2
        
        # 🔧 DEBUG: Show coverage calculation
        # print(f"    T_count = {T_count} surface points covered by Y")
        # print(f"    β = |T|/|S| = {T_count}/{len(S_surface)} = {β:.6f}")

    # 🔧 ADD: Position-dependent scoring to differentiate grasps
    # Even though grasps are at same position, we can score based on 
    # gripper placement relative to object geometry
    
    # Gripper jaw positions in world frame
    half_open = G.max_open / 2.0
    tip1_world = t + R @ np.array([ half_open, 0.0, 0.0])
    tip2_world = t + R @ np.array([-half_open, 0.0, 0.0])
    
    # 🔧 h_γ: Gripper alignment with object principal axes
    # Check how well gripper closing direction aligns with object axes
    gripper_closing_dir = R[:, 0]  # X-axis is closing direction
    
    # Calculate alignment with each principal axis
    axis_alignments = []
    for axis in S.R.T:  # Each row is a principal axis in world frame
        alignment = abs(np.dot(gripper_closing_dir, axis))
        axis_alignments.append(alignment)
    
    # Prefer grasps aligned with principal axes (more stable)
    max_axis_alignment = max(axis_alignments)
    
    # Shape-based curvature factor
    shape_curvature = 1.0 / (S.ε1 * S.ε2 + 0.1)
    γ = shape_curvature * (2.0 - max_axis_alignment)  # Lower is better
    γ = np.clip(γ, 0.1, 5.0)
    
    q_γ = 2.0
    h_γ = np.exp(- (γ**2) / q_γ)

    # 🔧 CORRECTED h_γ: Surface curvature at gripper contact points (Equation 10)
    
    # Calculate gripper jaw tip positions in world frame
    half_open = G.max_open / 2.0
    tip1_world = t + R @ np.array([ half_open, 0.0, 0.0])
    tip2_world = t + R @ np.array([-half_open, 0.0, 0.0])
    
    # Compute Gaussian curvature at both contact points
    curv1 = gaussian_curvature_at_point(tip1_world, S)
    curv2 = gaussian_curvature_at_point(tip2_world, S)
    
    # Average Gaussian curvature γ around the gripper endpoints
    γ = (curv1 + curv2) / 2.0
    
    # 🔧 DEBUG: Show curvature calculation
    # print(f"    tip1_world = {tip1_world}, curvature = {curv1:.6f}")
    # print(f"    tip2_world = {tip2_world}, curvature = {curv2:.6f}")
    # print(f"    γ = average_curvature = {γ:.6f}")
    
    # Score: lower curvature (flatter surface) gets higher score
    q_γ = 0.5  # Hyperparameter controlling curvature sensitivity
    h_γ = np.exp(- (γ**2) / q_γ)

    # -------- h_δ: Centroid proximity (unchanged) --------
    centroid = np.mean(X, axis=0)
    δ = np.linalg.norm(t - centroid)
    q_δ = 0.005
    h_δ = np.exp(- (δ**2) / q_δ)

    
    # : Final score with orientation consideration
    final_score = h_α * h_β * h_γ * h_δ


    # 🔧 DEBUG: Print enhanced breakdown
    # print(f"    α={α:.6f}, h_α={h_α:.6f}")
    # print(f"    β={β:.6f}, h_β={h_β:.6f}")
    # print(f"    γ={γ:.6f}, h_γ={h_γ:.6f} (axis_align={max_axis_alignment:.3f})")
    # print(f"    δ={δ:.6f}, h_δ={h_δ:.6f}")
    # print(f"    final_score={final_score:.10f}")
    # print(f"    surface_points: {len(Y)}/{N}")

    return final_score

def gaussian_curvature_at_point(point_world, S: Superquadric):
    """
    Compute Gaussian curvature of superquadric surface at a given world point.
    🔧 FIXED: Scale-aware curvature for small objects
    """
    # Transform point to SQ-local coordinates
    point_local = S.R.T @ (point_world - S.T)
    x, y, z = point_local
    
    eps1, eps2 = S.ε1, S.ε2
    ax, ay, az = S.ax, S.ay, S.az
    
    # 🔧 DEBUG: Show what's causing the high scale factor
    # print(f"      Scale params: ax={ax:.6f}, ay={ay:.6f}, az={az:.6f}")
    # print(f"      Shape params: ε1={eps1:.6f}, ε2={eps2:.6f}")
    
    # 🔧 CRITICAL: Scale-aware curvature calculation for small objects
    
    # Calculate characteristic size (geometric mean of scales)
    char_size = (ax * ay * az) ** (1/3)  # Geometric mean
    
    # 🔧 NEW: Normalize curvature by object size
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
    
    # 🔧 FIXED: Size-normalized scale factor
    # Smaller objects get LOWER curvature penalty (counter-intuitive but needed for small objects)
    size_normalized_factor = 1.0 / (size_ratio + 0.5)  # Add 0.5 to prevent division issues
    
    # Position factor: very conservative for small objects
    r_local = np.sqrt((x/ax)**2 + (y/ay)**2 + (z/az)**2)
    position_factor = 1.0 + 0.2 * r_local  # Much more conservative
    
    # 🔧 FINAL: Combine all factors with small object compensation
    base_curvature = shape_factor * size_normalized_factor * position_factor
    
    # 🔧 CRITICAL: For very small objects, clamp to much lower range
    if char_size < 0.05:  # Objects smaller than 5cm
        max_curvature = 1.5  # Much lower max for small objects
    elif char_size < 0.10:  # Objects smaller than 10cm
        max_curvature = 2.5
    else:
        max_curvature = 5.0  # Original max for larger objects
    
    gaussian_curv = np.clip(base_curvature, 0.1, max_curvature)
    
    # 🔧 DEBUG: Show the scale-aware calculation
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
                if support_test(Rg, tg, G, kdtree) and collision_test(Rg, tg, S, G, kdtree):
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


            # 6.3 Cull invalid candidates (Step 3)
            G_valid = []
            for Rg, tg in G_raw:
                if support_test(Rg, tg, G, kdtree) and collision_test(Rg, tg, S, G, kdtree):
                    G_valid.append((Rg, tg))
            print(f"[INFO] {len(G_valid)} grasps remain after support & collision filtering.")

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
    
    def filter_grasps_from_above(self, grasp_poses, workspace_z_min=0.0, approach_height=0.05):
        """
        Filter grasps to ensure they approach from above the workspace
        
        Args:
            grasp_poses: List of 4x4 transformation matrices
            workspace_z_min: Minimum Z value of workspace (table height)
            approach_height: Minimum height above object for approach
        
        Returns:
            List of filtered grasp poses that approach from above
        """
        filtered_grasps = []
        
        for i, grasp_pose in enumerate(grasp_poses):
            # Extract position and orientation
            position = grasp_pose[:3, 3]
            rotation_matrix = grasp_pose[:3, :3]
            
            # Get the gripper approach direction (usually -Z axis of gripper frame)
            approach_vector = -rotation_matrix[:, 2]  # Negative Z-axis (approach direction)
            
            # Check if grasp position is above workspace
            if position[2] <= workspace_z_min:
                # print(f"Grasp {i+1}: REJECTED - Below workspace (z={position[2]:.3f} <= {workspace_z_min})")
                continue
            
            # Check if approach vector points downward (positive Z component means upward approach)
            if approach_vector[2] <= 0:  # Should be approaching from above (downward)
                # Flip the gripper orientation to approach from above
                # print(f"Grasp {i+1}: FLIPPING - Converting to approach from above")
                
                # Rotate 180 degrees around X or Y axis to flip approach direction
                flip_rotation = np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])
                
                # Apply flip to rotation matrix
                new_rotation = rotation_matrix @ flip_rotation
                
                # Create new grasp pose with flipped orientation
                new_grasp_pose = np.eye(4)
                new_grasp_pose[:3, :3] = new_rotation
                new_grasp_pose[:3, 3] = position
                
                # Verify the new approach direction
                new_approach_vector = -new_rotation[:, 2]
                if new_approach_vector[2] > 0:  # Now approaching from above
                    #print(f"Grasp {i+1}: ACCEPTED after flip - Approach from above (z_approach={new_approach_vector[2]:.3f})")
                    filtered_grasps.append(new_grasp_pose)
                else:
                    # print(f"Grasp {i+1}: REJECTED - Still not approaching from above after flip")
                    pass
            else:
                # print(f"Grasp {i+1}: ACCEPTED - Already approaching from above (z_approach={approach_vector[2]:.3f})")
                filtered_grasps.append(grasp_pose)
            
            # Additional check: Ensure grasp is at reasonable height above object
            if len(filtered_grasps) > 0:
                final_position = filtered_grasps[-1][:3, 3]
                if final_position[2] < workspace_z_min + approach_height:
                    print(f"Grasp {i+1}: WARNING - Very close to workspace surface (z={final_position[2]:.3f})")
        
        print(f"Grasp filtering: {len(filtered_grasps)}/{len(grasp_poses)} grasps approach from above")
        return filtered_grasps