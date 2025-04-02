import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import torch.nn.functional as F
import open3d as o3d
import csv
from ultralytics import YOLO
from stl import mesh


# Helper function to visualize a point cloud
# def visualize_point_cloud(point_cloud, title="Point Cloud"):
#     # Create an Open3D PointCloud object and visualize
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
#     # Create coordinate frame
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
#     # Show point cloud with coordinate frame in a single window
#     o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name=title)

def visualize_point_cloud(pcd, title="Point Cloud"):
    '''
    visualize_point_cloud
    Visualize a point cloud using Open3D and display its first robust principal component
    '''
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Compute robust PCA of the point cloud
    points = np.asarray(pcd.points)
    centroid = np.median(points, axis=0)  # Use median for robustness
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    first_robust_pca = eigenvectors[:, np.argmax(eigenvalues)]

    # Create a line to represent the first robust principal component
    pca_line = o3d.geometry.LineSet()
    pca_line.points = o3d.utility.Vector3dVector([centroid, centroid + first_robust_pca * 0.1])
    pca_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    pca_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color for the PCA line

    # Show point cloud with coordinate frame and PCA line in a single window
    o3d.visualization.draw_geometries([pcd, coordinate_frame, pca_line], window_name=title)

    
    

# Filter out the outliers in the point cloud using the Statistical Outlier Removal filter
def filter_outliers_sor(pcd, nb_neighbors=20, std_ratio=1.5):
    # Apply statistical outlier removal
    filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Convert back to numpy array
    filtered_points = np.asarray(filtered_pcd.points)
    return filtered_points


# Function to fuse the point clouds based on centroid distance
def fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.1):
    # Group the point clouds by the class ID
    # The class dicts are of the form: {class_id1: [point_cloud1, point_cloud2, ...], class_id2: [point_cloud1, point_cloud2, ...]}
    pcs1 = []
    pcs2 = []
    class_dict1 = {}
    class_dict2 = {}

    # Iterate over the point clouds from camera 1 and group them by class ID
    # point_clouds_camera1 = [(pc, class_id), ...] is a list of tuples containing the point cloud and the class ID
    for pc, class_id in point_clouds_camera1:
        if class_id not in class_dict1:
            class_dict1[class_id] = [] # To store the point clouds for the class ID
        class_dict1[class_id].append(pc) # Append the point cloud to the list for the class ID

    # Iterate over the point clouds from camera 2 and group them by class ID
    for pc, class_id in point_clouds_camera2:
        if class_id not in class_dict2:
            class_dict2[class_id] = []
        class_dict2[class_id].append(pc)

    # After this loop, class_dict1 and class_dict2 contain the point clouds grouped by class ID

    # Initialize the fused point cloud list
    fused_point_clouds = []

    # Process each class ID
    # Get all the unique class IDs from both cameras
    # class_dict1.keys() returns a set of all the keys "class IDs" in the dictionary
    for class_id in set(class_dict1.keys()).union(class_dict2.keys()):
        # Get the point clouds for the current class ID from both cameras
        pcs1 = class_dict1.get(class_id, []) # pcs1 has the following format: [point_cloud1, point_cloud2, ...]
        pcs2 = class_dict2.get(class_id, [])

        # If there is only one point cloud with the same class ID from each camera we can directly fuse the pcs
        if len(pcs1) == 1 and len(pcs2) == 1:
            # Concatenate the point clouds along the vertical axis
            fused_pc = filter_outliers_sor(np.vstack((pcs1[0], pcs2[0])))
            fused_point_clouds.append((fused_pc, class_id))

        # If there are multiple point clouds with the same class ID from each camera, we need to find the best match
        else:
            for pc1 in pcs1:
                pc1 = filter_outliers_sor(pc1)
                best_distance = float('inf')
                best_match = None

                # Calculate the centroid of the point cloud from camera 1
                centroid1 = np.mean(pc1, axis=0)

                # Loop over all the point clouds from camera 2 with the same ID and find the best match based on centroid distance
                for pc2 in pcs2:
                    centroid2 = np.mean(pc2, axis=0)
                    # Calculate the Euclidean distance / L2 norm between the centroids
                    distance = np.linalg.norm(centroid1 - centroid2)

                    if distance < best_distance and distance < distance_threshold:
                        best_distance = distance
                        best_match = pc2
                        best_match = filter_outliers_sor(best_match)

                # If a match was found, fuse the point clouds
                if best_match is not None:
                    # Concatenate the point clouds along the vertical axis and filter out the outliers
                    fused_pc = np.vstack((pc1, best_match))
                    fused_point_clouds.append((fused_pc, class_id))
                    # Remove the matched point cloud from the list of point clouds from camera 2 to prevent duplicate fusion
                    pcs2 = [pc for pc in pcs2 if not np.array_equal(pc, best_match)]

                # If no match was found, simply add the point cloud from camera 1 to the fused point clouds
                else:
                    fused_point_clouds.append((pc1, class_id))

            # If any point clouds remain in the list from camera 2, add them to the fused point clouds
            for pc2 in pcs2:
                fused_point_clouds.append((pc2, class_id))

    return pcs1, pcs2, fused_point_clouds