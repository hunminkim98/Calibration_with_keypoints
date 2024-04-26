import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import json
import pprint
from keypoints_confidence_multi import extract_paired_keypoints_with_reference
from keypoints_confidence_multi import extract_high_confidence_keypoints
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import toml
import re
import pickle

# Constants for initial intrinsic matrix ( Factory setting in the paper but im using calibrate app in Matlab or OpenCV )
## It would be changed input data from Pose2Sim intrinsic calibration
K1 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

K2 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

K3 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

K4 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

Ks = [K1, K2, K3, K4]

###################### Data Processing ############################

# camera directories
ref_cam_dir = r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json1' # reference camera directory
other_cam_dirs = [r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json2', r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json3', r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json4'] # other camera directories
confidence_threshold = 0.8 # confidence threshold for keypoints pair extraction

# Call the function to extract paired keypoints
paired_keypoints_list = extract_paired_keypoints_with_reference(ref_cam_dir, other_cam_dirs, confidence_threshold)
print(f"type of paired_keypoints_list : {type(paired_keypoints_list)}")

print(f"Number of paired Frames: {len(paired_keypoints_list)}")
total_keypoints = sum([len(frame_keypoints) for frame_keypoints in paired_keypoints_list])
print(f"Total number of paired keypoints: {total_keypoints}")
frame_keypoints_counts = [len(frame) for frame in paired_keypoints_list]

def unpack_keypoints(paired_keypoints_list):
    """
    Unpacks the paired keypoints from a list of paired keypoints.

    Args:
        paired_keypoints_list (list): List of paired keypoints.

    Returns:
        tuple: A tuple containing two lists, where each list contains the x and y coordinates of the keypoints.
    """
    points1, points2 = [], []
    
    for frame in paired_keypoints_list:
        for point in frame:
            if len(point) == 2:
                u1, v1 = point[0]
                u2, v2 = point[1]
                points1.append((u1, v1))       
                points2.append((u2, v2))
    print(f"shape of points1 : {np.array(points1).shape}")
    print(f"shape of points2 : {np.array(points2).shape}")
    return points1, points2

def extract_individual_camera_keypoints(paired_keypoints_list):
    """
    Extracts individual camera keypoints from a list of paired keypoints.

    Args:
        paired_keypoints_list (list): A list of paired keypoints.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the keypoints
        for Camera1, where the keys are in the format "Camera1_1-X" (e.g., "Camera1_1-2", "Camera1_1-3", etc.),
        and the values are lists of keypoints for each frame. The second dictionary contains the keypoints
        for other cameras, where the keys are the camera indices (starting from 2), and the values are lists
        of keypoints for each frame.
    """
    camera1_keypoints_pairs = {}

    for i, camera_pair in enumerate(paired_keypoints_list):
        other_camera_index = i + 2  # camera index (starting from 2)
        camera1_key = f"Camera1_1-{other_camera_index}"  # e.g., "Camera1_1-2", "Camera1_1-3", etc.
        camera1_keypoints_pairs[camera1_key] = []
        
        for frame in camera_pair:
            frame_keypoints_camera1 = []
            for keypoints_pair in frame:
                if len(keypoints_pair) == 2:
                    frame_keypoints_camera1.append(keypoints_pair[0])
            
            camera1_keypoints_pairs[camera1_key].append(frame_keypoints_camera1)

    # Extract keypoints for other cameras
    other_cameras_keypoints = {}
    for i, camera_pair in enumerate(paired_keypoints_list):
        other_camera_index = i + 2  # camera index (starting from 2)
        other_cameras_keypoints[other_camera_index] = []

        for frame in camera_pair:
            frame_keypoints_other_camera = []
            for keypoints_pair in frame:
                if len(keypoints_pair) == 2:
                    frame_keypoints_other_camera.append(keypoints_pair[1])

            other_cameras_keypoints[other_camera_index].append(frame_keypoints_other_camera)
            
    return camera1_keypoints_pairs, other_cameras_keypoints

# Extract individual camera keypoints
camera1_keypoints_pairs, other_cameras_keypoints = extract_individual_camera_keypoints(paired_keypoints_list)

###################### Data Processing ############################

###################### Function of Extrinsics parameters optimisation ############################

def compute_fundamental_matrix(paired_keypoints_list):
    """
    Compute the fundamental matrix from paired keypoints.

    This function takes a list of paired keypoints and computes the fundamental matrix using the RANSAC algorithm.

    Args:
        paired_keypoints_list (list): A list of tuples, where each tuple contains two arrays of keypoints, one for each image.

    Returns:
        numpy.ndarray: The computed fundamental matrix.
    """
    points1, points2 = unpack_keypoints(paired_keypoints_list)
    
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)

    # Compute the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    return F



def compute_essential_matrix(F, K1, K2):
    """
    Compute the essential matrix given the fundamental matrix and camera calibration matrices.

    Args:
        F (numpy.ndarray): The fundamental matrix.
        K1 (numpy.ndarray): The calibration matrix of camera 1.
        K2 (numpy.ndarray): The calibration matrix of other camera.

    Returns:
        numpy.ndarray: The computed essential matrix.
    """
    E = K2.T @ F @ K1
    #print(f"Essential matrix: {E}")
    return E


def decompose_essential_matrix(E):
    """
    Decomposes the essential matrix into rotation and translation components.

    Parameters:
    E (numpy.ndarray): The essential matrix.

    Returns:
    list: A list of tuples, where each tuple contains a possible combination of rotation and translation matrices.
    """
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Ensure the rotation matrix is right-handed
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Two possible translations
    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Four possible combinations of R and t
    combinations = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

    return combinations


def cam_create_projection_matrix(K, R, t):
    """
    Creates the camera projection matrix.

    Args:
        K (numpy.ndarray): The camera's intrinsic parameters matrix.
        R (numpy.ndarray): The rotation matrix.
        t (numpy.ndarray): The translation vector.

    Returns:
        numpy.ndarray: The created projection matrix.
    """
    RT = np.hstack([R, t.reshape(-1, 1)])
    return K @ RT


def triangulate_points(paired_keypoints_list, P1, P2):
    """
    Triangulates a list of paired keypoints using the given camera projection matrices.

    Args:
        paired_keypoints_list (list): List of paired keypoints for each frame.
        P1 (array-like): Camera projection matrix for the reference camera.
        P2 (array-like): Camera projection matrix for the other camera.

    Returns:
        list: List of 3D points corresponding to the triangulated keypoints for each frame.
    """
    points_3d = []

    for frame in paired_keypoints_list:
        points_3d_frame = []

        for point in frame:
            x1, y1 = point[0]
            x2, y2 = point[1]

            # Triangulate the point
            point_3d = cv2.triangulatePoints(P1, P2, (x1, y1), (x2, y2))
            point_3d /= point_3d[3]

            points_3d_frame.append(point_3d[:3])

        points_3d.append(points_3d_frame)

    return points_3d

# Triangulate the points using all combinations of R and t
def triangulate_all_combinations(paired_keypoints_list, K, K1, combinations):
    """
    Triangulates 3D points from all combinations of camera poses.

    Args:
        paired_keypoints_list (list): List of paired keypoints.
        K (array): Camera calibration matrix for camera 1.
        K1 (array): Camera calibration matrix for camera 2.
        combinations (list): List of tuples containing rotation matrix R and translation vector t.

    Returns:
        list: List of 3D points obtained from triangulation for each combination.
    """
    all_points_3d = []

    for R, t in combinations:
        P1 = cam_create_projection_matrix(K, np.eye(3), np.zeros((3, 1)))
        P2 = cam_create_projection_matrix(K1, R, t)
        points_3d = triangulate_points(paired_keypoints_list, P1, P2)
        all_points_3d.append(points_3d)
    return all_points_3d

# Triangulation for multi-camera system
def weighted_triangulation(P_all,x_all,y_all,likelihood_all):
    '''
    Triangulation with direct linear transform,
    weighted with likelihood of joint pose estimation.
    
    INPUTS:
    - P_all: list of arrays. Projection matrices of all cameras
    - x_all,y_all: x, y 2D coordinates to triangulate
    - likelihood_all: likelihood of joint pose estimation
    
    OUTPUT:
    - Q: array of triangulated point (x,y,z,1.)
    '''
    
    A = np.empty((0,4))
    for c in range(len(x_all)):
        P_cam = P_all[c]
        A = np.vstack((A, (P_cam[0] - x_all[c]*P_cam[2]) * likelihood_all[c] ))
        A = np.vstack((A, (P_cam[1] - y_all[c]*P_cam[2]) * likelihood_all[c] ))
        
    if np.shape(A)[0] >= 4:
        S, U, Vt = cv2.SVDecomp(A)
        V = Vt.T
        Q = np.array([V[0][3]/V[3][3], V[1][3]/V[3][3], V[2][3]/V[3][3], 1])
    else: 
        Q = np.array([np.nan,np.nan,np.nan,1])
        
    return Q


# Visualize the 3D points
# def plot_3d_points(points_3d):
#     """
#     Plots a set of 3D points.

#     Args:
#         points_3d (list): List of frames, where each frame is a list of 3D points represented as (x, y, z) coordinates.

#     Returns:
#         None
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     for frame in points_3d:
#         for point in frame:
#             ax.scatter(point[0], point[1], point[2], c='b', marker='o')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     plt.show()



# Find the valid combination of R and t

### 2024.02.12 added
#This function should be modify to transfrom the 3D points to the camera coordinate system and check if all 3D points have positive depth
#But, the result is same as the previous version. 
###

## New version ##
def find_valid_rt_combination(all_points_3d, combinations):
    for i, (R, t) in enumerate(combinations):
        # make the transformation matrix
        transformation_matrix = np.hstack((R, t.reshape(-1, 1))) # transformation matrix
        transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1])) # homogeneous transformation matrix

        valid = True
        for frame_points_3d in all_points_3d[i]:
            for point_3d in frame_points_3d:
                # world to camera transformation
                point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates
                point_camera = transformation_matrix.dot(point_3d_homogeneous) # Transform the 3D point to the camera coordinate system
                Xc, Yc, Zc = point_camera[:3]

                # Check if all 3D points have positive depth
                if Zc <= 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return R, t
    print("No valid combination found.")
    return None, None

## Old version ##
# def find_valid_rt_combination(all_points_3d):
#     """
#     Finds a valid combination of rotation and translation for a given set of 3D points.

#     Parameters:
#     all_points_3d (list): A list of lists containing the 3D points for each frame.

#     Returns:
#     tuple: A tuple containing the valid rotation and translation combination.
#            If no valid combination is found, returns (None, None).
#     """
#     for i, frames_points_3d in enumerate(all_points_3d):
#         # Check if all 3D points have positive depth
#         valid = True
#         for frame_points_3d in frames_points_3d:
#             if not np.all(np.array(frame_points_3d)[:, 2] > 0):
#                 valid = False
#                 break
        
#         if valid:
#             return combinations[i]
#     print("No valid combination found.")
#     return None, None


# Compute the reprojection error with the valid R and t
def compute_reprojection_error(paired_keypoints_list, P1, P2):
    """
    Computes the reprojection error for a set of paired keypoints using the given projection matrices.

    Args:
        paired_keypoints_list (list): List of paired keypoints for each frame.
        P1 (array-like): Camera projection matrix for the reference camera.
        P2 (array-like): Camera projection matrix for the other camera.

    Returns:
        float: The mean reprojection error over all keypoints.
    """
    total_error = 0
    total_points = 0

    for frame in paired_keypoints_list:
        points1, points2 = zip(*frame)
        points1 = np.array(points1, dtype=float)
        points2 = np.array(points2, dtype=float)

        # Triangulate the points
        points_3d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d /= points_3d[3]

        # reproject the 3D points 
        points1_reprojected = P1 @ points_3d
        points1_reprojected /= points1_reprojected[2]

        points2_reprojected = P2 @ points_3d
        points2_reprojected /= points2_reprojected[2]

        # calculate the error
        error1 = np.sqrt(np.sum((points1_reprojected[:2].T - points1)**2, axis=1))
        error2 = np.sqrt(np.sum((points2_reprojected[:2].T - points2)**2, axis=1))

        total_error += np.sum(error1) + np.sum(error2)
        total_points += len(points1) + len(points2)

    mean_error = total_error / total_points
    return mean_error


###################### Function of Intrinsics parameters optimisation ############################

def compute_intrinsic_optimization_loss(x, points_3d, keypoints_detected, R, t):
    """
    Computes the loss for the intrinsic parameters optimization.

    Args:
    - x: Intrinsic parameters to optimize.
    - points_3d: List of 3D points (triangulated human body joints).
    - keypoints_detected: Original detected 2D keypoints.
    - R: Rotation matrix.
    - t: Translation vector.

    Returns:
    - The mean loss for the intrinsic parameters optimization.
    """
    f_x, f_y, u0, v0 = x  # Intrinsic parameters to optimize
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)

    total_loss = 0
    valid_keypoints_count = 0  # Counter for counting the number of valid detected points
    
    transformation_matrix = np.hstack((R, t.reshape(-1, 1)))  # transformation matrix
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # homogeneous transformation matrix
    # print(f"Transformation matrix: {transformation_matrix}")
    
    for frame_points_3d, frame_keypoints_detected in zip(points_3d, keypoints_detected):
        for point_3d, detected_point in zip(frame_points_3d, frame_keypoints_detected):
            if not isinstance(detected_point, tuple) or len(detected_point) != 2:
                continue
                
            u_detected, v_detected = detected_point
            valid_keypoints_count += 1
            
            # world to camera transformation
            point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates
            point_camera = transformation_matrix.dot(point_3d_homogeneous)
            Xc, Yc, Zc = point_camera[:3]  

            # Compute the loss
            loss = abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
            total_loss += loss

    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0
    print(f"mear_loss of intrinsic : {mean_loss}")
    return mean_loss

def optimize_intrinsic_parameters(points_3d, keypoints_detected, K, R, t):
    """
    Optimizes the intrinsic parameters using the given 3D points and detected keypoints.

    Args:
    - points_3d: List of 3D points (triangulated human body joints).
    - keypoints_detected: Original detected 2D keypoints.
    - K: Intrinsic parameters matrix.
    - R: Rotation matrix.
    - t: Translation vector.

    Returns:
    - The optimized intrinsic parameters matrix.
    """
    # Create the initial guess for the intrinsic parameters
    x0 = np.array([K[0, 0], K[1,1] ,K[0, 2], K[1, 2]])
     
    # Create the bounds for the intrinsic parameters
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    # Optimize the intrinsic parameters using the least squares method
    result = least_squares(compute_intrinsic_optimization_loss, x0, args=(points_3d, keypoints_detected, R, t), bounds=bounds, x_scale='jac', verbose=1, method='trf', loss= 'huber', diff_step=1e-8, tr_solver='lsmr', ftol=1e-12, max_nfev=50, xtol=1e-12, gtol=1e-12)

    # Create the optimized intrinsic matrix
    K_optimized = np.array([[result.x[0], 0, result.x[2]], [0, result.x[1], result.x[3]], [0, 0, 1]])

    return K_optimized

###################### Function of Intrinsics parameters optimisation ############################

###################### Parameter initialisation ############################

camera_Rt = {} # dictionary to store the extrinsic parameters for each camera
fundamental_matrices = {} # dictionary to store the fundamental matrices for each camera pair

# calculate initial extrinsic parameters for each camera pair
for j, K in enumerate(Ks):
    if j == 0:  # Skip the reference camera
        continue
    
    paired_keypoints = paired_keypoints_list[j-1]
    print(f"Camera {j+1} relative to Camera 1:")
    print(f"- Total keypoints extracted: {sum(len(frame_keypoints) for frame_keypoints in paired_keypoints)}")
    
    # Compute the fundamental matrix
    F = compute_fundamental_matrix(paired_keypoints)
    # print(f"Camera {j+1} relative to Camera 1: Fundamental matrix: {F}")

    # Store the fundamental matrix
    fundamental_matrices[(1, j+1)] = F

    # Compute the essential matrix
    E = compute_essential_matrix(F, K1, K)
    # print(f"Camera {j+1} relative to Camera 1: Essential matrix: {E}")

    # Decompose the essential matrix
    combinations = decompose_essential_matrix(E)
    # print(f"combinations : {combinations}")

    # Triangulate the points using all combinations of R and t
    all_points_3d = triangulate_all_combinations(paired_keypoints, K, K1, combinations)
    print(f"Camera {j+1} relative to Camera 1: Total 3D points: {sum(len(frame_points_3d) for frame_points_3d in all_points_3d)}")

    # Find the valid combination of R and t
    R, t = find_valid_rt_combination(all_points_3d, combinations)
    print(f"Camera {j+1} relative to Camera 1: R = {R}, t = {t}")

    # Rodrigues rotation vector
    R_rod, _ = cv2.Rodrigues(R)
    print(f"Rodrigues rotation vector: {R_rod}")

    # calculate the mean reprojection error
    P1 = cam_create_projection_matrix(K1, np.eye(3), np.zeros((3, 1)))
    P2 = cam_create_projection_matrix(K, R, t)
    mean_error = compute_reprojection_error(paired_keypoints, P1, P2)
    print(f"Camera {j+1} relative to Camera 1: Mean reprojection error: {mean_error}")

    # Store the extrinsic parameters
    camera_Rt[j+1] = (R, t)


# Print the extrinsic parameters and fundamental matrices
for camera, (R, t) in camera_Rt.items():
    print(f"Camera {camera}: R = {R}, t = {t}")
for camera_pair, F_matrix in fundamental_matrices.items():
    print(f"Fundamental matrix for Camera pair {camera_pair}: {F_matrix}")

###################### Parameter initialisation ############################

############################################# intrinsic jointly optimization #############################################

all_best_results = {}
# how many times to run the optimization
outer_iterations = 1
intrinsic_iterations = 1

# dictionary to store the optimization results
optimization_results = {}
Fix_K1 = Ks[0] # reference camera intrinsic parameters


# optimize the intrinsic and extrinsic parameters for each camera pair
for j in range(len(Ks)):
    if j == 0:
        continue # Skip the reference camera
    elif j == 1: # Camera0_1
        paired_keypoints = paired_keypoints_list[0]
    else:
        paired_keypoints = paired_keypoints_list[j-1]

    camera_pair_key = f"Camera0_{j}"  # e.g., "Camera0_1", "Camera0_2", etc.
    print(f"Optimizing for pair: {camera_pair_key}")
    K_optimized = Ks[j] 
    
    # Initialize the optimization results dictionary
    optimization_results[camera_pair_key] = {'K1': [], 'K2': [], 'R': [], 't': [], 'errors': [], 'losses': []}



    # optimize the extrinsic parameters
    for outer_iter in range(outer_iterations):
        print(f"--- Outer Iteration {outer_iter+1} for {camera_pair_key} ---")
        OPT_K = K_optimized # intrinsic parameters for the other camera
        if j == 1: # Camera0_1
            F = fundamental_matrices[(1, 2)] 
        else:
            F = fundamental_matrices[(1, j+1)]

        E = compute_essential_matrix(F, Fix_K1, OPT_K) # essential matrix
        combinations = decompose_essential_matrix(E) # decompose the essential matrix
        all_points_3d = triangulate_all_combinations(paired_keypoints, OPT_K, Fix_K1, combinations)
        R_optimized, t_optimized = find_valid_rt_combination(all_points_3d, combinations) # find the valid combination of R and t
        P1 = cam_create_projection_matrix(Fix_K1, np.eye(3), np.zeros((3, 1)))
        P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)
        outer_error = compute_reprojection_error(paired_keypoints,P1, P2) # calculate the mean reprojection error
        print(f"Camera pair {camera_pair_key} outer iteration {outer_iter+1}: Mean reprojection error: {outer_error}")
        
        # Store the optimization results
        optimization_results[camera_pair_key]['K1'].append(Fix_K1)
        optimization_results[camera_pair_key]['K2'].append(OPT_K)
        optimization_results[camera_pair_key]['R'].append(R_optimized)
        optimization_results[camera_pair_key]['t'].append(t_optimized)
        optimization_results[camera_pair_key]['errors'].append(outer_error)

        if R_optimized is None or t_optimized is None:
            break

        # optimize the intrinsic parameters
        for inner_iter in range(intrinsic_iterations):
            if j == 1: # Camera0_1
                camera1_key = f"Camera1_1-{2}"
            else:
                camera1_key = f"Camera1_1-{j+1}"
            
            P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)
            points_3d_optimized = triangulate_points(paired_keypoints, P1, P2) # update the 3D points
            loss = compute_intrinsic_optimization_loss([OPT_K[0, 0], OPT_K[1, 1], OPT_K[0, 2], OPT_K[1, 2]], points_3d_optimized, other_cameras_keypoints[j + 1], R_optimized, t_optimized)
            print(f"Camera pair {camera_pair_key} inner iteration {inner_iter+1}: Mean loss for OPT_K: {loss}")
            OPT_K_optimized = optimize_intrinsic_parameters(points_3d_optimized, other_cameras_keypoints[j + 1], OPT_K, R_optimized, t_optimized) # optimize the intrinsic parameters
            OPT_K = OPT_K_optimized # update the intrinsic parameters
            P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized) # update the projection matrix
            inner_error = compute_reprojection_error(paired_keypoints, P1, P2) # calculate the mean reprojection error
            print(f"Camera pair {camera_pair_key} inner iteration: Mean reprojection error: {inner_error}")

            # Store the optimization results
            optimization_results[camera_pair_key]['K1'].append(Fix_K1)
            optimization_results[camera_pair_key]['K2'].append(OPT_K)
            optimization_results[camera_pair_key]['R'].append(R_optimized)
            optimization_results[camera_pair_key]['t'].append(t_optimized)
            optimization_results[camera_pair_key]['errors'].append(inner_error)
            optimization_results[camera_pair_key]['losses'].append(loss)

        K_optimized = OPT_K # update the intrinsic parameters

    # Store the best results for each camera pair
    if optimization_results[camera_pair_key]['errors']:
        min_error_for_pair = min(optimization_results[camera_pair_key]['errors'])
        index_of_min_error = optimization_results[camera_pair_key]['errors'].index(min_error_for_pair)
        best_K1 = optimization_results[camera_pair_key]['K1'][index_of_min_error]
        best_K2 = optimization_results[camera_pair_key]['K2'][index_of_min_error]
        best_R = optimization_results[camera_pair_key]['R'][index_of_min_error]
        best_t = optimization_results[camera_pair_key]['t'][index_of_min_error]


        # Store the best results
        all_best_results[camera_pair_key] = {
            'K1': best_K1,
            'K2': best_K2,
            'R': best_R,
            't': best_t,
            'error': min_error_for_pair
        }

# Print the best results for each camera pair

for pair_key, results in all_best_results.items():
    print(f"Best results for {pair_key}:")
    print(f"- K1: {results['K1']}")
    print(f"- K2: {results['K2']}")
    print(f"- R: {results['R']}")
    print(f"- t: {results['t']}")
    print(f"- Minimum reprojection error: {results['error']}")


#################################################### intrinsic jointly optimization ####################################################

####################################################
##########EXTRINSIC PARAMETER OPTIMIZATION##########
####################################################

def compute_extrinsic_optimization_loss(x, ext_K, points_3d, points_2d, ext_R, ext_t):
    """
    Computes the loss for the extrinsic parameters optimization.

    Args:
    - x: Extrinsic parameters to optimize.
    - ext_K: Intrinsic parameters matrix.
    - points_3d: List of 3D points (triangulated human body joints).
    - points_2d: Original detected 2D keypoints.
    - ext_R: Rotation matrix.
    - ext_t: Translation vector.

    Returns:
    - The mean loss for the extrinsic parameters optimization.
    """
    f_x, f_y, u0, v0 = ext_K[0, 0], ext_K[1, 1], ext_K[0, 2], ext_K[1, 2]
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)

    # t vector from x
    # print(f"Optimization variable x: {x}")
    # t_magnitude = x[0]
    # normalized_t = ext_t / np.linalg.norm(ext_t)
    # recn_t = normalized_t * t_magnitude
    # print(f"Reconstructed t: {recn_t}")
    obj_t = x

    total_loss = 0
    valid_keypoints_count = 0  # Counter for counting the number of valid detected points


    transformation_matrix = np.hstack((ext_R, obj_t.reshape(-1, 1)))  # transformation matrix
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # homogeneous transformation matrix
    
    for frame_points_3d, frame_keypoints_detected in zip(points_3d, points_2d):
        for point_3d, detected_point in zip(frame_points_3d, frame_keypoints_detected):
            if not isinstance(detected_point, tuple) or len(detected_point) != 2:
                continue
                
            u_detected, v_detected = detected_point
            valid_keypoints_count += 1
            
            # world to camera transformation
            point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates
            point_camera = transformation_matrix.dot(point_3d_homogeneous)
            Xc, Yc, Zc = point_camera[:3]  

            # Compute the loss
            loss = abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
            total_loss += loss

    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0

    print(f"Mean loss of extrinsic : {mean_loss}")
    return mean_loss



def optimize_extrinsic_parameters(points_3d, other_cameras_keypoints, ext_K, ext_R, ext_t):
    """
    Optimizes the extrinsic parameters using the given 3D points and detected keypoints.

    Args:
    - points_3d: List of 3D points (triangulated human body joints).
    - other_cameras_keypoints: Original detected 2D keypoints for the other cameras.
    - ext_K: Intrinsic parameters matrix.
    - ext_R: Rotation matrix.
    - ext_t: Translation vector.

    Returns:
    - The optimized t vector.
    """
    # Create the initial guess for the extrinsic parameters (|T|) using the t vector magnitude
    # x0 = np.array([np.linalg.norm(ext_t)])
    x0 = ext_t
    print(f"Initial x0: {x0}")

    # Optimize the intrinsic parameters using the least squares method
    result = least_squares(compute_extrinsic_optimization_loss, x0, args=(ext_K, points_3d, other_cameras_keypoints, ext_R, ext_t), verbose=1, method='trf', diff_step=1e-8 , ftol=1e-12, max_nfev=50, xtol=1e-12, gtol=1e-12, x_scale='jac', loss='huber')

    optimized_t = result.x # optimized t vector
    print(f"Optimized t: {optimized_t}")
    # Create the optimized extrinsic t vector
    # t_magnitude = result.x[0]
    # print(f"Optimized t magnitude: {t_magnitude}")
    # t_optimized = ext_t / np.linalg.norm(ext_t) * t_magnitude

    return optimized_t

########################################
####### Multi-camera calibration #######
########################################
def unweighted_triangulation(P_all, x_coords, y_coords):
    """
    Perform triangulation using Direct Linear Transform.

    Parameters:
    - P_all: Projection matrices for all cameras.
    - x_coords: x coordinates of the keypoints from all cameras.
    - y_coords: y coordinates of the keypoints from all cameras.

    Returns:
    - The 3D point triangulated from the given coordinates.
    """
    A = np.empty((0, 4))
    for i, P in enumerate(P_all):
        A = np.vstack((A, P[0] - x_coords[i]*P[2]))
        A = np.vstack((A, P[1] - y_coords[i]*P[2]))
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[-1]  # Normalize

    return X[:3]

def create_P_all(all_best_results, Ks):
    P_all = []

    # Camera0_0에 대한 투영 행렬 계산
    K1 = Ks[0]
    R = np.eye(3)  # 대각 행렬
    t = np.zeros(3)  # 영벡터
    RT = np.hstack((R, t.reshape(-1, 1)))
    P = np.dot(K1, RT)
    P_all.append(P)

    for params in all_best_results.values():
        K = params['K2']
        R = params['R']
        t = params['t']
        # Validate and reshape R to 3x3 if necessary
        if R.shape != (3, 3):
            raise ValueError(f"Incorrect shape for R: {R.shape}")
        # Validate and reshape t to 3x1 if necessary
        if t.shape not in [(3,), (3, 1)]:
            if t.shape == (4,):  # Example condition, adjust based on your scenario
                t = t[:3]  # Assuming the last element is extraneous
            t = t.reshape(3, 1)
            print(f"t: {t}")
        # Concatenate R and t to form a 3x4 matrix
        RT = np.hstack((R, t.reshape(-1, 1)))
        # Multiply K by RT to get the projection matrix
        P = np.dot(K, RT)
        P_all.append(P)
    return P_all

def triangulate_points_all(P_all, all_keypoints):
    """
    Triangulate 3D points from the given keypoints.
    
    Args:
        - P_all: Projection matrices for all cameras.
        - all_keypoints: Keypoints detected from all cameras (without confidence values).
    """
    triangulated_points = []

    for frame_keypoints in all_keypoints:
        frame_points = {}

        for keypoint_idx in frame_keypoints:
            x_coords = []
            y_coords = []

            for cam in ['json1', 'json2', 'json3', 'json4']:  # Assuming camera names are 'json1', 'json2', 'json3', 'json4'
                if cam in frame_keypoints[keypoint_idx]:
                    keypoint = frame_keypoints[keypoint_idx][cam]
                    x_coords.append(keypoint[0])
                    y_coords.append(keypoint[1])

            X = unweighted_triangulation(P_all, x_coords, y_coords)
            frame_points[keypoint_idx] = X

        triangulated_points.append(frame_points)

    return triangulated_points

def plot_3d_points(triangulated_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame_points in triangulated_points:
        for point_id, point_coords in frame_points.items():
            ax.scatter(point_coords[0], point_coords[1], point_coords[2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def compute_reprojection_error(keypoints_detected, camera_params, P_all):
    n_cameras = camera_params.shape[0]
    print(f"Number of cameras: {n_cameras}")
    n_points = len(keypoints_detected)  # 총 프레임 수
    print(f"Number of points: {n_points}")

    error = np.zeros((n_points, n_cameras, 2))
    print(f"Error shape: {error.shape}")

    for i in range(n_cameras):
        for j in range(n_points):
            keypoints = keypoints_detected[j][i]
            print(f"keypoints: {keypoints}")
            valid_mask = np.isfinite(keypoints).all(axis=1)
            
            points_proj = (P_all[i] @ np.hstack((keypoints[valid_mask], np.ones((np.sum(valid_mask), 1)))).T).T
            points_proj = points_proj[:, :2] / points_proj[:, 2:]
            
            error[j, i, :] = points_proj - keypoints[valid_mask]

    return error


def bundle_adjustment_objective(camera_params, n_cameras, cam_ids, all_keypoints, P_all):
    camera_params = camera_params.reshape(n_cameras, 12)

    error = compute_reprojection_error(all_keypoints, camera_params, P_all)

    L = np.sqrt(np.sum(error**2, axis=1))
    print(f"cam_ids: {cam_ids}")
    b = 0.1
    TL = np.where(cam_ids == 1, (L + b * (n_cameras - 1)) / L, L)

    return TL.flatten()


def bundle_adjustment_jacobian_sparsity(n_cameras, n_points):
    m = n_points * n_cameras * 2
    n = n_cameras * 12

    A = lil_matrix((m, n), dtype=int)

    i = np.arange(n_points)
    for s in range(n_cameras):
        A[2 * s * n_points + 2 * i, s * 12:s * 12 + 12] = 1
        A[2 * s * n_points + 2 * i + 1, s * 12:s * 12 + 12] = 1

    return A

# Should look into what is problem. 2024.04.16
def optimize_bundle_adjustment(all_keypoints, P_all, max_iterations=100, tolerance=1e-4):
    n_cameras = len(P_all)
    n_points = len(all_keypoints)
    print(f"Number of points: {n_points}")
    cam_ids = [0] + [1] + [0] * (n_cameras - 2) # make cam_ids for constraints |T| = 1 in the cam 1 (Next to reference camera)

    camera_params = []
    for i, P in enumerate(P_all):
        if i == 0:  # reference camera
            R = np.eye(3)
            t = np.zeros((3, 1))
        else:
            R = P[:3, :3]
            print(f"R: {R}")
            t = P[:3, 3].reshape(3, 1)
            print(f"t: {t}")
        camera_params.append(np.hstack((R, t)).ravel())
    camera_params = np.concatenate(camera_params)
    x0 = camera_params

    def fun(params):
        camera_params = params.reshape(n_cameras, 12)
        camera_params[0, :] = x0[:12]  # 첫 번째 카메라(reference camera)의 파라미터는 고정
        return bundle_adjustment_objective(camera_params, n_cameras, cam_ids, all_keypoints, P_all)

    for _ in range(max_iterations):
        jac_sparsity = bundle_adjustment_jacobian_sparsity(n_cameras, n_points)

        res = least_squares(fun, x0, jac_sparsity=jac_sparsity, verbose=2, ftol=tolerance, method='trf', args=())

        optimized_camera_params = res.x.reshape(n_cameras, 12)
        optimized_camera_params[0, :] = x0[:12]  # 첫 번째 카메라(reference camera)의 파라미터는 고정

        # Check convergence
        if res.success or np.linalg.norm(res.x - x0) < tolerance:
            break
        else:
            x0 = res.x

    return optimized_camera_params


def run_bundle_adjustment(multi_cam_dir, confidence_threshold, Ks, all_best_results, max_iterations=100, tolerance=1e-4):
    # Extract high confidence keypoints from multiple cameras
    # output_dir = "C:/Users/5W555A/Desktop/Calibration/R_key_calib"
    all_keypoints = extract_high_confidence_keypoints(multi_cam_dir, confidence_threshold)
    # Save all_keypoints for inspection
    # all_keypoints_arr = np.array(all_keypoints, dtype=str)
    # all_keypoints_file = os.path.join(output_dir, 'all_keypoints.csv')
    # np.savetxt(all_keypoints_file, all_keypoints_arr, delimiter=',', fmt='%s')
        
            
    # Create projection matrices for all cameras
    P_all = create_P_all(all_best_results, Ks)
    print(f"P_all: {P_all}")

    points_3d_all = triangulate_points_all(P_all, all_keypoints)
    # points_3d_all_arr = np.array(points_3d_all, dtype=str)
    # points_3d_all_file = os.path.join(output_dir, 'points_3d_all.csv')
    # np.savetxt(points_3d_all_file, points_3d_all_arr, delimiter=',', fmt='%s')
    # should be well.
    # 2D structure: [{5: {'json1': (2449.83, 871.368), 'json2': (2286.5, 576.555), 'json3': (1551.67, 662.771), ...
    # 3D structure: [{5: array([ 0.40049726, -0.09746636,  1.28422782]), ...

    # Optimize camera parameters using bundle adjustment
    optimized_camera_params = optimize_bundle_adjustment(all_keypoints, P_all, max_iterations, tolerance)

    return optimized_camera_params

# Set the paths and parameters
multi_cam_dir = [r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json1', r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json2', r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json3', r'C:\Users\5W555A\Desktop\Calibration\R_key_calib\json4']
confidence_threshold = 0.8
max_iterations = 100
tolerance = 1e-4

# Run bundle adjustment
optimized_camera_params, optimized_points_3d = run_bundle_adjustment(multi_cam_dir, confidence_threshold, Ks, all_best_results, max_iterations, tolerance)

print("Optimized camera parameters:")
print(optimized_camera_params)
print("Optimized 3D points:")
print(optimized_points_3d)

































# N = 1 # how many times to run the optimization

# camera_params = {}

# # cam0 parameters
# camera_params["cam_00"] = {
#     "name": "cam0",
#     "size": [3840.0, 2160.0],  # 실제 이미지 크기로 변경해야 합니다.
#     "matrix": Ks[0].tolist(),
#     "distortions": [0.0, 0.0, 0.0, 0.0],  # 실제 왜곡 계수로 변경해야 합니다.
#     "rotation": np.eye(3).tolist(),
#     "translation": np.zeros((3, 1)).tolist(),
#     "fisheye": False,
# }

# for i, K in enumerate(Ks):
#     if i == 0:  # skip the reference camera
#         continue

#     camera_key = f"Camera1_1-{i+1}" # e.g., "Camera1_1-2", "Camera1_1-3", etc.
#     print(f"calibrating camera {i+1}...")

#     # import the best results for each camera pair
#     ext_K = all_best_results[f"Camera0_{i}"]['K2'] 
#     ext_R = all_best_results[f"Camera0_{i}"]['R'] # fixed R matrix
#     ext_t = all_best_results[f"Camera0_{i}"]['t']
#     ref_t = np.array([[0], [0], [0]]) # reference t vector |T| = 1


#     # projection matrix
#     P1 = cam_create_projection_matrix(Ks[0], np.eye(3), ref_t)
#     P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)

#     # triangulate points
#     points_3d = triangulate_points(paired_keypoints_list[i-1], P1, P2) # initial 3D points
#     before_optimization_error = compute_reprojection_error(paired_keypoints_list[i-1], P1, P2)
#     print(f"camera {i+1} before optimization error: {before_optimization_error}")

#     # keypoints for optimization
#     ref_keypoints_detected = camera1_keypoints_pairs[camera_key]
#     other_keypoints_detected = other_cameras_keypoints[i+1] # use the keypoints for the other camera

#     # Entrinsic and intrinsic parameter joint optimization
#     for n in range(N):

#         # extrinsic parameter optimization
#         print(f"before optimization t vector: {ext_t}")
#         optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t) # optimize extrinsic parameters
#         ext_t = optimized_t # update t vector
#         print(f"{n + 1}th optimized t vector: {ext_t}")

#         N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
#         ex_reprojection_error = compute_reprojection_error(paired_keypoints_list[i-1], P1, N_P2) # calculate the mean reprojection error
#         print(f"{n + 1}th error in extrinsic optimization = {ex_reprojection_error}")

#         # intrinsic parameter optimization
#         points_3d = triangulate_points(paired_keypoints_list[i-1], P1, N_P2) # update 3D points after extrinsic optimization
#         ext_K_optimized = optimize_intrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t) # optimize intrinsic parameters
#         ext_K = ext_K_optimized # update intrinsic parameters
#         print(f"{n + 1}th optimized K matrix: {ext_K}")

#         N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
#         in_reprojection_error = compute_reprojection_error(paired_keypoints_list[i-1], P1, N_P2) # calculate the mean reprojection error
#         print(f"{n + 1}th error in intrinsic optimization = {in_reprojection_error}")
#         points_3d = triangulate_points(paired_keypoints_list[i-1], P1, N_P2) # update 3D points after intrinsic optimization

#     # save result after optimization
#     all_best_results[f"Camera0_{i}"]['t'] = ext_t
#     all_best_results[f"Camera0_{i}"]['K2'] = ext_K

#     # ext_R matrix to rod vector
#     ext_R_rod, _ = cv2.Rodrigues(ext_R)
#     print(f"camera {i+1} R : {ext_R_rod}")

#     # Store camera parameters
#     camera_params[f"cam_{i+1:02d}"] = {
#         "name": f"cam{i+1}",
#         "size": [3840.0, 2160.0],  # 실제 이미지 크기로 변경해야 합니다.
#         "matrix": ext_K.tolist(),
#         "distortions": [0.0, 0.0, 0.0, 0.0],  # 실제 왜곡 계수로 변경해야 합니다.
#         "rotation": ext_R_rod.tolist(),
#         "translation": ext_t.tolist(),
#         "fisheye": False,
#     }

#     # print optimized results
#     for pair_key, results in all_best_results.items():
#         print(f"Best results for {pair_key}:")
#         print(f"- K2: {results['K2']}") # optimized intrinsic parameters
#         print(f"- t: {results['t']}") # optimized extrinsic parameters

#     # Save camera parameters to TOML file
#     output_dir = r"C:\Users\5W555A\Desktop\Calibration\R_key_calib"  # 출력 디렉토리 경로를 지정하세요.
#     output_name = "camera_params"  # 출력 파일 이름을 지정하세요.

#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{output_name}.toml")

#     # TOML 파일로 직접 쓰지 않고, 먼저 문자열로 변환합니다.
#     toml_str = toml.dumps(camera_params)

#     # 생성된 TOML 문자열에서 불필요한 쉼표를 제거합니다.
#     # 이 정규식은 배열의 마지막에 있는 쉼표를 찾아서 제거합니다.
#     toml_str_cleaned = re.sub(r",\s*\]", "]", toml_str)

#     # 사전 처리된 문자열을 파일에 씁니다.
#     with open(output_path, "w") as f:
#         f.write(toml_str_cleaned)