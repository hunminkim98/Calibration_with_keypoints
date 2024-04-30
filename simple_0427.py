import numpy as np
import cv2
import matplotlib.pyplot as plt
import toml
import time
import random
import os
import json
import pprint
from keypoints_confidence_multi import extract_paired_keypoints_with_reference
from scipy.optimize import least_squares
from scipy.optimize import minimize


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

Ks = [K1, K2, K3]

###################### Data Processing ############################

# camera directories
ref_cam_dir = r'D:\calibration\json1' # reference camera directory
other_cam_dirs = [r'D:\calibration\json2',r'D:\calibration\json3'] # other camera directories
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
def plot_3d_points(points_3d):
    """
    Plots a set of 3D points.

    Args:
        points_3d (list): List of frames, where each frame is a list of 3D points represented as (x, y, z) coordinates.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in points_3d:
        for point in frame:
            ax.scatter(point[0], point[1], point[2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



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
    # print(f"mear_loss of intrinsic : {mean_loss}")
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
    result = least_squares(compute_intrinsic_optimization_loss, x0, args=(points_3d, keypoints_detected, R, t), bounds=bounds, x_scale='jac', verbose=1, method='trf', loss= 'huber', diff_step=1e-8, tr_solver='lsmr', ftol=1e-12, max_nfev=150, xtol=1e-12, gtol=1e-12)

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

    # print(f"Mean loss of extrinsic : {mean_loss}")
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
    result = least_squares(compute_extrinsic_optimization_loss, x0, args=(ext_K, points_3d, other_cameras_keypoints, ext_R, ext_t), verbose=1, method='trf', diff_step=1e-8 , ftol=1e-8, max_nfev=100, xtol=1e-8, gtol=1e-8)

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

N = 10  # how many times to run the optimization

# reference camera의 pose 고정
ref_R = np.eye(3)  # identity matrix
ref_t = np.array([[0], [0], [0]])  # zero vector

# 두 번째 카메라의 translation vector 정규화 (스케일 기준)
t2_norm = all_best_results['Camera0_1']['t'] / np.linalg.norm(all_best_results['Camera0_1']['t'])

for i, K in enumerate(Ks):
    if i <= 1:  # skip the reference camera and the second camera
        continue

    camera_key = f"Camera1_1-{i+1}"  # e.g., "Camera1_1-3", "Camera1_1-4", etc.
    print(f"calibrating camera {i+1}...")

    # import the best results for each camera pair
    ext_K = all_best_results[f"Camera0_{i}"]['K2']
    ext_R = all_best_results[f"Camera0_{i}"]['R']  # fixed R matrix
    ext_t = all_best_results[f"Camera0_{i}"]['t']

    # projection matrix
    P1 = cam_create_projection_matrix(Ks[0], ref_R, ref_t)
    P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)

    # triangulate points
    points_3d = triangulate_points(paired_keypoints_list[i-1], P1, P2)  # initial 3D points
    before_optimization_error = compute_reprojection_error(paired_keypoints_list[i-1], P1, P2)
    print(f"camera {i+1} before optimization error: {before_optimization_error}")

    # keypoints for optimization
    ref_keypoints_detected = camera1_keypoints_pairs[camera_key]
    other_keypoints_detected = other_cameras_keypoints[i+1]  # use the keypoints for the other camera

    # Extrinsic and intrinsic parameter joint optimization
    for n in range(N):
        # extrinsic parameter optimization
        print(f"before optimization t vector: {ext_t}")
        optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t)
        ext_t = optimized_t  # update t vector
        print(f"{n + 1}th optimized t vector: {ext_t}")

        # 두 번째 카메라의 스케일에 맞춰 조정
        ext_t_adjusted = ext_t / np.linalg.norm(ext_t) * np.linalg.norm(t2_norm)
        print(f"after optimization t vector: {ext_t_adjusted}")
        ext_t = ext_t_adjusted

        N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)  # update projection matrix
        ex_reprojection_error = compute_reprojection_error(paired_keypoints_list[i-1], P1, N_P2)
        print(f"{n + 1}th error in extrinsic optimization = {ex_reprojection_error}")

        # intrinsic parameter optimization
        points_3d = triangulate_points(paired_keypoints_list[i-1], P1, N_P2)  # update 3D points after extrinsic optimization

    # save result after optimization
    all_best_results[f"Camera0_{i}"]['t'] = ext_t
    all_best_results[f"Camera0_{i}"]['K2'] = ext_K

    # ext_R matrix to rod vector
    ext_R_rod, _ = cv2.Rodrigues(ext_R)
    print(f"camera {i+1} R : {ext_R_rod}")

# print optimized results
for pair_key, results in all_best_results.items():
    print(f"Best results for {pair_key}:")
    print(f"- K2: {results['K2']}")  # optimized intrinsic parameters
    print(f"- t: {results['t']}")  # optimized extrinsic parameters

# 현재 코드가 존재하는 폴더 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# calib.toml 파일 경로 설정
calib_file = os.path.join(current_dir, "calib.toml")

# 결과를 저장할 dictionary 생성
calib_data = {}

for i in range(len(Ks)):
    camera_key = f"cam_{i+1:02d}"
    
    if i == 0:  # reference camera
        rotation = np.array([0, 0, 0], dtype=float)
        translation = np.array([0, 0, 0], dtype=float)
    else:
        rotation, _ = cv2.Rodrigues(all_best_results[f"Camera0_{i}"]['R'])
        rotation = rotation.flatten().tolist()
        translation = all_best_results[f"Camera0_{i}"]['t'].flatten().tolist()
    
    calib_data[camera_key] = {
        "name": camera_key,
        "size": [3840.0,2160.0],  # 실제 카메라 이미지 크기로 변경 필요
        "matrix": all_best_results[f"Camera0_{i}"]['K2'].tolist(),
        "distortions": [0.0, 0.0, 0.0, 0.0],  # 실제 distortion 값으로 변경 필요
        "rotation": rotation,
        "translation": translation,
        "fisheye": False
    }

# calib.toml 파일로 저장
with open(calib_file, "w") as f:
    toml.dump(calib_data, f)
