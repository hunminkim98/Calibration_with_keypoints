import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random
import os
import json
import pprint
from keypoints_confidence_multi import extract_paired_keypoints_with_reference
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from write_to_toml import write_to_toml

start_time = time.time()
# Constants for initial intrinsic matrix ( Factory setting in the paper but im using calibrate app in Matlab or OpenCV )
## It would be changed input data from Pose2Sim intrinsic calibration
image_size = [3840.0, 2160.0]  # image size
u0 = image_size[0] / 2  # principal point u0
v0 = image_size[1] / 2  # principal point v0




K1 = np.array([
    [ 1824.6097978600892, 0.0, u0],
    [ 0.0, 1826.6675222017589, v0],
    [ 0.0, 0.0, 1.0]
])

K2 = np.array([
    [ 1824.6097978600892, 0.0, u0],
    [ 0.0, 1826.6675222017589, v0],
    [ 0.0, 0.0, 1.0]
])

K3 = np.array([
    [ 1824.6097978600892, 0.0, u0],
    [ 0.0, 1826.6675222017589, v0],
    [ 0.0, 0.0, 1.0]
])

K4 = np.array([
    [ 1824.6097978600892, 0.0, u0],
    [ 0.0, 1826.6675222017589, v0],
    [ 0.0, 0.0, 1.0]
])

Ks = [K1, K2, K3, K4]


###################### Data Processing ############################

# camera directorie
# camera_directory = ['N_key_calib/json1','N_key_calib/json2', 'N_key_calib/json3', 'N_key_calib/json4']
camera_directory = ['pose/cal_json1','pose/cal_json2', 'pose/cal_json3', 'pose/cal_json4']
confidence_threshold = 0.8 # confidence threshold for keypoints pair extraction

# Call the function to extract paired keypoints

# paired_keypoints_list = extract_paired_keypoints_with_reference(ref_cam_dir, other_cam_dirs, confidence_threshold)
# print(f"type of paired_keypoints_list : {type(paired_keypoints_list)}")

def load_json_files(cam_dirs):
    """
    Load all JSON files from the given directories.

    Args:
    - cam_dirs: List of directories containing JSON files for cameras.

    Returns:
    - all_cam_data: A list containing data loaded from JSON files for each camera.
    """
    all_cam_data = []
    for cam_dir in cam_dirs:
        cam_files = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])
        cam_data = []
        for cam_file in cam_files:
            with open(cam_file, 'r') as file:
                data = json.load(file)
                cam_data.append(data)
        all_cam_data.append(cam_data)
    return all_cam_data



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
    other_cameras_keypoints = {}
    for i, camera_pair in enumerate(paired_keypoints_list):
        other_camera_index = i # camera index 
        other_cameras_keypoints[other_camera_index] = []

        for frame in camera_pair:
            frame_keypoints_other_camera = []
            for keypoints_pair in frame:
                if len(keypoints_pair) == 2:
                    frame_keypoints_other_camera.append(keypoints_pair[1])
                    
            other_cameras_keypoints[other_camera_index].append(frame_keypoints_other_camera)

            
    return other_cameras_keypoints


def create_paired_inlier(inliers1, inliers2):
    """
    Creates a list of paired inliers.

    Args:
        inliers1 (numpy.ndarray): Array of inlier points from camera 1.
        inliers2 (numpy.ndarray): Array of inlier points from camera 2.

    Returns:
        list of tuples: Each tuple contains paired points (tuples), 
                        where each sub-tuple is a point (x, y) from camera 1 and camera 2 respectively.
    """
    paired_inliers = [((p1[0], p1[1]), (p2[0], p2[1])) for p1, p2 in zip(inliers1, inliers2)]
    return paired_inliers


###################### Data Processing ############################

###################### Function of Extrinsics parameters optimisation ############################

def compute_fundamental_matrix(paired_keypoints_list):
    """
    Compute the fundamental matrix from paired keypoints and return inlier keypoints.

    This function takes a list of paired keypoints and computes the fundamental matrix using the RANSAC algorithm.
    It also filters out outliers based on the RANSAC result.

    Args:
        paired_keypoints_list (list): A list of tuples, where each tuple contains two arrays of keypoints, one for each image.

    Returns:
        numpy.ndarray: The computed fundamental matrix.
        numpy.ndarray: Points from the first image that are considered inliers.
        numpy.ndarray: Points from the second image that are considered inliers.
    """
    points1, points2 = unpack_keypoints(paired_keypoints_list)
    
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)

    # Compute the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # Filter points based on the mask
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]

    return F, inliers1, inliers2



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

def recover_pose_from_essential_matrix(E, points1_inliers, points2_inliers, K):
    """
    Recover the camera pose from the Essential matrix using inliers.

    Parameters:
    E (numpy.ndarray): The Essential matrix.
    points1_inliers (numpy.ndarray): The inlier points from the first image.
    points2_inliers (numpy.ndarray): The inlier points from the second image.
    K (numpy.ndarray): The camera intrinsic matrix (assuming the same for both cameras).

    Returns:
    numpy.ndarray, numpy.ndarray: The rotation matrix (R) and the translation vector (t).
    """
    # Ensure points are in the correct shape and type
    points1_inliers = points1_inliers.astype(np.float32)
    points2_inliers = points2_inliers.astype(np.float32)

    # Recovering the pose
    _, R, t, mask = cv2.recoverPose(E, points1_inliers, points2_inliers, K)

    return R, t, mask

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


# def triangulate_points(paired_keypoints_list, P1, P2):
#     """
#     Triangulates a list of paired keypoints using the given camera projection matrices.

#     Args:
#         paired_keypoints_list (list): List of paired keypoints for each frame.
#         P1 (array-like): Camera projection matrix for the reference camera.
#         P2 (array-like): Camera projection matrix for the other camera.

#     Returns:
#         list: List of 3D points corresponding to the triangulated keypoints for each frame.
#     """
#     points_3d = []

#     for frame in paired_keypoints_list:
#         points_3d_frame = []

#         for point in frame:
#             x1, y1 = point[0]
#             x2, y2 = point[1]

#             # Triangulate the point
#             point_3d = cv2.triangulatePoints(P1, P2, (x1, y1), (x2, y2))
#             point_3d /= point_3d[3] # normalize

#             points_3d_frame.append(point_3d[:3]) # remove homogeneous coordinate

#         points_3d.append(points_3d_frame)

#     return points_3d


def triangulate_points(paired_keypoints_list, P1, P2):
    """
    Triangulates a list of paired keypoints using the given camera projection matrices.

    Args:
        paired_keypoints_list (list): List of paired keypoints, where each item is a tuple containing 
                                      two sets of coordinates for the same keypoint observed in both cameras.
        P1 (array-like): Camera projection matrix for the reference camera.
        P2 (array-like): Camera projection matrix for the other camera.

    Returns:
        list: List of 3D points corresponding to the triangulated keypoints.
    """
    points_3d = []

    for keypoint_pair in paired_keypoints_list:
        (x1, y1), (x2, y2) = keypoint_pair

        # Convert coordinates to homogeneous format for triangulation
        point_3d_homogeneous = cv2.triangulatePoints(P1, P2, np.array([[x1], [y1]], dtype=np.float64), np.array([[x2], [y2]], dtype=np.float64))

        # Normalize to convert to non-homogeneous 3D coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        points_3d.append(point_3d)

    return points_3d



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

def compute_reprojection_error(precomputed_points_3d, keypoints_detected, P1, P2):
    """
    Computes the reprojection error for a set of paired keypoints using the given projection matrices
    and precomputed 3D points.

    Args:
        precomputed_points_3d (list): List of precomputed 3D points as NumPy arrays.
        keypoints_detected (list): List of paired keypoints, each represented as a tuple (2D point in camera 1, 2D point in camera 2).
        P1 (array-like): Camera projection matrix for the reference camera.
        P2 (array-like): Camera projection matrix for the other camera.

    Returns:
        float: The mean reprojection error over all keypoints.
    """
    total_error = 0
    total_points = 0

    # Ensure the length of 3D points matches the 2D keypoints
    assert len(precomputed_points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

    # Process each pair of 3D point and 2D keypoints
    for point_3d, (point1, point2) in zip(precomputed_points_3d, keypoints_detected):
        # Convert 3D point to homogeneous coordinates
        point_3d_homogeneous = np.append(point_3d.flatten(), 1)

        # Reproject the 3D point to the 2D image plane for both cameras
        point1_reprojected = P1 @ point_3d_homogeneous
        point1_reprojected /= point1_reprojected[2]

        point2_reprojected = P2 @ point_3d_homogeneous
        point2_reprojected /= point2_reprojected[2]

        # Compute reprojection errors for each camera's reprojected point
        error1 = np.linalg.norm(point1_reprojected[:2] - np.array(point1))
        error2 = np.linalg.norm(point2_reprojected[:2] - np.array(point2))

        total_error += error1 + error2
        total_points += 2

    mean_error = total_error / total_points if total_points > 0 else 0
    return mean_error


###################### Function of Intrinsics parameters optimisation ############################



def vectorize_params_for_intrinsic_loss(points_3d, keypoints_detected, R, t):

    # Initialize arrays
    u_detected = []
    v_detected = []
    Xc = []
    Yc = []
    Zc = []
    transformation_matrix = np.hstack((R, t.reshape(-1, 1)))  # transformation matrix
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # homogeneous transformation matrix

    # Make sure the number of 3D points matches the 2D keypoints
    assert len(points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

    # print("transformation matrix shape", transformation_matrix.shape)  # transformation_matrix.shape
    for point_3d, detected_point in zip(points_3d, keypoints_detected):
        if not isinstance(detected_point, (list, tuple, np.ndarray)) or len(detected_point) != 2:
            continue
        # detected point = (u, v)
        u, v = detected_point
        u_detected.append(u)
        v_detected.append(v)

        # world to camera transformation
        point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates
        point_camera = transformation_matrix.dot(point_3d_homogeneous)
        X, Y, Z = point_camera[:3]
        Xc.append(X)
        Yc.append(Y)
        Zc.append(Z)

    return np.array(u_detected), np.array(v_detected), np.array(Xc), np.array(Yc), np.array(Zc)

def compute_intrinsics_optimization_loss(x, u_detected, v_detected, Xc, Yc, Zc, u0, v0):
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
    f_x, f_y = x  # Intrinsic parameters to optimize
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)

    valid_keypoints_count = Xc.shape[0]
    
    loss = np.abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + np.abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
    total_loss = np.sum(loss)
   
    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0
    print(f"mean_loss of intrinsic : {mean_loss}")
    return mean_loss

def optimize_intrinsic_parameters(points_3d, keypoints_detected, K, R, t, tolerance_list, u0, v0):
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
    x0 = np.array([K[0, 0], K[1,1]])
    u_detected, v_detected, Xc, Yc, Zc= vectorize_params_for_intrinsic_loss(points_3d, keypoints_detected, R, t)
    # Optimize the intrinsic parameters using the least squares method
    result = least_squares(compute_intrinsics_optimization_loss, x0, args=(u_detected, v_detected, Xc, Yc, Zc, u0, v0), x_scale='jac', verbose=1, method='trf', loss= 'huber', diff_step=tolerance_list['diff_step'], tr_solver='lsmr', ftol=tolerance_list['ftol'], max_nfev=tolerance_list['max_nfev'], xtol=tolerance_list['xtol'], gtol=tolerance_list['gtol'])

    # Create the optimized intrinsic matrix
    K_optimized = np.array([[result.x[0], 0, u0], [0, result.x[1], v0], [0, 0, 1]])

    return K_optimized

###################### Function of Intrinsics parameters optimisation ############################



###################### Reference Camera Selection #########################

camera_Rt_list = [{} for _ in camera_directory]
valid_ref_cam_idx = []
# total_keypoints_for_valid_ref_cam = []
average_scores_for_valid_ref_cam = []
temp_paired_keypoints_global_list = []
all_cam_data = load_json_files(camera_directory)
# inliers_pair_list = []

# Find the minimum number of frames across all cameras
min_frames = min(len(cam_data) for cam_data in all_cam_data)
for i in range(len(all_cam_data)):
    all_cam_data[i] = all_cam_data[i][:min_frames]

print("Finding the best reference camera.... \n\n\n")
for i, cam_dir in enumerate(camera_directory):
    no_R_t_solutions_flag = False
    zero_keypoints_flag = False
    total_inliers_for_all_camera_pairing = 0
    total_score = 0
    print(f"Current assigned reference camera: {i}")
    paired_keypoints_list = extract_paired_keypoints_with_reference(all_cam_data, i, confidence_threshold)

    temp_idx = -1
    inliers_per_camera_pair = []
    for j, K in enumerate(Ks):
        if j == i:
            continue
        temp_idx += 1
        paired_keypoints = paired_keypoints_list[temp_idx]
        keypoints_for_current_camera_pairing = sum(len(frame_keypoints) for frame_keypoints in paired_keypoints)

        if keypoints_for_current_camera_pairing == 0:
            zero_keypoints_flag = True
            break

        F, inliers1, inliers2 = compute_fundamental_matrix(paired_keypoints)
        inliers_count = len(inliers1)
        print(f"Camera {i} inliers count: {inliers_count}")
        total_inliers_for_all_camera_pairing += inliers_count

        # paired_inliers = create_paired_inlier(inliers1, inliers2)  
        # inliers_per_camera_pair.append(paired_inliers) 

        E = compute_essential_matrix(F, Ks[i], K)
        R, t, mask = recover_pose_from_essential_matrix(E, inliers1, inliers2, K)

        if R is None or t is None:
            no_R_t_solutions_flag = True
            break

        P1 = cam_create_projection_matrix(Ks[i], np.eye(3), np.zeros((3, 1)))
        P2 = cam_create_projection_matrix(K, R, t)
        inliers_pair = create_paired_inlier(inliers1, inliers2)
        points_3d_int = triangulate_points(inliers_pair, P1, P2)
        mean_error = compute_reprojection_error(points_3d_int, inliers_pair, P1, P2)
        score = inliers_count / mean_error
        total_score += score
        print(f"Camera {j} relative to temporary Reference Camera {i}: Mean reprojection error: {mean_error}")
        camera_Rt_list[i][j] = [R, t]

    if not no_R_t_solutions_flag and not zero_keypoints_flag:
        average_score = total_score/(len(camera_directory)-1)
        valid_ref_cam_idx.append(i)
        # total_keypoints_for_valid_ref_cam.append(total_inliers_for_all_camera_pairing)
        average_scores_for_valid_ref_cam.append(average_score)
        temp_paired_keypoints_global_list.append(paired_keypoints_list)
        # inliers_pair_list.append(inliers_per_camera_pair)

# index_of_best_ref_cam = total_keypoints_for_valid_ref_cam.index(max(total_keypoints_for_valid_ref_cam))
index_of_best_ref_cam = average_scores_for_valid_ref_cam.index(max(average_scores_for_valid_ref_cam))
final_idx_of_ref_cam = valid_ref_cam_idx[index_of_best_ref_cam]
paired_keypoints_list = temp_paired_keypoints_global_list[index_of_best_ref_cam]
# inliers_per_camera_pair = inliers_pair_list[index_of_best_ref_cam]

print(f"Best reference camera: {final_idx_of_ref_cam}")
input("Press any key to continue...")
final_camera_Rt = camera_Rt_list[final_idx_of_ref_cam]
print("Final camera R, t:", final_camera_Rt)


###################### Intrinsic Optimization ############################

all_best_results = {}
iterations = 1
optimization_results = {}
Fix_K = Ks[final_idx_of_ref_cam]
inliers_pair_list = []
inlier2_list = []

tolerance_list = {
    'ftol': 1e-3,
    'xtol': 1e-4,
    'gtol': 1e-3,
    'max_nfev': 50,
    'diff_step': 1e-3
}

temp_idx = -1
for j in range(len(Ks)):
    if j == final_idx_of_ref_cam:
        continue
    temp_idx += 1
    # if j == 0 or j == 1:
    #     paired_keypoints = paired_keypoints_list[0]
    # else:
    paired_keypoints = paired_keypoints_list[temp_idx]
    
    _, inliers1, inliers2 = compute_fundamental_matrix(paired_keypoints)
    inliers_pair = create_paired_inlier(inliers1, inliers2)
    inliers_pair_list.append(inliers_pair)
    inlier2_list.append(inliers2)

    camera_pair_key = f"Camera{final_idx_of_ref_cam}_{j}"
    print(f"Optimizing for pair: {camera_pair_key}")
    K_optimized = Ks[j]
    R_optimized, t_optimized = final_camera_Rt[j]

    optimization_results[camera_pair_key] = {'K1': [], 'K2': [], 'R': [], 't': [], 'errors': []}

    for iteration in range(iterations):
        print(f"---Iteration {iteration + 1} for {camera_pair_key} ---")
        OPT_K = K_optimized

        P1 = cam_create_projection_matrix(Fix_K, np.eye(3), np.zeros((3, 1)))
        P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)

        if R_optimized is None or t_optimized is None:
            continue

        points_3d_int = triangulate_points(inliers_pair, P1, P2)
        OPT_K_optimized = optimize_intrinsic_parameters(points_3d_int, inliers2, OPT_K, R_optimized, t_optimized, tolerance_list, u0, v0)
        OPT_K = OPT_K_optimized

        P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)
        int_error = compute_reprojection_error(points_3d_int, inliers_pair, P1, P2)

        print(f"Camera pair {camera_pair_key} Second part of iteration {iteration + 1}: Mean reprojection error: {int_error}")

        optimization_results[camera_pair_key]['K1'].append(Fix_K)
        optimization_results[camera_pair_key]['K2'].append(OPT_K)
        optimization_results[camera_pair_key]['R'].append(R_optimized)
        optimization_results[camera_pair_key]['t'].append(t_optimized)
        optimization_results[camera_pair_key]['errors'].append(int_error)

        K_optimized = OPT_K

    if optimization_results[camera_pair_key]['errors']:
        min_error_for_pair = min(optimization_results[camera_pair_key]['errors'])
        index_of_min_error = optimization_results[camera_pair_key]['errors'].index(min_error_for_pair)
        best_K1 = optimization_results[camera_pair_key]['K1'][index_of_min_error]
        best_K2 = optimization_results[camera_pair_key]['K2'][index_of_min_error]
        best_R = optimization_results[camera_pair_key]['R'][index_of_min_error]
        best_t = optimization_results[camera_pair_key]['t'][index_of_min_error]

        all_best_results[camera_pair_key] = {
            'K1': best_K1,
            'K2': best_K2,
            'R': best_R,
            't': best_t,
            'error': min_error_for_pair
        }

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
def vectorize_params_for_extrinsic_loss(points_3d, points_2d):
    u_detected_list = []
    v_detected_list = []
    point_3d_homogeneous_list = []

    # Convert to numpy arrays
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # Check if points_3d and points_2d are not empty
    if points_3d.size == 0 or points_2d.size == 0:
        print("Warning: Empty points_3d or points_2d input.")
        return np.array(u_detected_list), np.array(v_detected_list), np.array(point_3d_homogeneous_list)

    # Flatten the points_3d array to match the expected shape (N, 3)
    points_3d_flat = points_3d.reshape(-1, 3)

    for point_3d, detected_point in zip(points_3d_flat, points_2d):
        if len(detected_point) != 2:
            continue

        u_detected, v_detected = detected_point

        # World to camera transformation
        point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates

        u_detected_list.append(u_detected)
        v_detected_list.append(v_detected)
        point_3d_homogeneous_list.append(point_3d_homogeneous)

    return np.array(u_detected_list), np.array(v_detected_list), np.array(point_3d_homogeneous_list)


def rotation_matrix_to_quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    return quaternion

def quaternion_to_rotation_matrix(quaternion):
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def compute_extrinsics_optimization_loss(x, ext_K, u_detected, v_detected, point_3d_homogeneous):
    """
    Computes the loss for the intrinsic parameters optimization.

    Args:
    - x: Extrinsic parameters to optimize (quaternion + translation).
    - points_3d: List of 3D points (triangulated human body joints).
    - keypoints_detected: Original detected 2D keypoints.
    - ext_K: Intrinsic parameters matrix.

    Returns:
    - The mean loss for the intrinsic parameters optimization.
    """
    # 쿼터니언과 번역 벡터 추출
    quaternion = x[:4]
    t = x[4:].reshape(3, 1)
    
    # 쿼터니언을 회전 행렬로 변환
    R_optimized = quaternion_to_rotation_matrix(quaternion)
    
    # 변환 행렬 생성
    transformation_matrix = np.hstack((R_optimized, t))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    point_camera = np.dot(point_3d_homogeneous, transformation_matrix.T)
    Xc, Yc, Zc, _ = point_camera.T
    valid_keypoints_count = Xc.shape[0]
    
    f_x, f_y, u0, v0 = ext_K[0, 0], ext_K[1, 1], ext_K[0, 2], ext_K[1, 2]
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)

    loss = np.abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + np.abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
    total_loss = np.sum(loss)
   
    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0

    return mean_loss


def optimize_extrinsic_parameters(points_3d, other_cameras_keypoints, ext_K, ext_R, ext_t, tolerance_list):
    """
    Optimizes the extrinsic parameters using the given 3D points and detected keypoints.

    Args:
    - points_3d: List of 3D points (triangulated human body joints).
    - other_cameras_keypoints: Original detected 2D keypoints for the other cameras.
    - ext_K: Intrinsic parameters matrix.
    - ext_R: Rotation matrix.
    - ext_t: Translation vector.

    Returns:
    - The optimized quaternion and t vector.
    """
    # 회전 행렬을 쿼터니언으로 변환
    initial_quaternion = rotation_matrix_to_quaternion(ext_R)
    initial_params = np.hstack((initial_quaternion, ext_t.flatten()))

    u_detected, v_detected, point_3d_homogeneous = vectorize_params_for_extrinsic_loss(points_3d, other_cameras_keypoints)
    
    # Optimize the extrinsic parameters using the least squares method
    result = least_squares(compute_extrinsics_optimization_loss, initial_params, args=(ext_K, u_detected, v_detected, point_3d_homogeneous), x_scale='jac', verbose=1, method='trf', loss='huber', diff_step=tolerance_list['diff_step'], tr_solver='lsmr', ftol=tolerance_list['ftol'], max_nfev=tolerance_list['max_nfev'], xtol=tolerance_list['xtol'], gtol=tolerance_list['gtol'])

    optimized_params = result.x
    optimized_quaternion = optimized_params[:4]
    optimized_t = optimized_params[4:]

    # 쿼터니언을 회전 행렬로 변환
    optimized_R = quaternion_to_rotation_matrix(optimized_quaternion)

    return optimized_R, optimized_t.reshape(3, 1)


########################################
####### Multi-camera calibration #######
########################################

def update_tolerance(n, tolerance_list):
    # Update the tolerance values based on the iteration number
    factor = 0.1 ** (n / N)
    tolerance_list['ftol'] = max(tolerance_list['ftol'] * factor, 1e-6)
    tolerance_list['xtol'] = max(tolerance_list['xtol'] * factor, 1e-6)
    tolerance_list['gtol'] = max(tolerance_list['gtol'] * factor, 1e-6)
    tolerance_list['diff_step'] = max(tolerance_list['diff_step'] * factor, 1e-8)
    return tolerance_list


N = 30 # how many times to run the optimization

tolerance_list = {
    'ftol': 1e-2,
    'xtol': 1e-2,
    'gtol': 1e-2,
    'max_nfev': 100,
    'diff_step': 1e-2
}

temp_idx = -1
for i, K in enumerate(Ks):
    if i == final_idx_of_ref_cam:  # skip the reference camera
        continue
    temp_idx += 1
    # if i == 0 or i == 1:
    #     inliers_pair = inliers_pair_list[0]
    # else:

    inliers_pair = inliers_pair_list[temp_idx]
    other_keypoints_detected = inlier2_list[temp_idx]

    print(f"calibrating camera {i}...")

    # import the best results for each camera pair
    ext_K = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['K2'] 
    ext_R = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['R'] # fixed R matrix
    ext_t = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['t']
    int_K_best  = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['K1']
    ref_t = np.array([[0], [0], [0]]) # reference camera t vector


    # projection matrix
    P1 = cam_create_projection_matrix(Ks[final_idx_of_ref_cam], np.eye(3), ref_t)
    P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)

    # triangulate points
    points_3d = triangulate_points(inliers_pair, P1, P2) # initial 3D points
    before_optimization_error = compute_reprojection_error(points_3d, inliers_pair, P1, P2)
    print(f"camera {i} before optimization error: {before_optimization_error}")

    # Entrinsic and intrinsic parameter joint optimization
    for n in range(N):
        # Update the tolerance values
        tolerance_list = update_tolerance(n, tolerance_list)

        # extrinsic parameter optimization
        print(f"before optimization t vector: {ext_t}")
        optimized_R, optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tolerance_list) # optimize extrinsic parameters
        ext_R, ext_t = optimized_R, optimized_t # update R and t vectors
        print(f"{n + 1}th optimized R matrix: {ext_R}")
        print(f"{n + 1}th optimized t vector: {ext_t}")

        N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
        ex_reprojection_error = compute_reprojection_error(points_3d, inliers_pair, P1, N_P2) # calculate the mean reprojection error
        print(f"{n + 1}th error in extrinsic optimization = {ex_reprojection_error}")

        # intrinsic parameter optimization
        points_3d = triangulate_points(inliers_pair, P1, N_P2) # update 3D points after extrinsic optimization
        ext_K_optimized = optimize_intrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tolerance_list, u0, v0) # optimize intrinsic parameters
        ext_K = ext_K_optimized # update intrinsic parameters
        print(f"{n + 1}th optimized K matrix: {ext_K}")

        N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
        in_reprojection_error = compute_reprojection_error(points_3d, inliers_pair, P1, N_P2) # calculate the mean reprojection error
        print(f"{n + 1}th error in intrinsic optimization = {in_reprojection_error}")
        points_3d = triangulate_points(inliers_pair, P1, N_P2) # update 3D points after intrinsic optimization

    # save result after optimization
    all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['t'] = ext_t
    all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['K2'] = ext_K

    # ext_R matrix to rod vector
    ext_R_rod, _ = cv2.Rodrigues(ext_R)
    print(f"camera {i} R : {ext_R_rod}")

# print optimized results
for pair_key, results in all_best_results.items():
    print(f"Best results for {pair_key}:")
    print(f"- K1: {results['K1']}")
    print(f"- K2: {results['K2']}")
    print(f"- R: {results['R']}")
    print(f"- t: {results['t']}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Calibration took {elapsed_time} seconds to finish.")

write_to_toml(all_best_results,[0.0, 0.0, 0.0], image_size)