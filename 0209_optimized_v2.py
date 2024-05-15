import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random
import os
import json
import pprint
from keypoints_confidence_multi_v2 import extract_paired_keypoints_with_reference
from scipy.optimize import least_squares
from scipy.optimize import minimize
from write_to_toml_v2 import write_to_toml

start_time = time.time()
# Constants for initial intrinsic matrix ( Factory setting in the paper but im using calibrate app in Matlab or OpenCV )
## It would be changed input data from Pose2Sim intrinsic calibration
image_size = (3840.0, 2160.0)  # image size
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
total_keypoints_for_all_camera = 0

###################### Data Processing ############################

# camera directorie
# ref_cam_dir = '/json3' # reference camera directory
# other_cam_dirs = ['pose/json2','pose/json1'] # other camera directories
camera_directory = [r'C:\Users\82102\Downloads\TalkFile_Calibration_with_keypoints.zip\cal_json1', r'C:\Users\82102\Downloads\TalkFile_Calibration_with_keypoints.zip\cal_json2', 
                    r'C:\Users\82102\Downloads\TalkFile_Calibration_with_keypoints.zip\cal_json3', r'C:\Users\82102\Downloads\TalkFile_Calibration_with_keypoints.zip\cal_json4']
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

def recover_pose_from_essential_matrix(E, points1, points2, K):
    """
    Recovers the pose (rotation and translation) from the essential matrix.

    Args:
        E (numpy.ndarray): The essential matrix.
        points1 (numpy.ndarray): Array of 2D points in the first image.
        points2 (numpy.ndarray): Array of corresponding 2D points in the second image.
        K (numpy.ndarray): The camera matrix.

    Returns:
        R (numpy.ndarray): The rotation matrix.
        t (numpy.ndarray): The translation vector.
        mask (numpy.ndarray): The mask indicating inliers.

    """
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)

    _, R, t, mask = cv2.recoverPose(E, points1, points2, K, None)
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
            point_3d /= point_3d[3] # normalize

            points_3d_frame.append(point_3d[:3]) # remove homogeneous coordinate

        points_3d.append(points_3d_frame)

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


# Not working. Need to fix
# def compute_reprojection_error(keypoints_detected, precomputed_points_3d, P1, P2):
#     """
#     Computes the reprojection error for a set of paired keypoints using the given projection matrices
#     and precomputed 3D points.

#     Args:
#         precomputed_points_3d (list): List of precomputed 3D points as NumPy arrays.
#         keypoints_detected (list): List of paired keypoints, each represented as a tuple (2D point in camera 1, 2D point in camera 2).
#         P1 (array-like): Camera projection matrix for the reference camera.
#         P2 (array-like): Camera projection matrix for the other camera.

#     Returns:
#         float: The mean reprojection error over all keypoints.
#     """
#     total_error = 0
#     total_points = 0

#     assert len(precomputed_points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

#     for i, (points_3d_set, frame_keypoints) in enumerate(zip(precomputed_points_3d, keypoints_detected)):
#         points_3d_set = np.array(points_3d_set)

#         for points_3d in points_3d_set:
#             points_3d = points_3d.reshape(3, 1)  # Ensure (3,) shape is converted to (3, 1)
#             # Convert 3D points to homogeneous coordinates
#             points_3d_homogeneous = np.vstack([points_3d, np.ones((1, 1))])

#             for point1, point2 in frame_keypoints:
#                 # Reproject the 3D point to the 2D image plane for both cameras
#                 point1_reprojected = P1 @ points_3d_homogeneous
#                 point1_reprojected /= point1_reprojected[2]

#                 point2_reprojected = P2 @ points_3d_homogeneous
#                 point2_reprojected /= point2_reprojected[2]

#                 # Compute reprojection errors for each camera's reprojected point
#                 error1 = np.linalg.norm(point1_reprojected[:2].flatten() - np.array(point1))
#                 error2 = np.linalg.norm(point2_reprojected[:2].flatten() - np.array(point2))

#                 total_error += error1 + error2
#                 total_points += 2

#     mean_error = total_error / total_points if total_points > 0 else 0
#     return mean_error

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
    # print("transformation matrix shape", transformation_matrix.shape)  # transformation_matrix.shape
    for frame_points_3d, frame_keypoints_detected in zip(points_3d, keypoints_detected):
        for point_3d, detected_point in zip(frame_points_3d, frame_keypoints_detected):
            if not isinstance(detected_point, tuple) or len(detected_point) != 2:
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
total_keypoints_for_valid_ref_cam = []
temp_paired_keypoints_global_list = []
all_cam_data = load_json_files(camera_directory)

# Find the minimum number of frames across all cameras
min_frames = min(len(cam_data) for cam_data in all_cam_data)
for i in range(len(all_cam_data)):
    all_cam_data[i] = all_cam_data[i][:min_frames]

print("Finding the best reference camera.... \n\n\n")
for i, cam_dir in enumerate(camera_directory):
    no_R_t_solutions_flag = False
    zero_keypoints_flag = False
    total_keypoints_for_all_camera_pairing = 0
    print(f"Current assigned reference camera: {i}")
    paired_keypoints_list = extract_paired_keypoints_with_reference(all_cam_data, i, confidence_threshold)

    temp_idx = -1
    for j, K in enumerate(Ks):
        if j == i:
            continue
        temp_idx += 1
        paired_keypoints = paired_keypoints_list[temp_idx]
        keypoints_for_current_camera_pairing = sum(len(frame_keypoints) for frame_keypoints in paired_keypoints)

        if keypoints_for_current_camera_pairing == 0:
            zero_keypoints_flag = True
            break
        total_keypoints_for_all_camera_pairing += keypoints_for_current_camera_pairing

        F = compute_fundamental_matrix(paired_keypoints)
        E = compute_essential_matrix(F, Ks[i], K)
        points1, points2 = unpack_keypoints(paired_keypoints)
        R, t, mask = recover_pose_from_essential_matrix(E, points1, points2, K)

        if R is None or t is None:
            no_R_t_solutions_flag = True
            break

        camera_Rt_list[i][j] = [R, t]

    if not no_R_t_solutions_flag and not zero_keypoints_flag:
        valid_ref_cam_idx.append(i)
        total_keypoints_for_valid_ref_cam.append(total_keypoints_for_all_camera_pairing)
        temp_paired_keypoints_global_list.append(paired_keypoints_list)

index_of_best_ref_cam = total_keypoints_for_valid_ref_cam.index(max(total_keypoints_for_valid_ref_cam))
final_idx_of_ref_cam = valid_ref_cam_idx[index_of_best_ref_cam]
paired_keypoints_list = temp_paired_keypoints_global_list[index_of_best_ref_cam]

print(f"Best reference camera: {final_idx_of_ref_cam}")

final_camera_Rt = camera_Rt_list[final_idx_of_ref_cam]
print("Final camera R, t:", final_camera_Rt)


###################### Intrinsic Optimization ############################

all_best_results = {}
iterations = 5
optimization_results = {}
Fix_K = Ks[final_idx_of_ref_cam]

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
    if j == 0 or j == 1:
        paired_keypoints = paired_keypoints_list[0]
    paired_keypoints = paired_keypoints_list[temp_idx]
    camera_pair_key = f"Camera{final_idx_of_ref_cam}_{j}"
    print(f"Optimizing for pair: {camera_pair_key}")
    K_optimized = Ks[j]
    R_optimized, t_optimized = final_camera_Rt[j]
    other_cameras_keypoints = extract_individual_camera_keypoints(paired_keypoints_list)

    optimization_results[camera_pair_key] = {'K1': [], 'K2': [], 'R': [], 't': [], 'errors': []}

    for iteration in range(iterations):
        print(f"---Iteration {iteration + 1} for {camera_pair_key} ---")
        OPT_K = K_optimized

        P1 = cam_create_projection_matrix(Fix_K, np.eye(3), np.zeros((3, 1)))
        P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)

        if R_optimized is None or t_optimized is None:
            continue

        points_3d_int = triangulate_points(paired_keypoints, P1, P2)
        OPT_K_optimized = optimize_intrinsic_parameters(points_3d_int, other_cameras_keypoints[temp_idx], OPT_K, R_optimized, t_optimized, tolerance_list, u0, v0)
        OPT_K = OPT_K_optimized

        P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)
        int_error = compute_reprojection_error(paired_keypoints, P1, P2)

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
    
    for frame_points_3d, frame_keypoints_detected in zip(points_3d, points_2d):
        for point_3d, detected_point in zip(frame_points_3d, frame_keypoints_detected):
            if not isinstance(detected_point, tuple) or len(detected_point) != 2:
                continue
                
            u_detected, v_detected = detected_point
            
            # world to camera transformation
            point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates
            
            u_detected_list.append(u_detected)
            v_detected_list.append(v_detected)
            point_3d_homogeneous_list.append(point_3d_homogeneous)
            
    return np.array(u_detected_list), np.array(v_detected_list), np.array(point_3d_homogeneous_list)


def compute_extrinsics_optimization_loss(x, ext_K, u_detected, v_detected, point_3d_homogeneous):
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
    f_x, f_y, u0, v0 = ext_K[0, 0], ext_K[1, 1], ext_K[0, 2], ext_K[1, 2]
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)
    obj_t = x
    transformation_matrix = np.hstack((ext_R, obj_t.reshape(-1, 1)))  # transformation matrix
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # homogeneous transformation matrix

    # transformation_matrix is a 2D array of shape (4, 4) and 
    #point_3d_homogeneous is a 2D array of shape (N, 4), where N is the number of points, we can do dot product
    point_camera = np.dot(point_3d_homogeneous, transformation_matrix.T)

    # point_camera is a 2D array of shape (N, 4). Xc, Yc, Zc are all 1D np arrays
    Xc, Yc, Zc, _ = point_camera.T
    valid_keypoints_count = Xc.shape[0]
    
    loss = np.abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + np.abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
    total_loss = np.sum(loss)
   
    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0
    print(f"mean_loss of extrinsic : {mean_loss}")
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
    - The optimized t vector.
    """
    # Create the initial guess for the extrinsic parameters (|T|) using the t vector magnitude
    # x0 = np.array([np.linalg.norm(ext_t)])
    x0 = ext_t.flatten()
    print(f"Initial x0: {x0}")
    u_detected, v_detected, point_3d_homogeneous= vectorize_params_for_extrinsic_loss(points_3d, other_cameras_keypoints)
    
    # Optimize the extrinsic parameters using the least squares method
    result = least_squares(compute_extrinsics_optimization_loss, x0, args=(ext_K, u_detected, v_detected, point_3d_homogeneous), x_scale='jac', verbose=1, method='trf', loss= 'huber', diff_step=tolerance_list['diff_step'], tr_solver='lsmr', ftol=tolerance_list['ftol'], max_nfev=tolerance_list['max_nfev'], xtol=tolerance_list['xtol'], gtol=tolerance_list['gtol'])



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

def update_tolerance(n, tolerance_list):
    # Update the tolerance values based on the iteration number
    factor = 0.1 ** (n / N)
    tolerance_list['ftol'] = max(tolerance_list['ftol'] * factor, 1e-6)
    tolerance_list['xtol'] = max(tolerance_list['xtol'] * factor, 1e-6)
    tolerance_list['gtol'] = max(tolerance_list['gtol'] * factor, 1e-6)
    tolerance_list['diff_step'] = max(tolerance_list['diff_step'] * factor, 1e-8)
    return tolerance_list


N = 30 # how many times to run the optimization

# tolerance_list = {
#     'ftol': 1e-6,
#     'xtol': 1e-6,
#     'gtol': 1e-6,
#     'max_nfev': 150,
#     'diff_step': 1e-3
# }

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
    print(f"calibrating camera {i}...")

    # import the best results for each camera pair
    ext_K = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['K2'] 
    ext_R = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['R'] # fixed R matrix
    ext_t = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['t']
    int_K_best  = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['K1']
    ref_t = np.array([[0], [0], [0]]) # reference camera t vector


    # projection matrix
    # TODO: Found a bug, all P1 should be reference, also why is Ks used here instead of the best value intrinsics
    # P1 = cam_create_projection_matrix(Ks[final_idx_of_ref_cam], np.eye(3), ref_t)
    P1 = cam_create_projection_matrix(Ks[final_idx_of_ref_cam], np.eye(3), ref_t)
    P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)
    # TODO: i-1 should be changed to i
    # triangulate points
    points_3d = triangulate_points(paired_keypoints_list[temp_idx], P1, P2) # initial 3D points
    before_optimization_error = compute_reprojection_error(paired_keypoints_list[temp_idx], P1, P2)
    print(f"camera {i} before optimization error: {before_optimization_error}")

    # keypoints for optimization
    other_keypoints_detected = other_cameras_keypoints[temp_idx] # use the keypoints for the other camera



    # Entrinsic and intrinsic parameter joint optimization
    for n in range(N):
        # Update the tolerance values
        tolerance_list = update_tolerance(n, tolerance_list)

        # extrinsic parameter optimization
        print(f"before optimization t vector: {ext_t}")
        optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tolerance_list) # optimize extrinsic parameters
        ext_t = optimized_t # update t vector
        print(f"{n + 1}th optimized t vector: {ext_t}")

        N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
        ex_reprojection_error = compute_reprojection_error(paired_keypoints_list[temp_idx], P1, N_P2) # calculate the mean reprojection error
        print(f"{n + 1}th error in extrinsic optimization = {ex_reprojection_error}")

        # intrinsic parameter optimization
        points_3d = triangulate_points(paired_keypoints_list[temp_idx], P1, N_P2) # update 3D points after extrinsic optimization
        ext_K_optimized = optimize_intrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tolerance_list, u0, v0) # optimize intrinsic parameters
        ext_K = ext_K_optimized # update intrinsic parameters
        print(f"{n + 1}th optimized K matrix: {ext_K}")

        N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
        in_reprojection_error = compute_reprojection_error(paired_keypoints_list[temp_idx], P1, N_P2) # calculate the mean reprojection error
        print(f"{n + 1}th error in intrinsic optimization = {in_reprojection_error}")
        points_3d = triangulate_points(paired_keypoints_list[temp_idx], P1, N_P2) # update 3D points after intrinsic optimization

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

write_to_toml(all_best_results)