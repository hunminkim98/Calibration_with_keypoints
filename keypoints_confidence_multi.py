import os
import json

def extract_paired_keypoints_with_reference(ref_cam_dir, other_cam_dirs, confidence_threshold):
    """
    Extracts paired keypoints (x, y) with reference camera and each of the other cameras.

    Args:
    - ref_cam_dir: Directory containing JSON files for the reference camera.
    - other_cam_dirs: List of directories containing JSON files for other cameras.
    - confidence_threshold: Confidence value threshold for keypoints.

    Returns:
    - all_paired_keypoints: A list containing paired keypoints for each camera pair.
    """
    all_paired_keypoints = []

    # Load and sort JSON files from the reference camera directory
    ref_cam_files = sorted([os.path.join(ref_cam_dir, f) for f in os.listdir(ref_cam_dir) if f.endswith('.json')])

    for cam_dir in other_cam_dirs:
        # Load and sort JSON files from the current camera directory
        cam_files = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])

        paired_keypoints_list = []
        for ref_file, cam_file in zip(ref_cam_files, cam_files):
            with open(ref_file, 'r') as file1, open(cam_file, 'r') as file2:
                ref_data = json.load(file1)
                cam_data = json.load(file2)

                if ref_data['people'] and cam_data['people']:
                    ref_keypoints = ref_data['people'][0]['pose_keypoints_2d']
                    cam_keypoints = cam_data['people'][0]['pose_keypoints_2d']

                    # Extract keypoints with confidence
                    ref_keypoints_conf = [(ref_keypoints[i], ref_keypoints[i+1], ref_keypoints[i+2]) for i in range(0, len(ref_keypoints), 3)]
                    cam_keypoints_conf = [(cam_keypoints[i], cam_keypoints[i+1], cam_keypoints[i+2]) for i in range(0, len(cam_keypoints), 3)]

                    # Filter keypoints based on confidence threshold and pair them
                    paired_keypoints = [((kp1[0], kp1[1]), (kp2[0], kp2[1])) for kp1, kp2 in zip(ref_keypoints_conf, cam_keypoints_conf) if kp1[2] >= confidence_threshold and kp2[2] >= confidence_threshold]
                    # e.g. [((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...]

                    if paired_keypoints:
                        paired_keypoints_list.append(paired_keypoints)

        all_paired_keypoints.append(paired_keypoints_list)

    return all_paired_keypoints

def extract_high_confidence_keypoints(cam_dirs, confidence_threshold):
    """
    Extracts high-confidence keypoints (x, y) from all cameras simultaneously,
    only if all cameras meet the confidence threshold for the corresponding keypoint.
    
    Args:
        - cam_dirs: List of directories containing JSON files for each camera.
        - confidence_threshold: Confidence value threshold for keypoints.
        
    Returns:
        - high_confidence_keypoints: A list of high-confidence keypoints for each frame,
                                     where each element is a dictionary with camera names as keys
                                     and corresponding keypoints as values.
    """
    high_confidence_keypoints = []
    
    # Get sorted list of JSON files for each camera
    cam_files = {}
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        cam_files[cam_name] = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])
    
    # Iterate over frames simultaneously for all cameras
    for frame_files in zip(*cam_files.values()):
        frame_keypoints = {}
        
        # Extract keypoints and confidence for each camera
        cam_keypoints = {}
        for cam_name, frame_file in zip(cam_files.keys(), frame_files):
            with open(frame_file, 'r') as file:
                data = json.load(file)
                
                if data['people']:
                    keypoints = data['people'][0]['pose_keypoints_2d']
                    keypoints_conf = [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]
                    cam_keypoints[cam_name] = keypoints_conf
        
        # Check if all cameras have the same number of keypoints
        if len(set(len(kp) for kp in cam_keypoints.values())) == 1:
            # Filter keypoints based on confidence threshold across all cameras
            for i in range(len(next(iter(cam_keypoints.values())))):
                if all(cam_keypoints[cam][i][2] >= confidence_threshold for cam in cam_keypoints):
                    kp_coords = {cam: (cam_keypoints[cam][i][0], cam_keypoints[cam][i][1]) for cam in cam_keypoints}
                    frame_keypoints[i] = kp_coords
        
        if frame_keypoints:
            high_confidence_keypoints.append(frame_keypoints)
    
    return high_confidence_keypoints


def extract_paired_nose_keypoints_with_reference(ref_cam_dir, other_cam_dirs, confidence_threshold):
    """
    Extracts paired nose keypoints (x, y) with reference camera and each of the other cameras.

    Args:
    - ref_cam_dir: Directory containing JSON files for the reference camera.
    - other_cam_dirs: List of directories containing JSON files for other cameras.
    - confidence_threshold: Confidence value threshold for keypoints.

    Returns:
    - all_paired_nose_keypoints: A list containing paired nose keypoints for each camera pair.
    """
    all_paired_nose_keypoints = []

    # Load and sort JSON files from the reference camera directory
    ref_cam_files = sorted([os.path.join(ref_cam_dir, f) for f in os.listdir(ref_cam_dir) if f.endswith('.json')])

    for cam_dir in other_cam_dirs:
        # Load and sort JSON files from the current camera directory
        cam_files = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])

        paired_nose_keypoints_list = []
        for ref_file, cam_file in zip(ref_cam_files, cam_files):
            with open(ref_file, 'r') as file1, open(cam_file, 'r') as file2:
                ref_data = json.load(file1)
                cam_data = json.load(file2)

                if ref_data['people'] and cam_data['people']:
                    ref_keypoints = ref_data['people'][0]['pose_keypoints_2d']
                    cam_keypoints = cam_data['people'][0]['pose_keypoints_2d']

                    # Extract nose keypoint (index 0) with confidence
                    ref_nose_keypoint_conf = (ref_keypoints[0], ref_keypoints[1], ref_keypoints[2])
                    cam_nose_keypoint_conf = (cam_keypoints[0], cam_keypoints[1], cam_keypoints[2])

                    # Filter nose keypoint based on confidence threshold and pair them
                    if ref_nose_keypoint_conf[2] >= confidence_threshold and cam_nose_keypoint_conf[2] >= confidence_threshold:
                        paired_nose_keypoint = ((ref_nose_keypoint_conf[0], ref_nose_keypoint_conf[1]), (cam_nose_keypoint_conf[0], cam_nose_keypoint_conf[1]))
                        paired_nose_keypoints_list.append(paired_nose_keypoint)

        all_paired_nose_keypoints.append(paired_nose_keypoints_list)

    return all_paired_nose_keypoints