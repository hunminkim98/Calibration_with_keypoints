import os
import json


def extract_paired_keypoints_with_reference(all_cam_data, index_of_ref, confidence_threshold):
    """
    Extracts paired keypoints (x, y) with reference camera and each of the other cameras.

    Args:
    - all_cam_dirs: List of directories containing JSON files for all cameras.
    - index_of_ref: Index of the reference camera in the list of all cameras.
    - confidence_threshold: Confidence value threshold for keypoints.

    Returns:
    - all_paired_keypoints: A list containing paired keypoints for each camera pair.
    """

    all_paired_keypoints = []
    index_of_faulty_json = []
    
    for idx_of_other_cams , cams in enumerate(all_cam_data): # For each  in all_cam_data:
        if idx_of_other_cams == index_of_ref:
            continue
        paired_keypoints_list = []
        for j in range(len(cams)): # each element is a frame
            ref_data = all_cam_data[index_of_ref][j]  # single frame
            cam_data = all_cam_data[idx_of_other_cams][j] # single frame
            try:
                ref_keypoints = ref_data['people'][0]['pose_keypoints_2d']
                ref_keypoints_conf = [(ref_keypoints[i], ref_keypoints[i+1], ref_keypoints[i+2]) for i in range(0, len(ref_keypoints), 3)]
                cam_keypoints = cam_data['people'][0]['pose_keypoints_2d']
                cam_keypoints_conf = [(cam_keypoints[i], cam_keypoints[i+1], cam_keypoints[i+2]) for i in range(0, len(cam_keypoints), 3)]
                paired_keypoints = [((kp1[0], kp1[1]), (kp2[0], kp2[1])) for kp1, kp2 in zip(ref_keypoints_conf, cam_keypoints_conf) if kp1[2] >= confidence_threshold and kp2[2] >= confidence_threshold]
                
                # e.g. [((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...] per frame
                
                if paired_keypoints:
                    paired_keypoints_list.append(paired_keypoints)
                # else:
                #     print(f"No paired keypoints found for frame {i}.")
            except:
                index_of_faulty_json.append(j)
                continue
        all_paired_keypoints.append(paired_keypoints_list)
    if index_of_faulty_json:
        print("Warning: Faulty JSON files found for frames:", index_of_faulty_json)        


    return all_paired_keypoints