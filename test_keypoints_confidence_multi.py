import unittest
from keypoints_confidence_multi import extract_high_confidence_keypoints

class TestExtractHighConfidenceKeypoints(unittest.TestCase):
    def test_extract_high_confidence_keypoints(self):
        cam_dirs = [
            '/path/to/cam1',
            '/path/to/cam2',
            '/path/to/cam3'
        ]
        confidence_threshold = 0.5
        
        high_confidence_keypoints = extract_high_confidence_keypoints(cam_dirs, confidence_threshold)
        
        # Assert that the output is a list
        self.assertIsInstance(high_confidence_keypoints, list)
        
        # Assert that each element in the output list is a dictionary
        for frame_keypoints in high_confidence_keypoints:
            self.assertIsInstance(frame_keypoints, dict)
            
            # Assert that each key in the dictionary is a camera name
            for cam_name in frame_keypoints.keys():
                self.assertIsInstance(cam_name, str)
                
                # Assert that each value in the dictionary is a tuple of coordinates
                kp_coords = frame_keypoints[cam_name]
                self.assertIsInstance(kp_coords, tuple)
                self.assertEqual(len(kp_coords), 2)
                self.assertIsInstance(kp_coords[0], float)
                self.assertIsInstance(kp_coords[1], float)
                
if __name__ == '__main__':
    unittest.main()