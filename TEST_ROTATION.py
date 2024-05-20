import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_quaternion(rotation_matrix):
    print("Rotation matrix:", rotation_matrix)
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    print("Quaternion:", quaternion)
    return quaternion

def quaternion_to_rotation_matrix(quaternion):
    print("Quaternion:", quaternion)
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    print("Rotation matrix:", rotation_matrix)
    return rotation_matrix

def test_rotation_conversion():
    # 기존 코드에서 사용된 실제 데이터
    ext_R = np.array([
        [0.999998, -0.001745, 0.001745],
        [0.001745, 0.999998, -0.001745],
        [-0.001745, 0.001745, 0.999998]
    ])
    
    # 회전 행렬을 쿼터니언으로 변환
    quaternion = rotation_matrix_to_quaternion(ext_R)
    print("Quaternion:", quaternion)
    
    # 쿼터니언을 다시 회전 행렬로 변환
    converted_matrix = quaternion_to_rotation_matrix(quaternion)
    
    # 결과 출력
    print("Original rotation matrix:")
    print(ext_R)
    print("Converted rotation matrix:")
    print(converted_matrix)
    
    # 변환된 행렬과 원래 행렬이 동일한지 확인
    if np.allclose(ext_R, converted_matrix):
        print("Test passed: The converted rotation matrix matches the original.")
    else:
        print("Test failed: The converted rotation matrix does not match the original.")

if __name__ == "__main__":
    test_rotation_conversion()
