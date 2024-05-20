import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def augment_keypoints_2d(original_points, image_size, num_augmented_points, noise_std=50, translation_range=(-50, 50)):
    """
    2D 공간에서 원래 키포인트를 기반으로 증강 키포인트를 생성합니다.
    """
    augmented_points = []
    for _ in range(num_augmented_points):
        # 원래 포인트에서 무작위로 하나 선택
        point = random.choice(original_points)
        
        # 이동 및 노이즈 추가
        dx = random.uniform(*translation_range)
        dy = random.uniform(*translation_range)
        noisy_point = point + np.array([dx, dy]) + np.random.normal(0, noise_std, 2)
        
        # 이미지 경계를 벗어나지 않도록 클리핑
        noisy_point[0] = np.clip(noisy_point[0], 0, image_size[0] - 1)
        noisy_point[1] = np.clip(noisy_point[1], 0, image_size[1] - 1)
        
        augmented_points.append(tuple(noisy_point))
    
    return augmented_points

def plot_keypoints(original, augmented, frame_index, image_size):
    """
    기존 키포인트와 증강된 키포인트를 비교하여 플롯합니다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Frame {frame_index + 1} Keypoints')
    
    axes[0].set_title('Original Keypoints')
    axes[0].set_xlim(0, image_size[0])
    axes[0].set_ylim(0, image_size[1])
    axes[0].invert_yaxis()
    for keypoints in original:
        x, y = zip(*keypoints)
        axes[0].scatter(x, y, label='Camera')
    
    axes[1].set_title('Original + Augmented Keypoints')
    axes[1].set_xlim(0, image_size[0])
    axes[1].set_ylim(0, image_size[1])
    axes[1].invert_yaxis()
    for keypoints in original:
        x, y = zip(*keypoints)
        axes[1].scatter(x, y, label='Original', c='blue')
    for keypoints in augmented:
        x, y = zip(*keypoints)
        axes[1].scatter(x, y, label='Augmented', c='red', marker='x')
    
    plt.show()

# 대규모 예제 데이터
keypoints = [
    # 첫 번째 프레임
    [
        [(1000, 2000), (1500, 2500), (2000, 3000)],  # 기준 카메라의 키포인트
        [(1100, 2100), (1600, 2600), (2100, 3100)]   # 다른 카메라의 키포인트
    ],
    # 두 번째 프레임
    [
        [(1200, 2200), (1700, 2700), (2200, 3200)],  # 기준 카메라의 키포인트
        [(1300, 2300), (1800, 2800), (2300, 3300)]   # 다른 카메라의 키포인트
    ]
]

image_size = [3840.0, 2160.0]

# 증강 데이터 생성
num_augmented_points = 100  # 생성할 증강 포인트 수
augmented_keypoints = []

for frame in keypoints:
    original_reference = frame[0]
    augmented_reference = augment_keypoints_2d(original_reference, image_size, num_augmented_points)
    augmented_keypoints.append([augmented_reference])

# 다른 카메라에서도 동일하게 증강된 키포인트를 생성
for i, frame in enumerate(augmented_keypoints):
    augmented_reference = frame[0]
    augmented_other_cameras = []
    for _ in range(1, len(keypoints[i])):
        augmented_other_cameras.append(augment_keypoints_2d(augmented_reference, image_size, num_augmented_points))
    frame.extend(augmented_other_cameras)

# 결과 출력 및 플롯
for i, (original, augmented) in enumerate(zip(keypoints, augmented_keypoints)):
    print(f"Frame {i + 1}:")
    print(f"Original:")
    print(f" 기준 카메라: {original[0]}")
    print(f" 다른 카메라: {original[1:]}")
    print(f"Augmented:")
    print(f" 기준 카메라: {augmented[0]}")
    print(f" 다른 카메라: {augmented[1:]}")
    plot_keypoints(original, augmented, i, image_size)
    print()
