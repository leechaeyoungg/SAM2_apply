import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import hydra
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 설정
checkpoint = r"C:\Users\dromii\segment-anything-2\checkpoints\sam2_hiera_large.pt"
model_cfg = r"sam2_hiera_l.yaml"

# Hydra의 config search path 설정
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=r"C:\Users\dromii\segment-anything-2\sam2_configs")

# SAM2 모델 로드
sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

# 자동 마스크 생성기 초기화
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# 이미지 로드 및 RGB 변환
image_path = r"D:\IE002833358_STD.jpg"  # 이미지 경로 설정
image = Image.open(image_path)
image = np.array(image.convert("RGB"))

# 마스크 생성
masks = mask_generator.generate(image)

# 탐지된 객체 수 출력
print(f"Detected objects: {len(masks)}")

# 생성된 마스크를 표시하는 함수
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

# 마스크를 이미지에 오버레이하여 표시
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()


