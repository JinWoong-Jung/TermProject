import ultralytics
from ultralytics import YOLO
import torch.nn as nn

yolo = YOLO('yolov8s.pt')

# backbone, neck, head를 직접 수정하려면 PyTorch 수준에서 접근해야 함
model = yolo.model  # 실제 nn.Module 객체

# 3. 마지막 레이어(head) 수정 -> Fine tuning 예시.
model.model[-1] = nn.Sequential(
    nn.Conv2d(1024, 1024, kernel_size=1),
    nn.ReLU(),
    model.model[-1]  # 원래 head
)

# 4. 수정한 모델을 다시 YOLO 객체에 넣기
yolo.model = model

# 5. 학습 시작 (Ultralytics API 사용 가능)
yolo.train(data='data.yaml', epochs=10, batch=64, imgsz=416) # epoch 조절하면서 할것.