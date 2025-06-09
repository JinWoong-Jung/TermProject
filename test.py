from ultralytics import YOLO
import os

# 1. 모델 불러오기 (train 결과 사용)
model = YOLO('runs/detect/train/weights/best.pt')  # train 이름 체크

# 2. 테스트 이미지 디렉토리 설정
test_dir = 'data/test/images'  # 반드시 여기에 .jpg/.png 이미지가 있어야 함

# 3. Inference 수행
results = model.predict(
    source=test_dir,         # 이미지 폴더 또는 단일 이미지 경로
    imgsz=416,               # 이미지 크기 (train 시 사용한 크기와 맞춰줌)
    conf=0.25,               # confidence threshold (필요 시 조절)
    save=True,               # 이미지 파일 저장
    save_txt=True,           # 텍스트(.txt) 결과도 저장
    project='runs/detect/predict',  # 결과 저장 위치
    name='images',          # runs/detect/predict/images
    exist_ok=True            # 이미 폴더가 있어도 덮어쓰기
)

print("Inference 완료. 결과는 runs/detect/predict/images 에 저장됨.")