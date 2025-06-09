import ultralytics
from ultralytics import YOLO

ultralytics.checks()
model = YOLO('yolov8s.pt')
print(type(model.names), len(model.names))

print(model.names)

model.train(data='data.yaml', epochs=100, patience=30, batch=32, imgsz=416)