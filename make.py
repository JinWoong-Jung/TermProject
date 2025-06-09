import pandas as pd
import os

facetest_path = 'data/facetest.txt'
faceval_path = 'data/faceval.txt'

with open(facetest_path, 'r') as f:
    lines = f.readlines()

facetest_list = list([line.strip().replace('.jpg', '') for line in lines])

print(facetest_list)

with open(faceval_path, 'r') as f:
    lines = f.readlines()

facevalid_list = list([line.strip().replace('.jpg', '') for line in lines])

print(facevalid_list)


# CSV 및 분할 기준 텍스트 파일 경로
csv_path = 'faces.csv'
facetest_path = 'data/facetest.txt'
faceval_path = 'data/faceval.txt'

# 출력 디렉토리 설정
label_dirs = {
    'test': 'data/test/labels',
    'valid': 'data/valid/labels',
    'train': 'data/train/labels'
}

# CSV 파일 로드
df = pd.read_csv('faces.csv')

# YOLO 형식으로 변환하는 함수 정의
def convert_to_yolo_format(row):
    x_center = ((row['x0'] + row['x1']) / 2) / row['width']
    y_center = ((row['y0'] + row['y1']) / 2) / row['height']
    w = (row['x1'] - row['x0']) / row['width']
    h = (row['y1'] - row['y0']) / row['height']
    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

# 각 row에 대해 YOLO 좌표 생성
df['yolo'] = df.apply(convert_to_yolo_format, axis=1)

# 이미지 이름 기준으로 그룹화하여 레이블 생성
label_dict = df.groupby('image_name')['yolo'].apply(list).to_dict()

# 출력 디렉토리 생성
output_dir = "data/train/labels"
os.makedirs(output_dir, exist_ok=True)

# 각 이미지별 YOLO 레이블을 개별 파일로 저장
for image_name, yolo_lines in label_dict.items():
    name = os.path.splitext(image_name)[0]
    label_file = name + ".txt"
    if name in facetest_list: output_dir=label_dirs['test']
    elif name in facevalid_list: output_dir=label_dirs['valid']
    else: output_dir=label_dirs['train']
    with open(os.path.join(output_dir, label_file), 'w') as f:
        f.write("\n".join(yolo_lines))
