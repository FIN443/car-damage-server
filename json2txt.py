import os
import json

damage = {
    'Breakage' : 0, 
    'Separated' : 1,
    'Crushed' : 2,
    'Scratched' : 3,
}

# json 파일이 있는 폴더 
json_path = "E:/2022yeardream/CV project/yolo-v4-tf.keras-master/yolo-v4-tf.keras-master/dataset/json/"
json_files = os.listdir(json_path)

# txt 파일 저장 장소 ( 미리 만들어 둘것 )
file = open("E:/2022yeardream/CV project/yolo-v4-tf.keras-master/yolo-v4-tf.keras-master/dataset/"+'anno-test/txt/' + ".txt", "w")

for fn in json_files:
#     num = str(i).zfill(4)
    with open(json_path + fn, 'r') as f:
        json_data = json.load(f)
        label_info = json_data['info']

        image_name = json_data['images']['file_name'] # 이미지 파일명
        # image_width = label_info['image']['width'] # 이미지 넓이
        # image_height = label_info['image']['height'] # 이미지 높이
        image_width = 800
        image_height = 600
        annotations =  json_data['annotations']
        image_name = image_name.replace(".jpg", "")
        
        # 
        file.write(f"{fn[:-5]}.jpg ")
        for annot in annotations:
            x1, y1, x2, y2 = annot['bbox'] # 객체 1개의 bbox 위치 좌표
            x2 += x1
            y2 += y1
            classes = annot['damage']
            file.write(f"{x1},{y1},{x2},{y2},{damage[classes]} ")
        file.write("\n")
file.close()

## txt 저장 형식
# 0506233_sc-202337.jpg 419,204,430,225,0 2,20,800,253,1 