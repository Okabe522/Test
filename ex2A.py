import cv2
from ultralytics import YOLO

img = cv2.imread("ex2.jpg")

model = YOLO("yolov8x.pt")

results = model(img, save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes



max_area = 0  
max_box = None 

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 座標を整数に変換
    area = (x2 - x1) * (y2 - y1)  # 面積を計算

    if area > max_area:  
        max_area = area
        max_box = (x1, y1, x2, y2)


#面積が最大のところを赤枠で描画
x1, y1, x2, y2 = max_box
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)


    
cv2.imshow("ex2A.jpg", img)
cv2.waitKey(0)










