import cv2
from ultralytics import YOLO
import numpy as np

img = cv2.imread("ex2.jpg")

model = YOLO("yolov8x.pt")
results = model(img, save=True, save_txt=True, save_conf=True)


def blue(img):
    #BGR色空間からHSV色空間への変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #色検出しきい値の設定
    lower = np.array([90,64,70])
    upper = np.array([150,255,255])

    #色検出しきい値範囲内の色を抽出するマスクを作成
    frame_mask = cv2.inRange(hsv, lower, upper)
    
    pixel = cv2.countNonZero(frame_mask)

    return pixel

boxes = results[0].boxes



for box in boxes:
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 座標を整数に変換

    if box.cls[0] == 0:
        
        # 人物の領域を切り出し
        person_region = img[y1:y2, x1:x2] #「y1 から y2 の高さ」と「x1 から x2 の幅」
        
        blue_mask = blue(person_region)
        
        if blue_mask > 0:
    
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),thickness=3)
        


    
cv2.imshow("ex2.jpg", img)
cv2.waitKey(0)