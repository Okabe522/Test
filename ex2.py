import cv2
from ultralytics import YOLO

img = cv2.imread("ex2.jpg")

model = YOLO("yolov8x.pt")

results = model(img, save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes
for box in boxes:
    print(box.data)
print(results[0].names)


for box in boxes:
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 座標を整数に変換

    if box.cls[0] == 0:
    
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),thickness=3)
        


    
cv2.imshow("ex2.jpg", img)
cv2.waitKey(0)