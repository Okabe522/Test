import cv2
from ultralytics import YOLO
import numpy as np
import torch


img = cv2.imread("ex1.jpg")

model = YOLO("yolov8x-pose.pt")
img_results = model(img, save=True, save_txt=True, save_conf=True)
img_keypoints = img_results[0].keypoints.data.cpu().numpy()

video_path = "ex3b.mp4"

cap = cv2.VideoCapture(video_path)
# 動画を開く

model = YOLO("yolov8x-pose.pt")

count = 0
distance = 0
distances = [] 

while cap.isOpened():
    success, frame = cap.read()
    # フレームを読み出す
    if success:
        video_results = model(frame, save=True, save_txt=True, save_conf=True)
        video_keypoints = video_results[0].keypoints.data.cpu().numpy()
        
        
        
        distance = np.linalg.norm(img_keypoints[0] - video_keypoints[0]).mean()
        distances.append((count, distance))  # フレーム番号と距離を記録
        


        
        count = count +1   
                
  
       


        cv2.imshow("ex3b.mp4", frame)
         # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break
  
if distances:
    min_frame, min_distance = distances[0]
    for frame, distance in distances:
        if distance < min_distance:
            min_frame, min_distance = frame, distance  
            
print(min_frame)   
    
cap.release()
cv2.destroyAllWindows()
