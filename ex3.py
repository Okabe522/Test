import cv2
from ultralytics import YOLO
import math

video_path = "ex3b.mp4"

cap = cv2.VideoCapture(video_path)
# 動画を開く

model = YOLO("yolov8x-pose.pt")

#角度を求める関数
def cos_formula(p1, p2, p3):
    a = math.dist(p1, p2)
    b = math.dist(p1, p3)
    c = math.dist(p2, p3)
    cos_c = (a**2 + b**2 - c**2)/(2*a*b)
    acos_c = math.acos(cos_c)
    return math.degrees(acos_c)



skelton =[[5,6],[6,8],[8,10],[5,7],[7,9],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]

while cap.isOpened():
    success, frame = cap.read()
    # フレームを読み出す
    if success:
        results = model(frame, save=True, save_txt=True, save_conf=True)
        keypoints = results[0].keypoints

        for i in range (0, 12):
    
            cv2.line(frame,(int(keypoints.data[0][skelton[i][0]][0]),int(keypoints.data[0][skelton[i][0]][1])),(int(keypoints.data[0][skelton[i][1]][0]),int(keypoints.data[0][skelton[i][1]][1])),(255,0,0),thickness=+5)  #画像　始点　終点　RGB　太さ

  
  
            #Right-elbow, Right-shoulder, Right-hip
            elbow_x = keypoints.data[0][6][0]
            shoulder_x = keypoints.data[0][8][0]
            hip_x = keypoints.data[0][12][0]
            
            elbow_y = keypoints.data[0][6][1]
            shoulder_y = keypoints.data[0][8][1]
            hip_y = keypoints.data[0][12][1]
            
            kakudo = cos_formula((elbow_x,elbow_y), (shoulder_x,shoulder_y), (hip_x,hip_y))
            
            if kakudo>=80 and kakudo <=100:
                cv2.line(frame,(int(keypoints.data[0][6][0]),int(keypoints.data[0][6][1])),(int(keypoints.data[0][8][0]),int(keypoints.data[0][8][1])),(0,0,255),thickness=+5)

                cv2.line(frame,(int(keypoints.data[0][8][0]),int(keypoints.data[0][8][1])),(int(keypoints.data[0][10][0]),int(keypoints.data[0][10][1])),(0,0,255),thickness=+5)
        
  


        cv2.imshow("ex3b.mp4", frame)
         # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break
cap.release()
cv2.destroyAllWindows()
