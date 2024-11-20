import cv2
from ultralytics import YOLO

img = cv2.imread("ex1.jpg")

model = YOLO("yolov8x-pose.pt")

results = model(img, save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data[0]) 

skelton =[[5,6],[6,8],[8,10],[5,7],[7,9],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]

for j in range (0, 12):
    
    cv2.line(img,(int(keypoints.data[0][skelton[j][0]][0]),int(keypoints.data[0][skelton[j][0]][1])),(int(keypoints.data[0][skelton[j][1]][0]),int(keypoints.data[0][skelton[j][1]][1])),(0,0,255),thickness=+5)  #画像　始点　終点　RGB　太さ


for i in range (5,17):
    
#丸
    cv2.circle(img, (int(keypoints.data[0][i][0]),int(keypoints.data[0][i][1])), 5, (0,255,255),thickness=-1) # 画像　(x座標,y座標) 半径　RGB　塗りつぶし
 



cv2.imshow("ex1.jpg", img)
cv2.waitKey(0)