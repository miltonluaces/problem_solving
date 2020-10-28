from Utils.Admin.Standard import *
import cv2
from FeatureDetection.Lines import GetLines

img = cv2.imread(imgPath + 'charts/chart1.png', 0)
cv2.imshow("chart", img) 
cv2.waitKey(0)

lines = GetLines(img, lwrThres=50, uprThres=150, rho=1, theta=np.pi/180, thres=15, minLineLenght=50, maxLineGap=20)
print(lines)