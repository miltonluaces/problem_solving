from Utils.Admin.Standard import *
import cv2

img = cv2.imread(imgPath + 'charts/chart1.png')
cv2.imshow("chart", img) 
#cv2.waitKey(0)

#converted = convert_hls(img)
image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
# White mask
lwr = np.uint8([0, 200, 0])
upr = np.uint8([255, 255, 255])
whiteMask = cv2.inRange(image, lwr, upr)

# yellow color mask
lwr = np.uint8([10, 0,   100])
upr = np.uint8([40, 255, 255])
yellowMask = cv2.inRange(image, lwr, upr)

# combine the mask
mask = cv2.bitwise_or(whiteMask, yellowMask)
cv2.imshow("mask",mask) 
cv2.waitKey(0)

res = img.copy()

