from Utils.Admin.Standard import *
import cv2

# Image ROI (Region of interest)
img = cv2.imread(imgPath + 'messi5.jpg', 0)
cv2.imshow('image',img)
roi = img[180:240, 210:290]
print(roi.shape)
print(img[220:320, 250:330].shape)
img[220:280, 250:330] = roi
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
