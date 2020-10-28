from Utils.Admin.Standard import *
import cv2
import cv2

img = cv2.imread(imgPath + 'messi5.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img2 = cv2.resize(img2Big,(450,280))
res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


#OR

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
