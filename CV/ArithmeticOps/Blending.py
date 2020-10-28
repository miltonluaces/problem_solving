from Utils.Admin.Standard import *
import cv2


img1 = cv2.imread(imgPath + 'messi5.jpg')
img2Big = cv2.imread(imgPath + 'shankill.jpg')
img2 = cv2.resize(img2Big,(450,280))

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
