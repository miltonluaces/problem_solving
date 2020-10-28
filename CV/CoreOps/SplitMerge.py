from Utils.Admin.Standard import *
import cv2


# Format manipulation (eliminate green)
img = cv2.imread(imgPath + 'messi5.jpg')
b,g,r = cv2.split(img)
print(b.shape, g.shape, r.shape)
img = cv2.merge((b,b,r))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

