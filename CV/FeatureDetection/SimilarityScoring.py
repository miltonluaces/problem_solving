from Utils.Admin.Standard import *
import cv2
from skimage.measure import compare_ssim

# load the two input images
imgA = cv2.imread(imgPath + 'cardA.jpg')
imgB = cv2.imread(imgPath + 'cardB.jpg')
imgC = cv2.imread(imgPath + 'cardC.jpg')

cv2.imshow('imgA', imgA)
cv2.imshow('imgB', imgB)
cv2.imshow('imgC', imgC)
cv2.waitKey(0)


# resize
stSize = (200,120)
imgA = cv2.resize(imgA,stSize)
imgB = cv2.resize(imgB,stSize)
imgC = cv2.resize(imgC,stSize)

cv2.imshow('imgA', imgA)
cv2.imshow('imgB', imgB)
cv2.imshow('imgC', imgC)
cv2.waitKey(0)

 
# convert the images to grayscale
imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

cv2.imshow('imgA', imgA)
cv2.imshow('imgB', imgB)
cv2.imshow('imgC', imgC)
cv2.waitKey(0)
cv2.destroyAllWindows()

(scoreAA, diffAA) = compare_ssim(imgA, imgA, full=True)
(scoreAB, diffAB) = compare_ssim(imgA, imgB, full=True)
(scoreAC, diffAC) = compare_ssim(imgA, imgC, full=True)
#diffAB = (diff * 255).astype("uint8")
print("AC: {}".format(scoreAC))
print("AB: {}".format(scoreAB))
print("AA: {}".format(scoreAA))