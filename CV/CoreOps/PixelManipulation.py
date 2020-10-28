from Utils.Admin.standard import *
import cv2

# Load data
img = cv2.imread(dataPath + 'Images/messi5.jpg')

# Get RGB pixel
px = img[100,100]
print(px)


# Get only blue pixel
blue = img[100,100,0]
print(blue)

# Set pixel value
img[100,100] = [255,255,255]
print(img[100,100])

# Properties
print(image.shape)
print(image.size)
print(image.dtype)

