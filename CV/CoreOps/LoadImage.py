from Utils.Admin.Standard import *
import cv2



# Load an color image in grayscale
img = cv2.imread(imgPath + 'messi5.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('messigray.png',img)

# Read and write pixel (see white pixel on the elbow)
print(img[100,100])
img[100,100] = [255,255,255]
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Image features
print('Size : ' , img.size)
print('Shape: ', img.shape)
print('DType: ', img.dtype)

# Matplotlib
img = cv2.imread(imgPath + 'messi5.jpg', 0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
