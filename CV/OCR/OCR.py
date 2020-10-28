#from wand.image import Image as Img
#with Img(filename='file_name.pdf', resolution=300) as img:
#    img.compression_quality = 99
#    img.save(filename='image_name.jpg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

# Load an color image in grayscale
img = cv2.imread("../../../../data/Images/inv1.jpeg",0)

#cv2.imshow('image', img)
#cv2.waitKey(0)"../../../../data/Images/inv1.jpeg"
#cv2.destroyAllWindows()

#plt.imshow(img)
#plt.show()

#text = pytesseract.image_to_string(Image.open("../../../../data/Images/inv1.jpeg"))
im = Image.open("../../../../data/Images/inv1.jpeg")
text = pytesseract.image_to_boxes(im, lang = 'eng')
print(text)