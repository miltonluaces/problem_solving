from Utils.Admin.Standard import *
from skimage.measure import compare_ssim
import numpy as np
import cv2

class ImgSim:

  
    def __init__(self, srcImgs, tarImgs, stSize, partial, thres):
        self.srcImgs = srcImgs
        self.tarImgs = tarImgs
        self.stSize = stSize
        self.partial = partial
        self.thres = thres

    def Calculate(self):
        self.srcImgs = self.preprocess(self.srcImgs)
        self.tarImgs = self.preprocess(self.tarImgs)
        self.scoreImgs()
         
    def preprocess(self, imgs):
            prepImgs = []
            for img in imgs:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, self.stSize)
                prepImgs.append(img)
            return prepImgs
     
    def scoreImgs(self):
        self.scores = np.zeros((len(self.srcImgs), len(self.tarImgs)), dtype=np.float64)
        t=-1; s=-1
        for tar in self.tarImgs:
            t=t+1; s=-1
            for src in self.srcImgs:
                s=s+1
                score = compare_ssim(src, tar)
                self.scores[t,s] = score
             
            



if __name__=="__main__":
    print('Test')
    imgA = cv2.imread(imgPath + 'cardA.jpg')
    imgB = cv2.imread(imgPath + 'cardB.jpg')
    imgC = cv2.imread(imgPath + 'cardC.jpg')
    srcImgs = [imgA, imgB]
    tarImgs = [imgB, imgC]

    iS = ImgSim(srcImgs, tarImgs, (200,120), False, 0.7)
    iS.Calculate()
    print(iS.scores)
