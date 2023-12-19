import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt


def sharpenImage(im):  # Need to check if sharpening input images would help accuracy
    kernel = np.ones((3,3), np.float32) / 90
    im = cv2.filter2D(im, -1, kernel)  # -1 means depth of image remains unchanged
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    im = 255 - im
    return im


img = cv2.imread("DATA/ocrTestImage.jpeg", 0)
img = sharpenImage(img)


def alignImg(im):
    coords = np.column_stack(np.where(im > 0))
    angle = cv2.minAreaRect(coords)[-1]  # -1 denotes angle parameter

    if angle < -45:
        angle = -(angle + 90)
    else:
        angle = -angle + 90
    h, w = im.shape  # height and width of img
    center = (w // 2, h // 2)
    rotMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, rotMatrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


img = alignImg(img)

pixelSum = np.sum(img == 255, axis=1)
rows=[]
seg=[]

count=0
mini=1000

for i in range(len(pixelSum)):
    mini=min(mini,pixelSum[i])
    if(pixelSum[i]<=250):
        count=count+1


    if(pixelSum[i]>250):
        seg.append(i)

    if(pixelSum[i]<=250) & (len(seg)>=100):
        rows.append(seg)
        seg=[]

    if(len(seg)>0):
        rows.append(seg)
        seg=[]

print("NO OF LINES",len(rows))
print(count,mini)

img2 = cv2.imread("DATA/ocrTestImage.jpeg", 0)
plt.imshow(img2)
plt.show()