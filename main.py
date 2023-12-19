import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt


def sharpenImage(im):  # Need to check if sharpening input images would help accuracy
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    im = cv2.filter2D(im, -1, kernel)  # -1 means depth of image remains unchanged
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    im = 255 - im
    return im


# image = cv2.imread("DATA/ocrTestImage.jpeg", cv2.IMREAD_COLOR)
img = cv2.imread("DATA/ocrTestImage.jpeg", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (1200,600))
img = sharpenImage(img)


# plt.imshow(img)
# plt.show()


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


a = np.sum(img == 255, axis=1)
rows = []
seg = []

# print(a)
# plt.imshow(img)
# plt.show()


for i in range(len(a)):
    if a[i] > 100:
        seg.append(i)
    if (a[i] <= 100) & (len(seg) >= 15):
        rows.append(seg)
        seg = []

    if len(seg) >= 0:
        rows.append(seg)
# number of row segments
len(rows)

plt.imshow(img[rows[0][0]:rows[0][-1], :])
plt.show()

# img2 = cv2.imread("DATA/ocrTestImage.jpeg", 0)
# plt.imshow(img)
# plt.show()
