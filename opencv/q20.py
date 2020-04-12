# @Auther : wuwuwu 
# @Time : 2020/4/11 
# @File : q20.py
# @Description : 直方图

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def BGR2GRAY(img):
    dst = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return dst.astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    gray = BGR2GRAY(img)

    plt.hist(gray.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()