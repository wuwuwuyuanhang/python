# @Auther : wuwuwu 
# @Time : 2020/4/15 
# @File : q23.py
# @Description : 直方图均衡化

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogramEqualization(img, Zmax=255):
    """
    直方图均衡化
    :param img:
    :param Zmax: 像素的最大取值
    :return:
    """
    H, W, C = img.shape
    S = H * W * C

    dst = img.copy()

    sum_h = 0

    for i in range(1, 255):
        index = np.where(img == i)
        sum_h += len(img[index])
        dst[index] = Zmax / S * sum_h

    return np.clip(dst, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('lenna.jpg')
    dst = histogramEqualization(img, Zmax=255)
    plt.figure()
    plt.hist(img.flatten(), bins=255, rwidth=0.8, range=(0, 255))
    plt.title('input histogram')
    plt.figure()
    plt.hist(dst.flatten(), bins=255, rwidth=0.8, range=(0, 255))
    plt.title('output histogram')
    plt.show()
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()