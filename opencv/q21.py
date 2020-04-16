# @Auther : wuwuwu 
# @Time : 2020/4/15 
# @File : q21.py
# @Description : 直方图归一化

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogramNormalization(img, a=0, b=255):
    """
    归一化
    :param img: 输入图片
    :param a: 最小值
    :param b: 最大值
    :return: 输出图片
    """
    c = img.min()
    d = img.max()

    dst = img.copy().astype(np.float32)
    dst = (b - a) / (d - c) * (dst - c)
    dst[dst < a] = a
    dst[dst > b] = b

    return dst.astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('lenna_dark.jpg')
    dst = histogramNormalization(img, a=0, b=255)
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
