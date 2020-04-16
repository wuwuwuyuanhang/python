# @Auther : wuwuwu 
# @Time : 2020/4/15 
# @File : q22.py
# @Description : 直方图操作

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histOperation(img, m0=128, s0=52):
    """
    变更直方图的均值与标准差
    :param img: 输入图片
    :param m0: 变更均值
    :param s0: 变更标准差
    :return: 输出图片
    """

    m = img.mean()
    s = img.std()

    dst = img.copy().astype(np.float32)

    dst = s0 / s * (dst - m) + m0

    dst = np.clip(dst, 0, 255).astype(np.uint8)

    return dst

if __name__ == '__main__':
    img = cv.imread('lenna_dark.jpg')
    dst = histOperation(img, m0=128, s0=52)
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