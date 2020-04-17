# @Auther : wuwuwu 
# @Time : 2020/4/16 
# @File : q24.py
# @Description : 伽马矫正

import cv2 as cv
import numpy as np

def gammaCorrection(img, c=1, g=2.2):
    """

    :param img:
    :param c:
    :param g:
    :return:
    """

    dst = img.copy() / 255.

    dst = dst ** (1 / g) / c * 255
    return np.clip(dst, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('lenna.jpg')
    dst = gammaCorrection(img, c=1, g=2.2)
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()