# @Auther : wuwuwu 
# @Time : 2020/4/16 
# @File : q25.py
# @Description : 最近邻插值

import cv2 as cv
import numpy as np

def nearestNeighborInterpolation(img, ax=1.0, ay=1.0):
    """
    
    :param img: 
    :param ax: 
    :param ay: 
    :return: 
    """""

    H, W, C = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    # index
    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))
    y = np.round(y / ay).astype(np.int)
    x = np.round(x / ax).astype(np.int)

    dst = img[y, x]

    return dst.astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('lenna.jpg')
    dst = nearestNeighborInterpolation(img, ax=1.5, ay=1.5)
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()