# @Auther : wuwuwu 
# @Time : 2020/4/22 
# @File : q28.py
# @Description : 仿射变换之平移

import cv2 as cv
import numpy as np

def afineTraslation(img, a=1., b=0., c=0., d=1., tx=0., ty=0.):
    """

    :param img:
    [ [a, b, tx],
      [c, d, ty],
      [0, 0, 1] ]
    :return:
    """

    H, W, C = img.shape

    y = np.arange(H).repeat(W).reshape(W, -1)
    x = np.tile(np.arange(W), (H, 1))

    index_y = a * y + b * x + ty
    index_x = c * y + d * x + tx

    index_y = np.clip(index_y, 0, H - 1).astype(np.int32)
    index_x = np.clip(index_x, 0, W - 1).astype(np.int32)

    dst = np.zeros_like(img)

    dst[index_y, index_x] = img[index_y, index_x]

    return dst

if __name__ == '__main__':
    img = cv.imread('lenna.jpg')
    dst = afineTraslation(img, a=1., b=0., c=0., d=1., tx=30., ty=-30)
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
