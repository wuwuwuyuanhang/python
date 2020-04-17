# @Auther : wuwuwu 
# @Time : 2020/4/16 
# @File : q26.py
# @Description : 双线性插值

import cv2 as cv
import numpy as np

def bilinearInterpolation(img, ax=1.0, ay=1.0):
    """

    :param img:
    :param ax:
    :param ay:
    :return:
    """

    H , W, C = img.shape

    aH = int(H * ay)
    aW = int(W * ax)

    # index
    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x =np.tile(np.arange(aW), (aH, 1))
    deltay = y / ay - np.minimum(np.round(y / ay), H - 2)
    deltax = x / ax - np.minimum(np.round(x / ax), W - 2)
    y = np.minimum(np.round(y / ay), H - 2).astype(np.int)
    x = np.minimum(np.round(x / ax), W - 2).astype(np.int)

    # expand dims
    deltax = np.repeat(np.expand_dims(deltax, axis=-1), 3, axis=-1)
    deltay = np.repeat(np.expand_dims(deltay, axis=-1), 3, axis=-1)

    dst = (1 - deltay) * (1 - deltax) * img[y, x] + deltay * (1 - deltax) * img[y + 1, x] + \
          (1 - deltay) * deltax * img[y, x + 1] + deltay * deltax * img[y + 1, x + 1]

    return np.clip(dst, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('lenna.jpg')
    dst = bilinearInterpolation(img, ax=1.5, ay=1.5)
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
