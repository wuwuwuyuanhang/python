# @Auther : wuwuwu 
# @Time : 2020/4/22 
# @File : q27.py
# @Description : 双三次插值

import cv2 as cv
import numpy as np

def bicubicInterpolation(img, ax=1., ay=1.):
    """

    :param img:
    :param ax:
    :param ay:
    :return:
    """

    H, W, C = img.shape

    aH = int(H * ay)
    aW = int(W * ax)

    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    y = y / ay
    x = x / ax

    yi = np.round(y).astype(np.int32)
    xi = np.round(x).astype(np.int32)

    yi = np.clip(yi, 1, H - 2)
    xi = np.clip(xi, 1, W - 2)

    dy = [np.abs(y - (yi - 1)), np.abs(y - yi), np.abs(y - (yi + 1)), np.abs(y - (yi + 2))]
    dx = [np.abs(x - (xi - 1)), np.abs(x - xi), np.abs(x - (xi + 1)), np.abs(x - (xi + 2))]

    def weight(t):
        a = -1
        t = np.repeat(np.expand_dims(t, axis=-1), C, axis=-1)
        dst = np.zeros_like(t)
        b = np.abs(t)

        index = np.where(b <= 1)
        dst[index] = (a + 2) * b[index] ** 3 - (a + 3) * b[index] ** 2 + 1

        index = np.where((b > 1) & (b <= 2))
        dst[index] = a * b[index] ** 3 - 5 * a * b[index] ** 2 + 8 * a * b[index] - 4 * a

        return dst

    dst = np.zeros((aH, aW, C), dtype=np.float32)

    sum = dst.copy()

    for i in range(1, 5):
        for j in range(1, 5):
            sum += weight(dy[i - 1]) * weight(dx[j - 1])
            index_y = np.minimum(yi + i - 2, H - 1)
            index_x = np.minimum(xi + j - 2, W - 1)
            dst += img[index_y, index_x] * weight(dy[i - 1]) * weight(dx[j - 1])

    dst /= sum

    return np.clip(dst, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('lenna.jpg')
    dst = bicubicInterpolation(img, 1.5, 1.5)
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()