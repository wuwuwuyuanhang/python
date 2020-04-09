'''
@Author: wuwuwu
@Date: 2019-12-01 16:43:50
@LastEditors: wuwuwu
@LastEditTime: 2019-12-01 16:50:46
@Description: 均值滤波
'''

import cv2 as cv
import numpy as np

def meanFilter(img, kernel_size=3):
    """
    均值滤波
    :param img: 输入图片
    :param kernel_size: 卷积和尺寸
    :return: 输出图片
    """
    H, W, C = img.shape

    padding = kernel_size // 2

    # Zero Padding
    dst = np.zeros((H + padding * 2, W + padding * 2, C), dtype=np.float32)
    dst[padding : H + padding, padding : W + padding] = img.copy().astype(np.float32)

    tmp = dst.copy()

    # Mean Filter
    for y in range(H):
        for x in range(W):
            for c in range(C):
                dst[y + padding, x + padding, c] = np.mean(
                    tmp[y : y + kernel_size, x : x + kernel_size, c]
                )

    dst = np.clip(dst, 0, 255)
    dst = dst[padding : H + padding, padding : W + padding].copy().astype(np.uint8)

    return dst

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = meanFilter(img, kernel_size=3)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()