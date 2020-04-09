'''
@Author: wuwuwu
@Date: 2019-10-28 21:47:11
@LastEditors: wuwuwu
@LastEditTime: 2019-12-01 16:39:27
@Description: 中值滤波
'''

import numpy as np
import cv2 as cv

def medianFilter(img, kernel_size=3):
    """
    中值滤波
    :param img: 输入图片
    :param kernel_size: 平滑范围
    :return: 输出图片
    """
    H, W, C = img.shape

    padding = kernel_size // 2

    # Zero Padding
    dst = np.zeros((H + padding * 2, W + padding * 2, C), dtype=np.uint8)
    dst[padding : H + padding, padding : W + padding] = img.copy()

    tmp = dst.copy()

    # Median Filter
    for y in range(H):
        for x in range(W):
            for c in range(C):
                dst[y + padding, x + padding, c] = np.median(
                    tmp[y : y + kernel_size, x : x + kernel_size, c]
                )

    return dst

if __name__ == '__main__':

    img = cv.imread("lenna_noise.jpg")
    dst = medianFilter(img, kernel_size=3)
    # cv.medianBlur(img, ksize=3, dst)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()