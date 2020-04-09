'''
@Author: wuwuwu
@Date: 2019-10-20 20:18:36
@LastEditors: wuwuwu
@LastEditTime: 2019-10-28 22:07:55
@Description: 高斯滤波
'''

import cv2 as cv
import numpy as np

def gaussianFilter(img, kernel_size=3, sigma=1.3):
    """
    高斯滤波
    :param img: 输入图片
    :param kernel_size: 高斯卷积核大小
    :param sigma: 高斯分布标准差
    :return: 输出图片
    """
    padding = kernel_size // 2
    # Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for y in range(-padding, -padding+kernel_size):
        for x in range(-padding, -padding+kernel_size):
            kernel[y+padding, x+padding] = np.exp( -(y**2 + x**2)/(2*sigma**2) ) / (2 * np.pi * sigma ** 2)
    kernel /= kernel.sum()

    # Zero Padding
    H, W, C = img.shape
    dst = np.zeros((H+padding*2, W+padding*2, C), dtype=np.float32)
    dst[padding:padding+H, padding:padding+W] = img.copy().astype(np.float32)

    tmp = dst.copy()

    # Gaussian Filter
    for y in range(H):
        for x in range(W):
            for c in range(C):
                dst[y+padding, x+padding, c] = np.sum(
                    tmp[y:y+kernel_size, x:x+kernel_size, c] * kernel
                )

    dst = np.clip(dst, 0, 255)
    dst = dst[padding:padding+H, padding:padding+W].astype(np.uint8)

    return dst

if __name__ == '__main__':

    img = cv.imread("lenna_noise.jpg")
    dst = gaussianFilter(img, kernel_size=3, sigma=1.3)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()