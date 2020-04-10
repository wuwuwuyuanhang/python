# @Auther : wuwuwu 
# @Time : 2020/4/9 
# @File : q12.py
# @Description : Motion 滤波

import cv2 as cv
import numpy as np

def motionFilter(img, kernel_size=3):
    """
    Motion Filter 取对角线方向的像素平均值
    :param img: 输入图片
    :param kernel_size: 卷积和尺寸
    :return: 输出图片
    """
    kernel = np.eye(kernel_size, dtype=np.float32)
    kernel /= kernel.sum()

    H, W, C = img.shape
    padding = kernel_size // 2

    # Zero Padding
    dst = np.zeros((H + padding * 2, W + padding * 2, C), dtype=np.float32)
    dst[padding : H + padding, padding : W + padding] = img.copy().astype(np.float32)

    tmp = dst.copy()

    # Motion Filter
    for y in range(H):
        for x in range(W):
            for c in range(C):
                dst[y + padding, x + padding, c] = np.sum(
                    tmp[y : y + kernel_size, x : x + kernel_size, c] * kernel
                )

    dst = dst[padding : H + padding, padding : W + padding].copy()
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    return dst

if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    dst = motionFilter(img, kernel_size=3)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()