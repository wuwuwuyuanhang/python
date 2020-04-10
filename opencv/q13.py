# @Auther : wuwuwu 
# @Time : 2020/4/9 
# @File : q13.py
# @Description : MAX-MIN滤波器

import cv2 as cv
import numpy as np

def BGR2GRAY(img):
    dst = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return dst.astype(np.float32)

def maxMinFilter(img, kernel_size=3):
    """
    MAX-MIN滤波器使用网格内像素的最大值和最小值的差值对网格内像素重新赋值
    :param img: 输入图片
    :param kernel_size:
    :return:输出图片
    """

    H, W = img.shape
    padding = kernel_size // 2
    # Zero Padding
    dst = np.zeros((H + padding * 2, W + padding * 2), dtype=np.uint8)
    dst[padding : H + padding, padding : W + padding] = img.copy()

    tmp = dst.copy()

    # Max-Min Filter
    for y in range(H):
        for x in range(W):
            dst[y + padding, x + padding] = np.max(tmp[y : y + kernel_size, x : x + kernel_size]) - \
                                            np.min(tmp[y : y + kernel_size, x : x + kernel_size])

    dst = dst[padding: H + padding, padding: W + padding].copy()
    dst = np.clip(dst, 0, 255)

    return dst

if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    gray = BGR2GRAY(img)
    dst = maxMinFilter(gray, kernel_size=3)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()