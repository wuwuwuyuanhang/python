# @Auther : wuwuwu 
# @Time : 2020/4/11 
# @File : q19.py
# @Description : LoG滤波器

import cv2 as cv
import numpy as np

def BGR2GRAY(img):
    dst = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return dst.astype(np.uint8)

def logFilter(img, kernel_size=5, sigma=3):
    """

    :param img:
    :param kernel_size:
    :return:
    """

    H, W = img.shape

    padding = kernel_size // 2

    # kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for y in range(-padding, -padding + kernel_size):
        for x in range(-padding, -padding + kernel_size):
            kernel[y + padding, x + padding] = (x**2 + y**2 - sigma**2) / (2 * np.pi * sigma**6) * np.exp( -(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Zero Padding
    dst = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float32)

    dst[padding : H + padding, padding : W + padding] = img.copy().astype(np.float32)

    tmp = dst.copy()

    # Filter
    for y in range(H):
        for x in range(W):
            dst[y + padding, x + padding] = np.sum(
                tmp[y : y + kernel_size, x : x + kernel_size] * kernel
            )

    dst = np.clip(dst, 0, 255)

    return dst[padding : H + padding, padding : W + padding].copy().astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    gray = BGR2GRAY(img)
    dst = logFilter(gray, kernel_size=5, sigma=3)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()