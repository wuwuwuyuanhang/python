# @Auther : wuwuwu 
# @Time : 2020/4/9 
# @File : q14.py
# @Description : 差分滤波器

import cv2 as cv
import numpy as np

def BGR2GRAY(img):
    dst = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return dst.astype(np.uint8)

def differentialFilter(img):
    """
    差分滤波器，求梯度
    :param img:
    :param kernel_size:
    :return:
    """

    padding = 1
    # vertical kernel
    kernel_v = [[0., -1., 0.], [0., 1., 0.], [0., 0., 0.]]
    # horizontal kernel
    kernel_h = [[0., 0, 0.], [-1., 1., 0.], [0., 0., 0.]]

    H, W = img.shape
    # Zero Padding
    dst = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float32)
    dst[padding : H + padding, padding : W + padding] = img.copy()

    dst_v = dst.copy()
    dst_h = dst.copy()

    for y in range(H):
        for x in range(W):
            dst_v[y + padding, x + padding] = np.sum(
                dst[y : y + 3, x : x + 3] * kernel_v
            )
            dst_h[y + padding, x + padding] = np.sum(
                dst[y : y + 3, x : x + 3] * kernel_h
            )

    dst_v = np.clip(dst_v, 0, 255)
    dst_h = np.clip(dst_h, 0, 255)

    dst_v = dst_v[padding : H + padding, padding : W + padding].copy().astype(np.uint8)
    dst_h = dst_h[padding : H + padding, padding : W + padding].copy().astype(np.uint8)

    return dst_v, dst_h

if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    gray = BGR2GRAY(img)
    dst_v, dst_h = differentialFilter(gray)
    cv.imshow("input", img)
    cv.imshow("output1", dst_v)
    cv.imshow("output2", dst_h)
    cv.waitKey(0)
    cv.destroyAllWindows()