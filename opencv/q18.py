# @Auther : wuwuwu 
# @Time : 2020/4/11 
# @File : q18.py
# @Description : Emboss

import cv2 as cv
import numpy as np

def BGR2GRAY(img):
    dst = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return dst.astype(np.uint8)

def embossFilter(img):
    """
    Laplacian Filter
    :param img:
    :return:
    """

    H, W = img.shape

    padding = 1

    # kernel
    kernel = [[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]]

    # Zero Padding

    dst = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float32)
    dst[padding: H + padding, padding: W + padding] = img.copy().astype(np.float32)

    tmp = dst.copy()

    for y in range(H):
        for x in range(W):
            dst[y + padding, x + padding] = np.sum(
                tmp[y: y + 3, x: x + 3] * kernel
            )
    dst = np.clip(dst, 0, 255)

    dst = dst[padding: H + padding, padding: W + padding].copy().astype(np.uint8)

    return dst


if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    gray = BGR2GRAY(img)
    dst = embossFilter(gray)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()