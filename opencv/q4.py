'''
@Author: wuwuwu
@Date: 2019-10-17 18:37:01
@LastEditors: wuwuwu
@LastEditTime: 2019-10-17 20:02:44
@Description: 大津二值化算法
'''

import cv2 as cv
import numpy as np

def otsuMethod(img):
    """
    大津阈值法
    :param img: 输入图片
    :return: 二值化图片
    """
    gray = img.copy()
    H, W, C = gray.shape
    if C == 3:
        gray = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]
        gray = gray.astype(np.uint8)

    max_sigma = 0
    max_t = 0

    for _t in range(1, 255):
        v0 = gray[np.where(gray < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0
        w0 = len(v0) / (H * W)

        v1 = gray[np.where(gray >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0
        w1 = len(v1) / (H * W)

        sigma = w0 * w1 * (m1 - m0) ** 2

        if sigma > max_sigma:
            max_sigma, max_t = sigma, _t

    print("阈值:{}".format(max_t))

    gray[gray < max_t] = 0
    gray[gray >= max_t] = 255

    return gray

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = otsuMethod(img)

    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()