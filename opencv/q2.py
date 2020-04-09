'''
@Author: wuwuwu
@Date: 2019-10-16 22:49:04
@LastEditors: wuwuwu
@LastEditTime: 2019-10-17 19:51:03
@Description: 灰度化
'''

import cv2 as cv
import numpy as np

def BGR2GRAY(img):
    """
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    :param img: 输入图片，通道为BGR
    :return: 输出图片，为灰度图
    """
    dst = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    return dst.astype(np.uint8)

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = BGR2GRAY(img)

    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()