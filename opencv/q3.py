'''
@Author: wuwuwu
@Date: 2019-10-16 22:55:23
@LastEditors: wuwuwu
@LastEditTime: 2019-10-17 18:34:25
@Description: 二值化
'''

import cv2 as cv
import numpy as np

def Thresholding(img, th=128):
    """
    大于等于阈值像素为255，小于为0
    :param img: 输入图片，可以为BGR，也可为单通道灰度图
    :param th: 阈值
    :return: 二值化图片
    """
    gray = img.copy()
    _, _, C = gray.shape
    if C == 3:
        gray = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    thresholding = gray.astype(np.uint8)
    thresholding[gray < th] = 0
    thresholding[gray >= th] = 255
    return thresholding

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    thresholding = Thresholding(img, 128)

    cv.imshow("input", img)
    cv.imshow("output", thresholding)
    cv.waitKey(0)
    cv.destroyAllWindows()