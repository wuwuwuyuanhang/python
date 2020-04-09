'''
@Author: wuwuwu
@Date: 2019-10-16 21:31:53
@LastEditors: wuwuwu
@LastEditTime: 2019-10-17 19:50:44
@Description: 通道交换
'''

import cv2 as cv
import numpy as np

def BGR2RGB(img):
    """
    cv2.imread()函数读取图片是按照BGR顺序读取的
    将BGR通道顺序改为RGB顺序
    :param img: 输入图片，通道为BGR
    :return: 输出图片，通道为RGB
    """
    b, g, r = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    return np.dstack([r, g, b])

if __name__ == '__main__':
    img = cv.imread("lenna.jpg")
    dst = BGR2RGB(img)

    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()