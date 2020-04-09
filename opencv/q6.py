'''
@Author: wuwuwu
@Date: 2019-10-18 21:23:59
@LastEditors: wuwuwu
@LastEditTime: 2019-10-18 21:33:49
@Description: 减色处理
'''

import cv2 as cv
import numpy as np

def dicreaseColor(img):
    """
    减色处理，8位量化图片转为2位量化，取值为32，96，160，244
    :param img: 输入图片
    :return: 输出图片
    """
    return img // 64 * 64 + 32

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = dicreaseColor(img)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()