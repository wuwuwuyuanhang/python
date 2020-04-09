'''
@Author: wuwuwu
@Date: 2019-10-18 21:34:13
@LastEditors: wuwuwu
@LastEditTime: 2019-10-18 21:53:52
@Description: 平均池化
'''

import cv2 as cv
import numpy as np

def averagePooling(img, kernel_size=3):
    """
    平均池化，在kernel_size*kernel_size范围内的像素均值为该区域像素
    :param img: 输入图片，通道数任意
    :param kernel_size: 池化核大小
    :return: 输出图片
    """
    dst = img.copy()
    H, W, C = img.shape
    Nh = int(H / kernel_size)
    Nw = int(W / kernel_size)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                dst[y*kernel_size:(y+1)*kernel_size, x*kernel_size:(x+1)*kernel_size, c] = np.mean(
                    img[y*kernel_size:(y+1)*kernel_size, x*kernel_size:(x+1)*kernel_size, c]
                ).astype(np.uint8)

    return dst

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = averagePooling(img, kernel_size=8)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()