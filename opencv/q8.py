'''
@Author: wuwuwu
@Date: 2019-10-20 19:41:29
@LastEditors: wuwuwu
@LastEditTime: 2019-10-20 19:48:45
@Description: 最大池化
'''

import cv2 as cv
import numpy as np

def maxPooling(img, kernel_size=3):
    """
    最大池化，在kernel_size*kernel_size范围内的像素最大值为该区域像素
    :param img: 输入图片
    :param kernel_size: 池化核大小
    :return: 输出图片
    """
    dst = img.copy()
    H, W, C = img.shape
    Nh = H // kernel_size
    Nw = W // kernel_size

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                dst[y*kernel_size:(y+1)*kernel_size, x*kernel_size:(x+1)*kernel_size, c] = np.max(
                    img[y*kernel_size:(y+1)*kernel_size, x*kernel_size:(x+1)*kernel_size, c]
                ).astype(np.uint8)

    return dst

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = maxPooling(img, kernel_size=8)
    cv.imshow("input", img)
    cv.imshow("output", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()