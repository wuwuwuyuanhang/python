'''
@Author: wuwuwu
@Date: 2019-10-17 20:06:23
@LastEditors: wuwuwu
@LastEditTime: 2019-10-18 21:15:46
@Description: HSV变换 将色相反转（色相值加180），然后再用RGB色彩空间表示图片
'''

import cv2 as cv
import numpy as np


def BGR2HSV(img):
    """
    BGR色彩空间转为HSV色彩空间
    :param img: 输入BGR图片
    :return: 输出HSV图片
    """
    img = img.astype(np.float32) / 255.
    Max = np.max(img, axis=2).copy()
    Min = np.min(img, axis=2).copy()
    min_arg = np.argmax(img, axis=2)
    H = np.zeros_like(Max)
    H[np.where(Min == Max)] = 0
    index = np.where(min_arg == 0)
    H[index] = 60 * (img[:,:,1][index] - img[:,:,2][index]) / (Max[index] - Min[index]) + 60
    index = np.where(min_arg == 1)
    H[index] = 60 * (img[:,:,0][index] - img[:,:,1][index]) / (Max[index] - Min[index]) + 180
    index = np.where(min_arg == 2)
    H[index] = 60 * (img[:,:,2][index] - img[:,:,0][index]) / (Max[index] - Min[index]) + 30

    S = Max.copy() - Min.copy()
    V = Max.copy()

    dst = np.dstack([H, S, V])
    return dst

def HSV2RGB(hsv):
    """
    HVS色彩空间转为RGB色彩空间
    :param hsv: 输入HSV色彩空间图片
    :return: 输出RGB色彩空间
    """
    C = hsv[:,:,1].copy()
    V = hsv[:,:,2].copy()
    H_ = hsv[:,:,0] / 60
    X = C * (1 - np.abs(H_ % 2 - 1))
    Z = np.zeros_like(H_)
    vals = [[C, X, Z], [X, C, Z], [Z, C, X], [Z, C, X], [Z, X, C], [X, Z, C], [C, Z, X]]
    dst = np.zeros_like(hsv)

    for i in range(6):
        index = np.where((i <= H_) & (H_ < i + 1))
        dst[:,:,0][index] = (V - C)[index] + vals[i][0][index]
        dst[:,:,1][index] = (V - C)[index] + vals[i][1][index]
        dst[:,:,2][index] = (V - C)[index] + vals[i][2][index]

    dst = np.clip(dst, 0, 1)
    dst = (255 * dst).astype(np.uint8)
    return dst

if __name__ == '__main__':

    img = cv.imread("lenna.jpg")
    dst = BGR2HSV(img)
    dst[:,:,0] = (dst[:,:,0] + 180) % 360
    dst1 = HSV2RGB(dst)
    cv.imshow("input", img)
    cv.imshow("HSV", dst)
    cv.imshow("output", dst1)
    cv.waitKey(0)
    cv.destroyAllWindows()