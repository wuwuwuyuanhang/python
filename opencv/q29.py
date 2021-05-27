# @Auther : wuwuwu 
# @Time : 2020/4/22 
# @File : q29.py
# @Description : 仿射变换之放大缩小

import cv2 as cv
import numpy as np

def afineZoom(img, a=1., b=0., c=0., d=1., tx=0., ty=0.):

    H, W, C = img.shape

    aH = int(a * H)
    dW = int(d * W)

    y = np.arange(aH).repeat(dW).reshape(dW, -1)
    x = np.tile(np.arange(dW), (aH, 1))

