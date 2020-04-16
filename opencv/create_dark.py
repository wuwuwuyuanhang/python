# @Auther : wuwuwu 
# @Time : 2020/4/15 
# @File : q21.py
# @Description : 创造dark图

import cv2 as cv
import numpy as np

img = cv.imread('lenna.jpg')
img = img / 255. * 120
img = np.clip(img, 0, 255).astype(np.uint8)
cv.imshow('dark', img)
cv.imwrite('lenna_dark.jpg', img)
cv.waitKey(0)
cv.destroyAllWindows()