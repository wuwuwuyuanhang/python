'''
@Author: wuwuwu
@Date: 2019-10-20 19:54:19
@LastEditors: wuwuwu
@LastEditTime: 2019-10-20 21:04:57
@Description: 制作噪声图
'''

import cv2 as cv
import numpy as np

img = cv.imread("lenna.jpg")

H, W, _ = img.shape

for i in range(100):
    y = np.random.randint(0, H)
    x = np.random.randint(0, W)
    cv.circle(img, (x, y), 1, (0, 0, 0), -1, 8, 0)

cv.imshow("result", img)
cv.imwrite("lenna_noise.jpg", img)
cv.waitKey(0)
cv.destroyAllWindows