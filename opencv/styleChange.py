# @Auther : wuwuwu 
# @Time : 2021/1/28 
# @File : styleChange.py
# @Description : 利用 opencv_contrib 中的模块进行 油画 水彩 铅笔画(素描+彩色) 风格转换

import cv2 as cv

img = cv.imread('lenna.jpg')
oil_size = 0
oil_dynRatio = 1
color_sigma_s = 50
color_sigma_r = 0
pencil_sigma_s = 50
pencil_sigma_r = 0
pencil_shade_factor = 10

def oilStyle(img, size, dynRatio):
    return cv.xphoto.oilPainting(img, size, dynRatio)


def oilSizeControl(size):
    oil_size = size * 2 + 1
    oil_img = oilStyle(img, oil_size, oil_dynRatio)
    cv.imshow('oil painting', oil_img)


def colorStyle(img, sigma_s, sigma_r):
    # sigma_s 控制领域的大小，范围1~200
    # sigma_r 控制领域内不同颜色的平均方式。较大的sigma_r导致恒定颜色的较大区域，范围0~1
    return cv.stylization(img, sigma_s, sigma_r)


def colorSigmasControl(sigma_s):
    color_sigma_s = sigma_s
    color_img = colorStyle(img, color_sigma_s, color_sigma_r)
    cv.imshow('color image', color_img)


def colorSigmarControl(sigma_r):
    color_sigma_r = sigma_r / 100
    color_img = colorStyle(img, color_sigma_s, color_sigma_r)
    cv.imshow('color image', color_img)


def pencilStyle(img, sigma_s, sigma_r, shade_factor):
    # sigma_s 和 sigma_r 与上述相同
    # shade_factor 是输出图像强度的简单缩放。值越高，结果越亮。范围0~0.1。
    return cv.pencilSketch(img, sigma_s, sigma_r, shade_factor)


def pencilSigmasControl(sigma_s):
    pencil_sigma_s = sigma_s
    pencil_img_gray, pencil_img_color = pencilStyle(img, pencil_sigma_s, pencil_sigma_r, pencil_shade_factor)
    cv.imshow('gray pencil', pencil_img_gray)
    cv.imshow('color pencil', pencil_img_color)


def pencilSigmarControl(sigma_r):
    pencil_sigma_r = sigma_r / 100
    pencil_img_gray, pencil_img_color = pencilStyle(img, pencil_sigma_s, pencil_sigma_r, pencil_shade_factor)
    cv.imshow('gray pencil', pencil_img_gray)
    cv.imshow('color pencil', pencil_img_color)


def pencilShadefactorControl(shade_factor):
    pencil_shade_factor = shade_factor / 1000
    pencil_img_gray, pencil_img_color = pencilStyle(img, pencil_sigma_s, pencil_sigma_r, pencil_shade_factor)
    cv.imshow('gray pencil', pencil_img_gray)
    cv.imshow('color pencil', pencil_img_color)


if __name__ == '__main__':
    # oil_res = oilStyle(img, 7, 1)
    # color_res = colorStyle(img, 50, 0.6)
    # dst_gray, dst_color = pencilStyle(img, 60, 0.07, 0.05)

    cv.namedWindow('origin image')
    cv.namedWindow('oil painting')
    cv.namedWindow('color image')
    cv.namedWindow('gray pencil')
    cv.namedWindow('color pencil')

    cv.imshow('origin image', img)
    cv.imshow('oil painting', img)
    cv.imshow('color image', img)
    cv.imshow('gray pencil', img)
    cv.imshow('color pencil', img)

    cv.createTrackbar('size', 'oil painting', 0, 5, oilSizeControl)

    cv.createTrackbar('sigma_s', 'color image', 1, 200, colorSigmasControl)
    cv.createTrackbar('sigma_r', 'color image', 0, 100, colorSigmarControl)

    cv.createTrackbar('sigma_s', 'origin image', 1, 200, pencilSigmarControl)
    cv.createTrackbar('sigma_r', 'origin image', 0, 100, pencilSigmarControl)
    cv.createTrackbar('shade_factor', 'origin image', 0, 100, pencilShadefactorControl)

    cv.waitKey(0)
    cv.destroyAllWindows()