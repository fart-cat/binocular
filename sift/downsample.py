import cv2 as cv
import numpy as np
import time

# 高斯金字塔
def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramids_images = []  # 空列表
    for i in range(level):
        dst = cv.pyrDown(temp)  # 先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
        pyramids_images.append(dst)  # 在列表末尾添加新的对象
        cv.imshow("pyramid_down_"+str(i), dst)
        temp = dst.copy()
    return pyramids_images

img = cv.imread("G:/camera/leftimg/left_calib_50.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", gray)
pyramid_demo(gray)
cv.waitKey(0)
cv.destroyAllWindows()