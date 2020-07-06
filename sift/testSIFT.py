import cv2
import numpy as np

img = cv2.imread('G:/camera/leftimg/left_calib_90.jpg')
img2 = cv2.imread('G:/camera/rightimg/right_calib_90.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray, None)
kp2 = sift.detect(gray2, None)
print(len(kp))
print(len(kp2))

img = cv2.drawKeypoints(gray, kp, gray, None,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 绘制关键点
img2 = cv2.drawKeypoints(gray2, kp2, gray2, None,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 绘制关键点

cv2.imshow('left', img)
cv2.imshow('right', img2)
cv2.waitKey(0)