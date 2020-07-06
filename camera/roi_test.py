import cv2 as cv
import time

img = cv.imread("G:/camera/leftimg/left_calib_50.jpg")
img_r = img[200:450,200:440]#y0:y1,x0:x1
cv.imshow("left",img_r)
cv.imshow("org",img)
cv.waitKey(0)
cv.destroyAllWindows()