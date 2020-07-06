import cv2
import numpy as np
import time
import camera_configs


def draw_min_rect(img, cnt):  # conts = contours
    img = np.copy(img)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
    return img

#获取点集最大轮廓面积对应的集合
def getMaxAreaContour(cnts):
    maxArea = float()
    maxAreaIdx = 0
    for i in range(len(cnts)):
        temprea = cv2.contourArea(cnts[i])
        if temprea>maxArea:
            maxArea = temprea
            maxAreaIdx = i
    return cnts[maxAreaIdx]

#获取两点对应坐标
def filter_matches(kp1, kp2, matches, ratio = 0.5):
    #匹配优化
    mkp1, mkp2 = [], []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    return p1, p2

#读取图片
index = 15
disp_avg = []
disp_avg_rec = []
depth = []
while True:
    if index == 200:
        break
    left = cv2.imread("G:/camera/leftimg/left_calib_" + str(index) + ".jpg")
    right = cv2.imread("G:/camera/rightimg/right_calib_" + str(index) + ".jpg")
    # 校正
    img1_rectified = cv2.remap(left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    # 3) SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(img1_rectified, None)
    kp2, des2 = sift.detectAndCompute(img2_rectified, None)

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    p1, p2= filter_matches(kp1, kp2, matches, ratio=0.5)

    # 指定识别颜色
    low_hsv = [35, 43, 46]
    high_hsv = [77, 255, 255]
    # 获取左图校正后的框选矩形
    left_hsv = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(low_hsv)
    higher_hsv = np.array(high_hsv)
    left_mask = cv2.inRange(left_hsv, lowerb=lower_hsv, upperb=higher_hsv)
    colimage, left_cnts, hierarchy = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_cnts = getMaxAreaContour(left_cnts)#获取最大点集
    lx, ly, lw, lh = cv2.boundingRect(left_cnts)
    left_area = (lx, ly, lx + lw, ly + lh)

    disp = []
    # print(len(p1))
    # 筛选出在该区域的匹配点
    for i in range(len(p1)):
        x = p1[i][0]
        y = p1[i][1]
        if left_area[0] < x < left_area[2] and left_area[1] < y < left_area[3]:
            d = (p1[i])[0] - (p2[i])[0]
            if (d > 0):
                disp.append(d)
    # 求该区域内的横坐标之差平均
    if len(disp)!=0:
        avg = sum(disp) / len(disp)
        disp_avg.append(avg)
        disp_avg_rec.append(1/avg)
        depth.append(index*10)
    index+=5
print(disp_avg)
print(disp_avg_rec)
print(depth)
print(len(depth))
