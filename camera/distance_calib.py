import cv2
import numpy as np
import time
import camera_configs

left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(2)

def getDepth(disp_avg_rec):
    depth = 1.077*(10**5)*(disp_avg_rec**2)+3.488*(10**4)*disp_avg_rec-7.833
    return depth

def draw_min_rect(img,cnt,depth):  # conts = contours
    img = np.copy(img)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
    cv2.putText(img,str(round(depth,2)), (x+1,y+(h//2)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,255),1)
    return img

#获取点集最大轮廓面积对应的集合
def getMaxAreaContour(cnts):
    if(len(cnts)) == 0:
        return
    maxArea = float()
    maxAreaIdx = 0
    for i in range(len(cnts)):
        temprea = cv2.contourArea(cnts[i])
        if temprea>maxArea:
            maxArea = temprea
            maxAreaIdx = i
    return cnts[maxAreaIdx]

#根据颜色特征框选指定原左右图像区域
def boxByColor(left_img,right_img,low_hsv,high_hsv,depth):
    left_dst = cv2.GaussianBlur(left_img, (5, 5), 0)
    right_dst = cv2.GaussianBlur(right_img, (5, 5), 0)
    left_hsv = cv2.cvtColor(left_dst, cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right_dst, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(low_hsv)
    higher_hsv = np.array(high_hsv)
    left_mask = cv2.inRange(left_hsv, lowerb=lower_hsv, upperb=higher_hsv)
    right_mask = cv2.inRange(right_hsv, lowerb=lower_hsv, upperb=higher_hsv)
    colimage, left_cnts, hierarchy = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colimage, right_cnts, hierarchy = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_out = draw_min_rect(left_img, getMaxAreaContour(left_cnts),depth)#画出矩形并把矩形的坐标，宽高返回
    right_out = draw_min_rect(right_img, getMaxAreaContour(right_cnts),depth)#画出矩形并把矩形的坐标，宽高返回
    return left_out,right_out

#获取两点对应坐标
def filter_matches(kp1, kp2, matches, ratio = 0.5):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    return p1, p2


while True:
    ret1, left = left_camera.read()
    ret2, right = right_camera.read()
    if not ret1 or not ret2:
        break
    img1_rectified = cv2.remap(left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    # 3) SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(img1_rectified, None)
    kp2, des2 = sift.detectAndCompute(img2_rectified, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)  # 代表基准图像上的一个特征点会在另一个图像上有两个匹配点
    #获取匹配的各特征点
    p1, p2= filter_matches(kp1, kp2, matches, ratio=0.5)
    # 指定识别颜色
    low_hsv = [35, 43, 46]
    high_hsv = [77, 255, 255]
    # 获取左图校正后的框选矩形
    left_dst = cv2.GaussianBlur(img1_rectified, (5, 5), 0)
    left_hsv = cv2.cvtColor(left_dst, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(low_hsv)
    higher_hsv = np.array(high_hsv)
    left_mask = cv2.inRange(left_hsv, lowerb=lower_hsv, upperb=higher_hsv)
    colimage, left_cnts, hierarchy = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_cnts = getMaxAreaContour(left_cnts)
    lx, ly, lw, lh = cv2.boundingRect(left_cnts)
    left_area = (lx, ly, lx + lw, ly + lh)

    disp = []
    for i in range(len(p1)):
        x = p1[i][0]
        y = p1[i][1]
        if left_area[0] < x < left_area[2] and left_area[1] < y < left_area[3]:
            d = (p1[i])[0] - (p2[i])[0]
            if (d > 0):
                disp.append(d)
    if (len(disp)) != 0:
        disp_avg = sum(disp) / len(disp)
        depth = getDepth(1 / disp_avg);
        print("depth"+str(depth))
    else:
        print("depth"+"null")

