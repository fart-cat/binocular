import cv2
import numpy as np
import time
import camera_configs


def getDepth(disp_avg_rec):
    depth = 1.077*(10**5)*(disp_avg_rec**2)+3.488*(10**4)*disp_avg_rec-7.833
    return depth


#获取两点对应坐标并返回符合要求的视差列表
def filter_matches(kp1, kp2, matches,area, ratio = 0.5):
    disp=[]
    for m,n in matches:
        if m.distance < n.distance * ratio:
            lp=kp1[m.queryIdx].pt#左图特征点
            if lx < lp[0] < lx + lw and ly < lp[1] < ly + lh:
                rp=kp2[m.trainIdx].pt
                d = lp[0] - rp[0]
                if (d > 0):
                    disp.append(d)
    return disp


left=cv2.imread("G:/camera/leftimg/left_calib_190.jpg")
right=cv2.imread("G:/camera/rightimg/right_calib_190.jpg")
start=time.time()
#获取图片并裁剪出ROI区域
img1_rectified = cv2.remap(left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
img1_rectified = img1_rectified[200:400,150:490]#y0:y1,x0:x1
img2_rectified = img2_rectified[200:400,150:490]#y0:y1,x0:x1

 # 3) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create(1000)
kp1, des1 = sift.detectAndCompute(img1_rectified, None)
kp2, des2 = sift.detectAndCompute(img2_rectified, None)
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)  # 代表基准图像上的一个特征点会在另一个图像上有两个匹配点

# 指定识别颜色
low_hsv = [35, 43, 46]
high_hsv = [77, 255, 255]#绿色
# 获取左图颜色范围区域

left_hsv = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
lower_hsv = np.array(low_hsv)
higher_hsv = np.array(high_hsv)
left_mask = cv2.inRange(left_hsv, lowerb=lower_hsv, upperb=higher_hsv)
#获取边界点集
colimage, left_cnts, hierarchy = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
left_cnts = max(left_cnts, key=cv2.contourArea)

lx, ly, lw, lh = cv2.boundingRect(left_cnts)
area=(lx,ly,lw,lh)
#获取落在该区域内特征点的视差列表
disp= filter_matches(kp1, kp2, matches, area)

if (len(disp)) != 0:
    disp_avg = sum(disp) / len(disp)
    depth = getDepth(1 / disp_avg);
    print("depth"+str(depth))
    flann_time = (time.time() - start)
    print("flann_time:", '% 4f' % (flann_time * 1000))
else:
    print("depth"+"null")
    flann_time = (time.time() - start)
    print("flann_time:", '% 4f' % (flann_time * 1000))

