import cv2
import numpy as np
import time
import camera_configs
import RPi.GPIO as gpio


#初始化操作
IN_LIST =[7,11,13,15,35,37]
ENA=35
ENB=37
gpio.setmode(gpio.BOARD)
gpio.setup(IN_LIST, gpio.OUT)
pwmA = gpio.PWM(ENA, 500)
pwmB = gpio.PWM(ENB, 500)
pwmA.start(0)
pwmB.start(0)
#定义当前树莓派状态
state='s'
#初始化SIFT算子
sift = cv2.xfeatures2d.SIFT_create(1000)
#读取摄像头视频流
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(2)
left_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
right_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
left_camera.set(cv2.CAP_PROP_FPS,2)
right_camera.set(cv2.CAP_PROP_FPS,2)

#最大速度
def speed_up(IN_LIST,pwmA,pwmB):
    global state
    pwmA.ChangeDutyCycle(70)
    pwmB.ChangeDutyCycle(70)
    gpio.output(IN_LIST[0], gpio.LOW)
    gpio.output(IN_LIST[1], gpio.HIGH)
    gpio.output(IN_LIST[2], gpio.LOW)
    gpio.output(IN_LIST[3], gpio.HIGH)
    state='u'

#减速
def speed_down(IN_LIST,pwmA,pwmB):
    global state
    pwmA.ChangeDutyCycle(40)
    pwmB.ChangeDutyCycle(40)
    gpio.output(IN_LIST[0], gpio.LOW)
    gpio.output(IN_LIST[1], gpio.HIGH)
    gpio.output(IN_LIST[2], gpio.LOW)
    gpio.output(IN_LIST[3], gpio.HIGH)
    state='d'

def stop(IN_LIST,pwmA,pwmB):
    global state
    gpio.output(IN_LIST[0], gpio.LOW)
    gpio.output(IN_LIST[1], gpio.LOW)
    gpio.output(IN_LIST[2], gpio.LOW)
    gpio.output(IN_LIST[3], gpio.LOW)
    state='s'

def getDepth(disp_avg_rec):
    depth = 1.077*(10**5)*(disp_avg_rec**2)+3.488*(10**4)*disp_avg_rec-7.833
    return depth

#获取两点对应坐标并返回符合要求的视差列表
def filter_matches(kp1, kp2, matches,area, ratio = 0.5):
    disp=[]
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            lp = kp1[m.queryIdx].pt  # 左图特征点
            if lx < lp[0] < lx + lw and ly < lp[1] < ly + lh:
                rp = kp2[m.trainIdx].pt
                d = lp[0] - rp[0]
                if (d > 0):
                    disp.append(d)
    return disp


while True:
    #跳帧处理
    start=time.time();
    for i in range(6):
        ret1, left = left_camera.read()
        ret2, right = right_camera.read()
    # ret1, left = left_camera.read()
    # ret2, right = right_camera.read()
    if not ret1 or not ret2:
        stop(IN_LIST, pwmA, pwmB)
        break
    img1_rectified = cv2.remap(left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    img1_rectified = img1_rectified[200:450,200:440]  # y0:y1,x0:x1
    img2_rectified = img2_rectified[200:450,200:440]  # y0:y1,x0:x1
    img1_gray = cv2.cvtColor(img1_rectified, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.cvtColor(img2_rectified, cv2.IMREAD_GRAYSCALE)
    # 3) SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    if des1 is None or des2 is None:
        stop(IN_LIST, pwmA, pwmB)
        flann_time = (time.time() - start)
        print("当前状态：停止*" + "当前距离：null" )
        print("处理时间：", '% 4f' % (flann_time * 1000))
        continue
    matches = bf.knnMatch(des1, des2, k=2)  # 代表基准图像上的一个特征点会在另一个图像上有两个匹配点
    # 指定识别颜色
    low_hsv = [35, 43, 46]
    high_hsv = [77, 255, 255]  # 绿色

    left_hsv = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(low_hsv)
    higher_hsv = np.array(high_hsv)
    left_mask = cv2.inRange(left_hsv, lowerb=lower_hsv, upperb=higher_hsv)
    # 获取边界点集
    colimage, left_cnts, hierarchy = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(len(left_cnts))==0:
        #没有颜色区域
        speed_up(IN_LIST, pwmA, pwmB)
        flann_time = (time.time() - start)
        print("当前状态：加速*" + "当前距离：null" )
        print("处理时间：", '% 4f' % (flann_time * 1000))
        continue
    left_cnts = max(left_cnts, key=cv2.contourArea)
    lx, ly, lw, lh = cv2.boundingRect(left_cnts)
    area = (lx, ly, lw, lh)
    # 获取落在该区域内特征点的视差列表
    disp = filter_matches(kp1, kp2, matches, area)
    depth=0
    if (len(disp)) != 0:
        disp_avg = sum(disp) / len(disp)
        depth = getDepth(1 / disp_avg);

    if(depth>=1200 or depth<=0):
        if state=='u':
            flann_time = (time.time() - start)
            print("当前状态：加速*" +"当前距离："+ str(depth))
            print("处理时间：", '% 4f' % (flann_time * 1000))
            continue
        speed_up(IN_LIST, pwmA, pwmB)
        flann_time = (time.time() - start)
        print("当前状态：加速*" + "当前距离：" + str(depth))
        print("处理时间：", '% 4f' % (flann_time * 1000))
    elif(depth>500 and depth<1200):
        if state=='d':
            flann_time = (time.time() - start)
            print("当前状态：减速*" + "当前距离：" + str(depth))
            print("处理时间：", '% 4f' % (flann_time * 1000))
            continue
        #减速
        speed_down(IN_LIST, pwmA, pwmB)
        flann_time = (time.time() - start)
        print("当前状态：减速*" + "当前距离：" + str(depth))
        print("处理时间：", '% 4f' % (flann_time * 1000))

    else:
        if state=='s':
            flann_time = (time.time() - start)
            print("当前状态：停止*" + "当前距离：" + str(depth))
            print("处理时间：", '% 4f' % (flann_time * 1000))
            continue
        stop(IN_LIST, pwmA, pwmB)
        flann_time = (time.time() - start)
        print("当前状态：停止*" + "当前距离：" + str(depth))
        print("处理时间：", '% 4f' % (flann_time * 1000))

pwmA.stop()
pwmB.stop()
gpio.cleanup()