import cv2 as cv
import time
import os

AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 5 # 自动拍照间隔

left_camera = cv.VideoCapture(0)
right_camera = cv.VideoCapture(2)

counter = 1
utc = time.time()
left_folder = os.getcwd()+"/" # 拍照文件目录
right_folder = os.getcwd()+"/"# 拍照文件目录

def shot_left(pos, frame):
    global counter
    path = left_folder+pos + "_" + str(counter) + ".jpg"

    cv.imwrite(path, frame)
    print("snapshot saved into: " + path)

def shot_right(pos,frame):
    global counter
    path = right_folder + pos + "_" + str(counter) + ".jpg"
    cv.imwrite(path, frame)
    print("snapshot saved into: " + path)

ret, left_frame = left_camera.read()
ret, right_frame = right_camera.read()
shot_left("left", left_frame)
shot_right("right", right_frame)

left_camera.release()
right_camera.release()
