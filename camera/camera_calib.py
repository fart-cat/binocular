import cv2 as cv
import time

AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2 # 自动拍照间隔

cv.namedWindow("left")
cv.namedWindow("right")
cv.moveWindow("left", 0, 0)
cv.moveWindow("right", 640, 0)
right_camera = cv.VideoCapture(2)
left_camera = cv.VideoCapture(1)

counter = 145
utc = time.time()
left_folder = "G:/camera/leftimg/" # 拍照文件目录
right_folder = "G:/camera/rightimg/" # 拍照文件目录

def shot_left(pos, frame):
    global counter
    path = left_folder + pos + "_calib_" + str(counter) + ".jpg"

    cv.imwrite(path, frame)
    print("snapshot saved into: " + path)

def shot_right(pos,frame):
    global counter
    path = right_folder + pos + "_calib_" + str(counter) + ".jpg"
    cv.imwrite(path, frame)
    print("snapshot saved into: " + path)

while True:
    ret, left_frame = left_camera.read()
    ret, right_frame = right_camera.read()

    cv.imshow("left", left_frame)
    cv.imshow("right", right_frame)

    key = cv.waitKey(1)

    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot_left("left", left_frame)
        shot_right("right", right_frame)
        counter += 1
        utc = now

    if key == ord("q"):
        break
    elif key == ord("s"):
        shot_left("left", left_frame)
        shot_right("right", right_frame)
        counter += 5

left_camera.release()
right_camera.release()
cv.destroyWindow("left")
cv.destroyWindow("right")