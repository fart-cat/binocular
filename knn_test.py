import cv2
import numpy as np
import time
import camera_configs


# 1) 打开图像
img1 = cv2.imread("G:/camera/left_1.jpg")
img2 = cv2.imread("G:/camera/left_1.jpg")

img1_rectified = cv2.remap(img1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(img2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
img1_rectified = img1_rectified[200:400,200:440]#y0:y1,x0:x1
img2_rectified = img2_rectified[200:400,200:440]#y0:y1,x0:x1


# 3) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create(1000)

kp1, des1 = sift.detectAndCompute(img1_rectified, None)
kp2, des2 = sift.detectAndCompute(img2_rectified, None)


# 设置算法运行开始时间
start = time.time()
# 4) Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# print(matches)
goodMatch = []
# 匹配优化
for m, n in matches:
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
flann_img_out = cv2.drawMatchesKnn(img1_rectified, kp1, img2_rectified, kp2, goodMatch, None, flags=2)
flann_time = (time.time() - start)
print("flann_time:", '% 4f' % (flann_time*1000))

cv2.imshow('flannmatch', flann_img_out)#展示图片
cv2.waitKey(0)
cv2.destroyAllWindows()
