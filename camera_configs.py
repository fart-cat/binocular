# filename: camera_configs.py
import cv2
import numpy as np

#双目相机标定主要是为了获得摄像头的
# 内参(f,1/dx,1/dy,cx,cy)、
# 畸变参数(k1,k2,k3,p1,p2)和外参(R,t)，
# 用于接下来的双目校正和深度图生成。
left_camera_matrix = np.array([[987.85030,-28.70530,194.83777],[0,1012.80311,427.19180],[0,0,1]])
left_distortion = np.array([[-0.21856,0.27954,0.03674,0.00119,0.00000]])

right_camera_matrix = np.array([[983.89385,-11.60923,240.51926],[0,1007.58179,449.02209],[0,0,1]])
right_distortion = np.array([[-0.33169,0.53157,0.02329,0.012363,0.00000]])

R=np.array([[0.99996,-0.00763,-0.00505],[0.00758,0.99992,-0.01007],[0.00513,0.01003,0.99994]])#旋转矩阵
T = np.array([-31.95957,0.25588,1.38946])#平移矩阵

size = (640,480)  #open windows size
R1,R2,P1,P2,Q,validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix,right_distortion,size,R,T)
left_map1,left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,left_distortion,R1,P1,size,cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,right_distortion,R2,P2,size,cv2.CV_16SC2)
# left_camera_matrix = np.array([[0.99985,-0.00838,-0.01487],[0.00879,0.99956,0.02822],[0.01462,-0.02835,0.99949]])
# T = np.array([-29.98908,-0.17384,1.72763])
