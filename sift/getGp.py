import cv2 as cv
import numpy as np
import time

#降采样
def downsample(img,step = 2):
    return img[::step,::step]

#卷积
def convolve(filter,mat,padding,strides):
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:,:,i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0,mat_size[0],strides[1]):
                    temp.append([])
                    for k in range(0,mat_size[1],strides[0]):
                        val = (filter*pad_mat[j*strides[1]:j*strides[1]+filter_size[0],
                                      k*strides[0]:k*strides[0]+filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j * strides[1]:j * strides[1] + filter_size[0],
                                    k * strides[0]:k * strides[0] + filter_size[1]]).sum()
                    channel[-1].append(val)


            result = np.array(channel)

    return result

#获取高斯核
def GuassianKernel(sigma , dim):

    temp = [t - (dim//2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2*sigma*sigma
    result = (1.0/(temp*np.pi))*np.exp(-(assistant**2+(assistant.T)**2)/temp)
    return result

#获取高斯金字塔和高斯差分金字塔
def getDoG(img,n,sigma0,S = None,O = None):

    if S == None:
        S = n + 3
    if O == None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)
    sigma = [[(k**s)*sigma0*(1<<o) for s in range(S)] for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GuassianPyramid = []
    for i in range(O):
        GuassianPyramid.append([])
        for j in range(S):
            dim = int(6*sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            GuassianPyramid[-1].append(convolve(GuassianKernel(sigma[i][j], dim),samplePyramid[i],[dim//2,dim//2,dim//2,dim//2],[1,1]))
    DoG = [[GuassianPyramid[o][s+1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]
    # DoG=0

    return DoG,GuassianPyramid,samplePyramid



src = cv.imread("G:/camera/leftimg/left_calib_90.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# cv.imshow("input image",src)
SIFT_SIGMA = 1.6
SIFT_INIT_SIGMA = 0.5 # 假设的摄像头的尺度
sigma0 = np.sqrt(SIFT_SIGMA**2-SIFT_INIT_SIGMA**2)
n = 3
DoG,GuassianPyramid,samplePyramid = getDoG(gray, n,sigma0)

t=0
for i in DoG[1]:
    BGRimage = i.astype(np.uint8)
    cv.imshow("dog"+"-"+str(t),BGRimage)
    t=t+1

cv.waitKey(0)
cv.destroyAllWindows()