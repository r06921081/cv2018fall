import cv2
import numpy as np
from time import time as now
from threading import Thread
import sys

img = cv2.imread('testdata/1b.png')
img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)),interpolation=cv2.INTER_CUBIC)
l = img.shape[0]
w = img.shape[1]


B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]
weight = [0.1, 0.2 , 0.7 ]
gray = np.array(weight[0] * R + weight[1] * G + weight[2] * B, dtype=np.uint8)
weighted = np.concatenate( (np.reshape(img[:,:,0] * weight[0], img.shape[:2] + (1,)),
                            np.reshape(img[:,:,1] * weight[1], img.shape[:2] + (1,)),
                            np.reshape(img[:,:,2] * weight[2], img.shape[:2] + (1,)),),
                            axis=2
                            )


BGRborded = None# cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101)
gidborded = None# cv2.copyMakeBorder(gray, r, r, r, r, cv2.BORDER_REFLECT_101)

BGRdst = np.copy(img) * 0
dst = np.copy(gray) * 0
Rdst = np.copy(gray) * 0
Gdst = np.copy(gray) * 0
Bdst = np.copy(gray) * 0



def getfilted(src, dst, f, i, j, r):
    h_r = np.zeros((2*r + 1, 2*r + 1))
    sigma = 3
    tmp = np.zeros((2*r + 1, 2*r + 1))
    tmp += src[0][i-r:i+r+1, j-r:j+r+1] - src[0][i, j]
    
    h_r = np.exp(-(tmp**2) / (2 * sigma**2))
    # print(rrr)
    # for ii in range(-r, r + 1):
    #     for jj in range(-r, r + 1):
    #         h_r[ii + r, jj + r] = np.exp(-(src[0][i + ii, j + jj] - src[0][i, j])**2 / (2 * sigma**2))
    #         # h_r[ii + r, jj + r] = np.exp(-(tmp)**2 / (2 * sigma**2))
    #         # tmp = np.exp(-(src[0][i + ii, j + jj] - src[0][i, j])**2 / (2 * sigma**2))
    #         # h_r = tmp
    # print(np.sum(h_r - rrr))
    # print(np.sum(rrr))
    # print(tmp)

    f = f * h_r

    fsum = np.sum(f)
    # if sumf > 0:
    # # print(sumf)
    f = f / fsum

    # print(f)
    dst[0][i - r, j - r] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1] * f)

def BGRgetfilted(src, dst, f, i, j, r):
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    sigma = 3
    BGRtmp = np.zeros((2*r + 1, 2*r + 1, 3))
    
    BGRtmp += src[0][i-r:i+r+1, j-r:j+r+1] - src[0][i, j]
    h_r = np.exp(-(BGRtmp**2) / (2 * sigma**2))
    
    f = f * h_r

    f = f / np.sum(f, axis=(0,1))

    dst[0][i - r, j - r] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1] * f, axis=(0,1))

def JBfilted(src, dst, gid, f, i, j, r, sigma):
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    # sigma = 3
    tmp = np.zeros((2*r + 1, 2*r + 1,3))
    
    # tmp += gid[0][i-r:i+r+1, j-r:j+r+1] - gid[0][i, j] 
    
    # h_r = np.exp(-(tmp**2) / (2 * sigma**2))
    tmp[:,:,0] += gid[0][i-r:i+r+1, j-r:j+r+1] - gid[0][i, j]
    tmp[:,:,1] += gid[0][i-r:i+r+1, j-r:j+r+1] - gid[0][i, j]
    tmp[:,:,2] += gid[0][i-r:i+r+1, j-r:j+r+1] - gid[0][i, j]
    h_r = np.exp(-(tmp[:,:,0]**2 + tmp[:,:,1]**2 + tmp[:,:,2]**2) / (2 * sigma**2))
    # print(h_r.shape)
    
    f = f[:,:,0] * h_r
    # f = f * np.concatenate((np.reshape(h_r, h_r.shape + (1,)),np.reshape(h_r, h_r.shape + (1,)),np.reshape(h_r, h_r.shape + (1,))), axis=2)

    f = f / np.sum(f)
    
    dst[0][i - r, j - r][0] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1,0] * f)
    dst[0][i - r, j - r][1] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1,1] * f)
    dst[0][i - r, j - r][2] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1,2] * f)
    # if dst[0][i - r, j - r,0] > 255 or dst[0][i - r, j - r,1] > 255 or dst[0][i - r, j - r,2]>255 or dst[0][i - r, j - r,0] <0 or dst[0][i - r, j - r,1] <0 or dst[0][i - r, j - r,2] <0 :
    #     print('dddd')
    

# ths = []
# dst = [dst]
pics = []
begin = now()
for sigma_s in [1]:
    kernel_size = (sigma_s * 3) * 2 + 1
    r = int((kernel_size - 1) / 2)
    if BGRborded is None or gidborded is None:
        BGRborded = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101)
        gidborded = cv2.copyMakeBorder(gray, r, r, r, r, cv2.BORDER_REFLECT_101)
    f = np.zeros((kernel_size, kernel_size, 3))
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            # print(np.exp(-(i**2 + j**2) / (2 * sigma**2)))
            f[i + r, j + r][0] = np.exp(-(i**2 + j**2) / (2 * sigma_s**2))
            f[i + r, j + r][1] = np.exp(-(i**2 + j**2) / (2 * sigma_s**2))
            f[i + r, j + r][2] = np.exp(-(i**2 + j**2) / (2 * sigma_s**2))

    for sigma_r in [0.2]:
        for i in range(r, r + l):
            for j in range(r, r + w):
                tmp = 0
                o = BGRborded[i-r:i+r+1, j-r:j+r+1]
                
                th = Thread(target=JBfilted, args=([BGRborded], [BGRdst], [gidborded], f, i, j, r, sigma_r,))
                th.start()
        
        pics.append([str(sigma_s) + '-' + str(sigma_r),BGRdst.copy()])
        
       
# print(cun)
print('use', now() - begin, 'sec')
# bf = np.concatenate((np.reshape(Bdst, (l, w, 1)),np.reshape(Gdst, (l, w, 1)),np.reshape(Rdst, (l, w, 1))), axis=2)
# cv2.imshow('bf', BGRdst)

# for name, p in pics:
#     cv2.imshow(name, p)
cv2.imshow('gray', BGRdst)
cv2.imshow('or', img)
cv2.imshow('g', gray)
cv2.imshow('d', np.abs(np.abs(img - BGRdst)))
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

cv2.imshow('gray', dst)

print(l, w)
for i in gray:
    # print(i)
    pass
print(len(gray[0]))