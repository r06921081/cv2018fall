import cv2
import numpy as np
from time import time as now
from threading import Thread
import sys

class bilateral_filter(object):
  def __init__(self):
    self.img = cv2.imread('testdata/1b.png')
    self.img = self.img/255.
    self.l = self.img.shape[0]
    self.w = self.img.shape[1]
    self.sigma_s = [1, 2, 3]
    self.sigma_r = [0.05, 0.1, 0.2]
    self.weight = [0.114, 0.587, 0.299]
    self.gray = np.array(self.weight[0] * self.img[:,:,0] + self.weight[1] * self.img[:,:,1] + self.weight[2] * self.img[:,:,2])
    self.BGRborded = None #cv2.copyMakeBorder(self.img, r, r, r, r, cv2.BORDER_REFLECT_101)
    self.gborded = None #cv2.copyMakeBorder(self.gray, r, r, r, r, cv2.BORDER_REFLECT_101)
    self.BGRdst = np.copy(self.img) * 0
    self.gdst = np.copy(self.gray) * 0
    self.f = None
    self.r = 0
    pass
  
  def grayconv(self, i, j, sig_r):
    r = self.r
    h_r = np.zeros((2*r + 1, 2*r + 1))
    tmp = np.zeros((2*r + 1, 2*r + 1))
    tmp += self.gborded[i-r:i+r+1, j-r:j+r+1] - self.gborded[i, j]
    
    h_r = np.exp(-(tmp**2) / (2 * sig_r**2))

    local_f = self.f[:,:,0] * h_r
    
    local_f = local_f / np.sum(local_f)
    
    self.gdst[i - r, j - r] = np.sum(self.gborded[i-r:i+r+1, j-r:j+r+1] * local_f)

  def BGRconv(self, i, j, sig_r):
    r = self.r
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    BGRtmp = np.zeros((2*r + 1, 2*r + 1, 3))
    
    BGRtmp += self.BGRborded[i-r:i+r+1, j-r:j+r+1] - self.BGRborded[i, j]
    h_r = np.exp((-1 * np.sum(BGRtmp**2, axis=2)) / (2 * (sig_r**2)))
    
    local_f = np.multiply(self.f, h_r[:,:, np.newaxis]) 

    local_f = local_f / np.sum(local_f, axis=(0,1))

    
    self.BGRdst[i - r, j - r] = np.sum(self.BGRborded[i-r:i+r+1, j-r:j+r+1] * local_f, axis=(0,1))
  
  def BGRbilateralfilter(self):
    begin = now()
    for sig_s in self.sigma_s[2:]:
      for sig_r in self.sigma_r[2:]:
        # sig_s = 3
        r = self.r = (sig_s * 3)
        self.BGRborded = cv2.copyMakeBorder(self.img, r, r, r, r, cv2.BORDER_REFLECT_101)
        
        kernel_size = self.kernel_size = 2 * r + 1
        self.f = np.zeros((kernel_size, kernel_size, 3))
        for i in range(-r, r + 1):
          for j in range(-r, r + 1):
            self.f[i + r, j + r] = np.exp(-(i**2 + j**2) / (2 * sig_s**2))
        for i in range(r, r + self.l):
          for j in range(r, r + self.w):
            self.BGRconv(i, j, sig_r)
    print('use', now() - begin, 'sec')
  
  def graybilateralfilter(self):
    begin = now()
    for sig_s in self.sigma_s[2:]:
      for sig_r in self.sigma_r[2:]:
        # sig_s = 3
        r = self.r = (sig_s * 3)
        self.gborded = cv2.copyMakeBorder(self.gray, r, r, r, r, cv2.BORDER_REFLECT_101)
        
        kernel_size = self.kernel_size = 2 * r + 1
        self.f = np.zeros((kernel_size, kernel_size, 3))
        for i in range(-r, r + 1):
          for j in range(-r, r + 1):
            self.f[i + r, j + r] = np.exp(-(i**2 + j**2) / (2 * sig_s**2))
        for i in range(r, r + self.l):
          for j in range(r, r + self.w):
            self.grayconv(i, j, sig_r)
    print('use', now() - begin, 'sec')
  
  def BGRshow(self):
    cv2.imshow('BGR', self.BGRdst)
    # print(self.BGRdst)

  def gshow(self):
    cv2.imshow('gray', self.gdst)

if __name__ == '__main__':
  img = bilateral_filter()
  img.BGRbilateralfilter()
  img.BGRshow()
  img.graybilateralfilter()
  img.gshow()
  cv2.waitKey(0)
  cv2.destroyAllWindows()
exit()
img = cv2.imread('testdata/1b.png')
img = img/255.
# img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)),interpolation=cv2.INTER_CUBIC)

l = img.shape[0]
w = img.shape[1]

kernel_size = 19
r = int((kernel_size - 1) / 2)
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]
weight = [0.114, 0.587, 0.299]
gray = np.array(weight[0] * R + weight[1] * G + weight[2] * B)
# weighted = np.concatenate( (np.reshape(img[:,:,0] * weight[0], img.shape[:2] + (1,)),
#                             np.reshape(img[:,:,1] * weight[1], img.shape[:2] + (1,)),
#                             np.reshape(img[:,:,2] * weight[2], img.shape[:2] + (1,)),),
#                             axis=2
#                             )


BGRborded = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101)
gborded = cv2.copyMakeBorder(gray, r, r, r, r, cv2.BORDER_REFLECT_101)
Rborded = cv2.copyMakeBorder(R, r, r, r, r, cv2.BORDER_REFLECT_101)
Gborded = cv2.copyMakeBorder(G, r, r, r, r, cv2.BORDER_REFLECT_101)
Bborded = cv2.copyMakeBorder(B, r, r, r, r, cv2.BORDER_REFLECT_101)

# l = 3
# w = 3
# cun = 0
BGRdst = np.copy(img) * 0
gdst = np.copy(gray) * 0
# dst = np.copy(gray) * 0
# Rdst = np.copy(gray) * 0
# Gdst = np.copy(gray) * 0
# Bdst = np.copy(gray) * 0
# print(dst.shape)
# exit()
# def w(x, y, sigma):
#     result = np.exp(-(x**2 + y**2) / (2 * sigma**2))
#     return result

def getfilted(src, dst, f, i, j, r):
    f = f[:,:,0].copy()
    h_r = np.zeros((2*r + 1, 2*r + 1))
    sigma = 0.2
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
    # f = f.copy()
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    sigma = 0.2
    BGRtmp = np.zeros((2*r + 1, 2*r + 1, 3))
    
    BGRtmp += src[0][i-r:i+r+1, j-r:j+r+1] - src[0][i, j]
    h_r = np.exp(-(np.sum(BGRtmp**2, axis=2)) / (2 * (sigma**2)))
    
    gf = np.multiply(f, h_r[:,:, np.newaxis]) 

    gf = gf / np.sum(gf, axis=(0,1))

    dst[0][i - r, j - r] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1] * gf, axis=(0,1))

def JBfilted(src, dst, gid, f, i, j, r, weight):
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    sigma = 3
    BGRtmp = np.zeros((2*r + 1, 2*r + 1, 3))
    
    BGRtmp += src[0][i-r:i+r+1, j-r:j+r+1] - src[0][i, j]
    h_r = np.exp(-(BGRtmp**2) / (2 * sigma**2))
    
    f = f * h_r

    f = f / np.sum(f, axis=(0,1))

    dst[0][i - r, j - r] = np.sum(src[0][i-r:i+r+1, j-r:j+r+1] * f, axis=(0,1))
    


# ths = []
# dst = [dst]
f = np.zeros((kernel_size, kernel_size, 3))
begin = now()
sigma = 3
for i in range(-r, r + 1):
    for j in range(-r, r + 1):
        # print(np.exp(-(i**2 + j**2) / (2 * sigma**2)))
        f[i + r, j + r][0] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
        f[i + r, j + r][1] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
        f[i + r, j + r][2] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
for i in range(r, r + l):
    for j in range(r, r + w):
        tmp = 0
        o = BGRborded[i-r:i+r+1, j-r:j+r+1]
        # th = Thread(target=opop, args=(o, f, i - r, j - r, [dst],))
        # th.start()
        # Rth = Thread(target=getfilted, args=([Rborded], [Rdst], f, i, j, r, ))
        # Rth.start()
        # Gth = Thread(target=getfilted, args=([Gborded], [Gdst], f, i, j, r, ))
        # Gth.start()
        # Gth = Thread(target=getfilted, args=([gborded], [gdst], f, i, j, r, ))
        # Gth.start()
        BGRgetfilted([BGRborded], [BGRdst], f, i, j, r)
        # th = Thread(target=BGRgetfilted, args=([BGRborded], [BGRdst], f, i, j, r, ))
        # th.start()
        # ths.append(th)
        # th = Thread(target=jbfilted, args=([borded], [dst], [guidance], f, i, j, r, ))
        # th.start()
        # for k_i in range(-r, r + 1):
        #     for k_j in range(-r, r + 1):
        #         # tmp = / w()
        #         # pass
        #         # tmp += borded[i + k_i, j + k_j] / (kernel_size)**2
        #         # tmp += borded[i + k_i, j + k_j] * f[k_i + r, k_j + r]
        #         # th = Thread(target=opop, args=(borded[i + k_i, j + k_j], f[k_i + r, k_j + r], tmp,))
        #         # ths.append(th)
        #         # th.start()
        #         pass


        
        # print(int(tmp))
        # np.int8(1)
        # top = np.int16(borded[i - 1, j - 1]*-1 + borded[i - 1, j + 1])
        # mid = np.int16(borded[i    , j - 1]*-2 + borded[i    , j + 1]*2)
        # but = np.int16(borded[i + 1, j - 1]*-1 + borded[i + 1, j + 1])

        # tmp = np.int16(np.abs((top+mid+but)))
        # if tmp > 255:
        #     tmp = np.uint8(255)
        # else:
        #     tmp = np.uint8(tmp)

        # print(j)
        # dst[i - r, j - r] = tmp
    for t in ths:
        t.join()
# print(cun)
print('use', now() - begin, 'sec')
# bf = np.concatenate((np.reshape(Bdst, (l, w, 1)),np.reshape(Gdst, (l, w, 1)),np.reshape(Rdst, (l, w, 1))), axis=2)
cv2.imshow('bf', BGRdst)


cv2.imshow('or', img)
print(BGRdst)
print(img)
d = BGRdst - img
# d[0,0] = [255,255,255]
cv2.imshow('d', d*255)
print(d)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

# cv2.imshow('gray', dst)

# print(l, w)
# for i in gray:
#     # print(i)
#     pass
# print(len(gray[0]))