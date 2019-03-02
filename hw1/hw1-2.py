import cv2
import numpy as np
from time import time as now
from threading import Thread
import sys

class bilateral_filter(object):
  def __init__(self, filepath, sigma_s, sigma_r):
    self.img = cv2.imread(filepath)
    self.img = self.img/255.
    self.l = self.img.shape[0]
    self.w = self.img.shape[1]
    self.sigma_s = sigma_s
    self.sigma_r = sigma_r
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
    
    BGRtmp += np.repeat(np.reshape(self.gborded[i-r:i+r+1, j-r:j+r+1] - self.gborded[i, j], (self.kernel_size, self.kernel_size, 1)), 3, axis=2)      
    h_r = np.exp((-1 * np.sum(BGRtmp**2, axis=2)) / (2 * (sig_r**2)))
    
    local_f = np.multiply(self.f, h_r[:,:, np.newaxis]) 

    local_f = local_f / np.sum(local_f, axis=(0,1))

    
    self.BGRdst[i - r, j - r] = np.sum(self.BGRborded[i-r:i+r+1, j-r:j+r+1] * local_f, axis=(0,1))
  
  def BGRbilateralfilter(self):
    begin = now()
    for sig_s in [self.sigma_s]:
      for sig_r in [self.sigma_r]:
        # sig_s = 3
        r = self.r = (sig_s * 3)
        self.BGRborded = cv2.copyMakeBorder(self.img, r, r, r, r, cv2.BORDER_REFLECT_101)
        self.gborded = cv2.copyMakeBorder(self.gray, r, r, r, r, cv2.BORDER_REFLECT_101)
        
        kernel_size = self.kernel_size = 2 * r + 1
        self.f = np.zeros((kernel_size, kernel_size, 3))
        for i in range(-r, r + 1):
          for j in range(-r, r + 1):
            self.f[i + r, j + r] = np.exp(-(i**2 + j**2) / (2 * sig_s**2))
        for i in range(r, r + self.l):
          for j in range(r, r + self.w):
            self.BGRconv(i, j, sig_r)
    print('use', now() - begin, 'sec')
    return self.BGRdst
  
  def graybilateralfilter(self):
    begin = now()
    for sig_s in [self.sigma_s]:
      for sig_r in [self.sigma_r]:
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
  img = bilateral_filter(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
  result = img.BGRbilateralfilter()
  cv2.imwrite("./result/jbf_" + sys.argv[2] + ',' + sys.argv[3] + sys.argv[1].split('/')[-1], result * 255)
  # img.BGRshow()
  # # img.graybilateralfilter()
  # # img.gshow()
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
