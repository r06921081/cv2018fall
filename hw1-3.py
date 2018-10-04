import cv2
import numpy as np
from time import time as now
import time
from threading import Thread
from multiprocessing import Process
import sys

class bilateral_filter(object):
  def __init__(self, sigma_s, sigma_r, weight = (0.114, 0.587, 0.299)):
    self.img = cv2.imread('testdata/1b.png')
    self.img = self.img/np.float(255)
    # self.img = cv2.resize(self.img,(int(self.img.shape[0]/2),int(self.img.shape[1]/2)),interpolation=cv2.INTER_CUBIC)
    self.l = self.img.shape[0]
    self.w = self.img.shape[1]
    self.sigma_s = sigma_s
    self.sigma_r = sigma_r
    self.weight = weight#[0.7, 0.2, 0.1]
    self.gray = np.array(self.weight[0] * self.img[:,:,0] + self.weight[1] * self.img[:,:,1] + self.weight[2] * self.img[:,:,2])
    self.BGRborded = None #cv2.copyMakeBorder(self.img, r, r, r, r, cv2.BORDER_REFLECT_101)
    self.gborded = None #cv2.copyMakeBorder(self.gray, r, r, r, r, cv2.BORDER_REFLECT_101)
    self.BGRdst = np.copy(self.img) * 0
    self.BGRjbfdst = np.copy(self.img) * 0
    self.gdst = np.copy(self.gray) * 0
    self.f = None
    self.r = 0
    pass
  
  def grayconv(self, i, j):
    r = self.r
    h_r = np.zeros((2*r + 1, 2*r + 1))
    tmp = np.zeros((2*r + 1, 2*r + 1))
    tmp += self.gborded[i-r:i+r+1, j-r:j+r+1] - self.gborded[i, j]
    
    h_r = np.exp(-(tmp**2) / (2 * self.sigma_r**2))

    local_f = self.f[:,:,0] * h_r
    
    local_f = local_f / np.sum(local_f)
    
    self.gdst[i - r, j - r] = np.sum(self.gborded[i-r:i+r+1, j-r:j+r+1] * local_f)

  def BGRconv(self, i, j):
    r = self.r
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    BGRtmp = np.zeros((2*r + 1, 2*r + 1, 3))
    
    BGRtmp += self.BGRborded[i-r:i+r+1, j-r:j+r+1] - self.BGRborded[i, j]
    h_r = np.exp((-1 * np.sum(BGRtmp**2, axis=2)) / (2 * (self.sigma_r**2)))
    
    local_f = np.multiply(self.f, h_r[:,:, np.newaxis]) 

    local_f = local_f / np.sum(local_f, axis=(0,1))
    
    self.BGRdst[i - r, j - r] = np.sum(self.BGRborded[i-r:i+r+1, j-r:j+r+1] * local_f, axis=(0,1))
  
  def BGRjointconv(self, i, j):
    r = self.r
    h_r = np.zeros((2*r + 1, 2*r + 1, 3))
    BGRtmp = np.zeros((2*r + 1, 2*r + 1, 3))
    
    
    BGRtmp += self.BGRborded[i-r:i+r+1, j-r:j+r+1] - self.gborded[i, j]
    # BGRtmp[:, :, 1] += self.BGRborded[i-r:i+r+1, j-r:j+r+1, 1] - self.gborded[i, j]
    # BGRtmp[:, :, 2] += self.BGRborded[i-r:i+r+1, j-r:j+r+1, 2] - self.gborded[i, j]
    # BGRtmp[:,:,2] += self.gborded[i-r:i+r+1, j-r:j+r+1] - self.BGRborded[i, j, 2]
    h_r += np.exp(-(BGRtmp**2 / (2 * (self.sigma_r**2)))) #* self.f
    # h_r[:,:,1] += np.exp(-(BGRtmp[:, :, 1]**2 / (2 * (self.sigma_r**2)))) * self.f[:,:,0]
    # h_r[:,:,2] += np.exp(-(BGRtmp[:, :, 2]**2 / (2 * (self.sigma_r**2)))) * self.f[:,:,0]
    
    # print(h_r[:,:,0])
    local_f = np.multiply(self.f, h_r) 
    # local_f = self.f.copy()
    # local_f = local_f / np.sum(local_f, axis=(0,1))
    # print(local_f)
    # exit()
    # print(local_f.shape)
    # print(np.sum(self.BGRborded[i-r:i+r+1, j-r:j+r+1] * local_f, axis=(0,1)).shape)
    # h_r = h_r / np.sum(h_r, axis=(0, 1))
    local_f = local_f / np.sum(local_f, axis=(0, 1))

    # print(h_r[:,:,2])
    
    re = np.sum(self.BGRborded[i-r:i+r+1, j-r:j+r+1] * local_f, axis=(0, 1))

    self.BGRjbfdst[i - r, j - r] += re
    # print(self.BGRjbfdst[i - r, j - r])
  
  def BGRbilateralfilter(self):
    begin = now()
   
    r = self.r = (self.sigma_s * 3)
    self.BGRborded = cv2.copyMakeBorder(self.img, r, r, r, r, cv2.BORDER_REFLECT_101)
    
    kernel_size = self.kernel_size = 2 * r + 1
    self.f = np.zeros((kernel_size, kernel_size, 3))
    for i in range(-r, r + 1):
      for j in range(-r, r + 1):
        self.f[i + r, j + r] = np.exp(-(i**2 + j**2) / (2 * self.sigma_s**2))
    for i in range(r, r + self.l):
      for j in range(r, r + self.w):
        self.BGRconv(i, j)
    print('use', now() - begin, 'sec')
    return self.BGRdst
  
  def BGRjointbilateralfilter(self):
    begin = now()

    # sig_s = 3
    self.r = r = (self.sigma_s * 3)
    self.BGRborded = cv2.copyMakeBorder(self.img, r, r, r, r, cv2.BORDER_REFLECT_101)
    self.gborded = cv2.copyMakeBorder(self.gray, r, r, r, r, cv2.BORDER_REFLECT_101)

    self.kernel_size = kernel_size = 2 * r + 1
    self.f = np.zeros((kernel_size, kernel_size, 3))
    for i in range(-r, r + 1):
      for j in range(-r, r + 1):
        self.f[i + r, j + r] += [np.exp(-(i**2 + j**2) / (2 * self.sigma_s**2)), np.exp(-(i**2 + j**2) / (2 * self.sigma_s**2)), np.exp(-(i**2 + j**2) / (2 * self.sigma_s**2))]
    print(self.f.shape)
    for i in range(r, r + self.l):
      for j in range(r, r + self.w):
        self.BGRjointconv(i, j)
    print('use', now() - begin, 'sec')
    return self.BGRjbfdst

  def graybilateralfilter(self):
    begin = now()

    # sig_s = 3
    r = self.r = (self.sigma_s * 3)
    self.gborded = cv2.copyMakeBorder(self.gray, r, r, r, r, cv2.BORDER_REFLECT_101)
    
    kernel_size = self.kernel_size = 2 * r + 1
    self.f = np.zeros((kernel_size, kernel_size, 3))
    for i in range(-r, r + 1):
      for j in range(-r, r + 1):
        self.f[i + r, j + r] = np.exp(-(i**2 + j**2) / (2 * self.sigma_s**2))
    for i in range(r, r + self.l):
      for j in range(r, r + self.w):
        self.grayconv(i, j)
    print('use', now() - begin, 'sec')
  
  def BGRshow(self, img_p):
    if img_p == 'src':
      cv2.imshow('BGR' + img_p, self.img)
    elif img_p == 'dst':
      cv2.imshow('BGR' + img_p, self.BGRdst)
    # print(self.BGRdst)
  
  def BGRjbfshow(self, img_p):
    if img_p == 'src':
      cv2.imshow('BGRjbf' + img_p, self.img)
    elif img_p == 'dst':
      cv2.imshow('BGRjbf' + img_p, self.BGRjbfdst)
    # print(self.BGRdst)

  def gshow(self, img_p):
    if img_p == 'src':
      cv2.imshow('gray' + img_p, self.gray)
    elif img_p == 'dst':
      cv2.imshow('gray' + img_p, self.gdst)
  

if __name__ == '__main__':
  # img = bilateral_filter()
  # img.gshow('src')
  # bf = img.BGRbilateralfilter()
  # img.BGRshow('dst')
  # jbf = img.BGRjointbilateralfilter()
  # img.BGRjbfshow('dst')
  # cv2.imshow('d', bf - jbf)
  weights = {}
  # for i in range(1,11):
  #   print(i)
  count = 1
  for i in range(0,11):
    for j in range(0,11):
      for k in range(0,11):
        if i + j + k == 10:
          try:
            tmp = weights[(i/10, j/10 ,k/10)]
          except:
            weights[(i/10, j/10 ,k/10)] = count
            count += 1

  vote = [0] * 66

  begin = now()
  sigmas = []
  for sigma_s in [1, 2, 3]:
    for sigma_r in [0.05, 0.1, 0.2]:
      sigmas.append([sigma_s, sigma_r])
      continue
      for weight in weights.keys():
        print(weight)
        img = bilateral_filter(sigma_s, sigma_r)
        # img.gshow('src')
        bf = img.BGRbilateralfilter()
        # img.BGRshow('dst')
        jbf = img.BGRjointbilateralfilter()
        img.BGRjbfshow('dst')
        cv2.imshow('d', bf - jbf)
        vote[weights[weight]-1] += np.sum(np.abs(bf - jbf)) / (bf.shape[0] * bf.shape[1])
        print(np.sum(np.abs(bf - jbf)) / (bf.shape[0] * bf.shape[1]))
        print(vote)
        pass
  print(sigmas)

  def sigpr(sigma_s, sigma_r, bf):
    vo = [0] * 66
    for weight in weights.keys():
      # if len(ths) == 6:
      #   while len(ths) != 0:
      #     ths.pop().join()
      print(weight)
      img = bilateral_filter(sigma_s, sigma_r, weight)
      # img.gshow('src')
      # bf = img.BGRbilateralfilter()
      # img.BGRshow('dst')
      jbf = img.BGRjointbilateralfilter()
      g = img.gray
      # img.BGRjbfshow('dst')
      # cv2.imshow('d', bf - jbf)
      cv2.imwrite('./jbf/' + str(weight) + '-' + str(sigma_s) + ',' + str(sigma_r) + '.png', (jbf)*255 - 1)
      cv2.imwrite('./d/' + str(weight) + '-' + str(sigma_s) + ',' + str(sigma_r) + '.png', (bf - jbf)*255 - 1)
      cv2.imwrite('./g/' + str(weight) + '.png', g*255 - 1)
      vo[weights[weight]-1] += np.sum(np.abs(bf - jbf)) / (bf.shape[0] * bf.shape[1])
      print(np.sum(np.abs(bf - jbf)) / (bf.shape[0] * bf.shape[1]))
      print(vo)
    np.save(str(sigma_s)+','+str(sigma_r)+'.npy', vo)

  img = bilateral_filter(sigma_s, sigma_r)
  bf = img.BGRbilateralfilter()
  prs = []
  for ss in sigmas:
    # sigpr(ss[0], ss[1], bf)
    pr = Process(target=sigpr, args=(ss[0], ss[1], bf, ))
    pr.start()
    prs.append(pr)
    # break

  for p in  prs:
    p.join()

  neighbor = {}
  for weight in weights:
    neighbor[weight] = []
    for tmp in [np.array(weight)*10 + np.array((1, 0, -1)), 
                np.array(weight)*10 + np.array((-1, 0, 1)), 
                np.array(weight)*10 + np.array((1, -1, 0)), 
                np.array(weight)*10 + np.array((-1, 1, 0)), 
                np.array(weight)*10 + np.array((0, 1, -1)), 
                np.array(weight)*10 + np.array((0, -1, 1))]:
      if tmp[0] >= 0 and tmp[1] >= 0 and tmp[2] >= 0:
        neighbor[weight].append(tuple(tmp/10))
  

  print(vote)
  print('use', (now() - begin)/3600, 'hours')
  res = []
  for ss in sigmas:
    res.append(np.load(str(ss[0])+','+str(ss[1])+'.npy'))

  re = np.array(res)
  re = np.reshape(re, (9, 66))

  vote = [0] * 66
  for contest in re:
    leaderboard = []
    for weight in weights:
      # print(weight)
      neibs = neighbor[weight]
      # print(neibs)
      mincount = 0
      for neib in neibs:
        if contest[weights[weight] - 1] <= contest[weights[neib] - 1]:
          mincount += 1
          
      if mincount == len(neibs):
        leaderboard.append([contest[weights[weight] - 1], weights[weight] - 1])
        # vote[weights[weight] - 1] += 1
  
    top = np.argsort(np.array(leaderboard)[:, 0], axis=0)
    # print(top)
    for rank in top:
      vote[leaderboard[rank][1]] += 6 - rank
      # print(3 - rank)

  print(vote)
  print(np.argsort(vote))
  print(np.argsort(vote)[65], np.argsort(vote)[64], np.argsort(vote)[63])
  print(list(weights.items())[np.argsort(vote)[65]][0], list(weights.items())[np.argsort(vote)[64]][0], list(weights.items())[np.argsort(vote)[63]][0])
  
  exit()


  re = np.sum(re, axis=0)
  print(re)
  exit()
  re = [0.6384649064654565, 0.6133621462055469, 0.6066006362093878, 0.6153353962783971, 0.6362802582897247, 
        0.6676098484966828, 0.7070112275043754, 0.750898893098228, 0.7984366212128828, 0.8490220543505694, 
        0.9036927355361263, 0.6032386546994084, 0.5683844044273545, 0.5527001558460362, 0.5545591716525394, 
        0.5703811117283054, 0.5989392458496148, 0.6391278199564489, 0.6880702719713552, 0.7435727939029279, 
        0.807022325501487, 0.5735828491280682, 0.5293886596614193, 0.50435787914058, 0.4988857463891088, 
        0.5105580530746842, 0.5385100264533873, 0.583228986921582, 0.6414656218652192, 0.7110505472314426, 
        0.549857063329691, 0.49726534427275454, 0.464228326712274, 0.4527930499347915, 0.46345859438669096, 
        0.4970884653770057, 0.5509798951928983, 0.6218744634348314, 0.533001467792114, 0.47503937850357086, 
        0.4367339631283272, 0.4246539920806742, 0.4414536704809961, 0.48434651401151874, 0.5479258382669634, 
        0.5251164973199861, 0.46590019418188333, 0.4288806254030107, 0.42336991132715546, 0.44866139830583607, 
        0.4990850232489915, 0.5289707928513122, 0.47519490447681517, 0.4468253923577614, 0.45064587282527063, 
        0.48120693211048543, 0.5496596762724546, 0.5092671139208373, 0.49132729872151293, 0.49954840163825065, 
        0.5940089749632802, 0.5658742296883345, 0.5536662371037796, 0.6571428844677907, 0.6357864667821954, 
        0.7296096595097712]
  re2 = [0.71283387, 0.69222905, 0.6885125 , 0.69959056, 0.72309709, 0.75714534,
         0.79893665, 0.84487072, 0.89384731, 0.94424156, 0.99691524, 0.68512255,
         0.65624662, 0.64459702, 0.64944241, 0.66825895, 0.6995897 , 0.74178431,
         0.79123212, 0.84518786, 0.90526866, 0.66330406, 0.62700131, 0.60749506,
         0.60573904, 0.62052785, 0.65018857, 0.69368559, 0.74836904, 0.8130824 ,
         0.64711749, 0.60466672, 0.57877381, 0.57134246, 0.5831879 , 0.61330448,
         0.66162313, 0.72630711, 0.63696291, 0.59064085, 0.5602167 , 0.54973104,
         0.56165502, 0.59748794, 0.65374363, 0.63389307, 0.58587323, 0.55476523,
         0.54640651, 0.56443557, 0.60621889, 0.63902662, 0.59383056, 0.56901608,
         0.56842239, 0.59053315, 0.65729032, 0.62258044, 0.60666663, 0.60999732,
         0.69633778, 0.67131715, 0.66002534, 0.75176164, 0.73204194, 0.81602595,]
  print(np.min(re))
  count = 1
  for i in re:
    if i == np.min(re):
      print(count)
    count += 1
  print(weights)
  # print(weights[(0.0, 0.0, 1)])
  
  # img.BGRjbfshow('src')
  # img.BGRshow('dst')
  # img.graybilateralfilter()
  # img.gshow('dst')
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
