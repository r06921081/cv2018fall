import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  #left_fold  = 'image_2/'
  #right_fold = 'image_3/'
  #disp_L = 'disp_occ_0/'
  #disp_R = 'disp_occ_1/'

  left_fold  = './'
  right_fold = './'
  disp_L = './'
  disp_R = './'

  #image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  left_image = ["TL0.png","TL1.png","TL2.png", "TL3.png", "TL4.png", "TL5.png","TL6.png", "TL7.png", "TL8.png","TL9.png"]
  right_image = ["TR0.png","TR1.png","TR2.png", "TR3.png", "TR4.png", "TR5.png","TR6.png", "TR7.png", "TR8.png","TR9.png"]
  disp_image = ["TLD0.pfm","TLD1.pfm","TLD2.pfm","TLD3.pfm","TLD4.pfm","TLD5.pfm","TLD6.pfm","TLD7.pfm","TLD8.pfm","TLD9.pfm"]
  #train = image[:160]
  #val   = image[160:]

  left_train  = [filepath+left_fold+img for img in left_image]
  right_train = [filepath+right_fold+img for img in right_image]
  disp_train_L = [filepath+disp_L+img for img in disp_image]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = left_train
  right_val = right_train
  disp_val_L = disp_train_L
  #disp_val_R = [filepath+disp_R+img for img in val]

  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
