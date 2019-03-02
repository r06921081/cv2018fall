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

def dataloader(left_filepath, right_filepath):

  #left_fold  = './'
  #right_fold = './'
  #left_image = ["TL0.png","TL1.png","TL2.png", "TL3.png", "TL4.png", "TL5.png","TL6.png", "TL7.png", "TL8.png","TL9.png"]
  #right_image = ["TR0.png","TR1.png","TR2.png", "TR3.png", "TR4.png", "TR5.png","TR6.png", "TR7.png", "TR8.png","TR9.png"]


  #image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]


  #left_test  = [filepath+left_fold+img for img in left_image]
  #right_test = [filepath+right_fold+img for img in right_image]
  left_test  = [left_filepath] 
  right_test = [right_filepath]
  return left_test, right_test
