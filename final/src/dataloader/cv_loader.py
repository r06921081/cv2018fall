import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataloader import preprocess
import numpy as np
import re
import sys
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    img = Image.open(path).convert('RGB')
    
    img = img.resize( (1242, 376), Image.BILINEAR )
    #img.save(path+"resize_image.png")
    return img

def disparity_loader(path):
    pfm = readPFM(path)
    pfm = cv2.resize(pfm, (1242, 376))
    #cv2.imwrite(path+"new_disparity.png", pfm)
    #print(pfm.shape) # resize to (1242, 375)
    #exit()
    return pfm

def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
             
           left_img   = processed(left_img)
           right_img  = processed(right_img)
           #print("left_img: ", left_img.shape)
           #print("right_img: ", right_img.shape)
           #print("dataL: ", dataL.shape)
           #exit()
           
           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           #dataL = dataL.crop((w-1232, h-368, w, h))
           dataL = dataL[h-368:h, w-1232:w]
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)

           processed = preprocess.get_transform(augment=False) 
           #processed = get_transform(augment=False) 
            
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
