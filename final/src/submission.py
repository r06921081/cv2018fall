from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
from PIL import Image
import cv2
import sys
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--left_datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--right_datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--output', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
elif args.KITTI == 'cv':
   from dataloader import CV_submission_loader as DA 
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.left_datapath, args.right_datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[
        2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)

def main():
   processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):

       imgL_o = skimage.io.imread(test_left_img[inx]).astype('float32')
       print(imgL_o.shape)
       
       
       ###
       #imgL_o = np.stack([imgL_o, imgL_o, imgL_o], 2)
       ###
       imgR_o = skimage.io.imread(test_right_img[inx]).astype('float32')
       ###
       #imgR_o = np.stack([imgR_o, imgR_o, imgR_o], 2)
       ###
       
       
       raw_h, raw_w = imgL_o.shape[0], imgL_o.shape[1]
       
       imgL_o = cv2.resize(imgL_o, (1242, 376))
       imgR_o = cv2.resize(imgR_o, (1242, 376))
       #imgL_o = imgL_o.resize( (1242, 376), Image.BILINEAR )
       #imgR_o = imgR_o.resize( (1242, 376), Image.BILINEAR )
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       # pad to (384, 1248)
       top_pad = 384-imgL.shape[2]
       left_pad = 1248-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = 384-imgL_o.shape[0]
       left_pad  = 1248-imgL_o.shape[1]
       img = pred_disp[top_pad:,:-left_pad]
       
       #writePFM(str(inx)+"result_syn.pfm", cv2.resize(img, (384,512)).astype(np.float32))
       writePFM(args.output, cv2.resize(img, (raw_w,raw_h)).astype(np.float32))
       #skimage.io.imsave(test_left_img[inx].split('/')[-1]+"sub.png",(img).astype('uint16'))

if __name__ == '__main__':
   main()






