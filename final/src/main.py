import numpy as np
import argparse
import cv2
import time
from util import writePFM, readPFM
import subprocess, sys
from funtions import hist, fillpixel

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')

def equal(L, R):
    L = L.astype(np.float)
    R = R.astype(np.float)
    if len(L.shape) == 3:
        ch = 3
    else:
        ch = 1
    L = np.reshape(L, L.shape[:2] + (ch,))
    R = np.reshape(R, R.shape[:2] + (ch,))
    for i in range(ch):
        R[:,:,i] = R[:,:,i] - (np.mean(R[:,:,i]) - np.mean(L[:,:,i]))
        R[:,:,i] = np.clip(R[:,:,i], 0, 255)
        veo = np.var(L[:,:,i])/np.var(R[:,:,i])
        R[:,:,i] = R[:,:,i]*(veo)**0.5
        
    return L.astype(np.uint8), R.astype(np.uint8)

def transform(image, window_size=3):
    
    window_size += 2
    half_window_size = window_size // 2

    image = cv2.copyMakeBorder(image, top=half_window_size, left=half_window_size, right=half_window_size, bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)
    rows, cols = image.shape
    census = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    census1 = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    census2 = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    
    center_pixels = image[half_window_size:rows - half_window_size, half_window_size:cols - half_window_size]
    offsets1 = [(0,0),(0,2),(0,4),(2,0),(2,4),(4,0),(4,2),(4,4)]
    offsets2 = [(2,4),(4,0),(4,2),(4,4)]
        
    for (row, col) in offsets1:
        census1 = (census1 << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
        
    return (census1)

def getlrdisp(left_img, right_img):
    left_img = hist(left_img)
    right_img = hist(right_img)
    left_img, right_img = equal(left_img, right_img)
    
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    left_img = transform(left_img)
    right_img = transform(right_img)
    
    image_size = left_img.shape
    width = left_img.shape[1]

    MAX_FEATURES = 2000
    GOOD_MATCH_PERCENT = 0.05

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(left_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(right_img, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    
    matches = matches[:numGoodMatches]
    imMatches = cv2.drawMatches(left_img, keypoints1, right_img, keypoints2, matches, None)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    differ = []
    for i, match in enumerate(matches):
        
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        x1 = int(keypoints1[match.queryIdx].pt[0])
        x2 = int(keypoints2[match.trainIdx].pt[0])
        # print("i: ", str(index), " ", int(keypoints1[match.queryIdx].pt[0]), " ", int(keypoints2[match.trainIdx].pt[0]),str(x2-x1))
        differ.append(x2-x1)
    differ.append(0)
    
    x_range = range(len(differ)) 
    
    cu = {}
    for i in differ:
        try:
            cu[i] += 1
        except:
            cu[i] = 1
    differ = list(set(differ))
    differ.sort()
    number = []
    for i in differ:
        number.append(cu[i])
    
    idx = differ.index(0)
    print("i: ", idx, " ", number[idx])
    scale_factor = number[idx]
    in_differ = np.array(differ[::-1])*-1
    in_number = np.array(number[::-1])*-1
    in_differ = in_differ.tolist()
    in_number = in_number.tolist()

    if len(in_number[:in_differ.index(0)]) != 0:
        maxd = in_differ[np.argmin(in_number[:in_differ.index(0)])]
    else:
        maxd = 0
    maxd = -maxd

    if len(number[:differ.index(0)]) != 0:
        mind = differ[np.argmax(number[:differ.index(0)])]
    else:
        mind = 0
    """
    if scale_factor <= 10:
        scale_factor = 3
    elif scale_factor > 10 and scale_factor <15:
        scale_factor = 5
    elif scale_factor >= 15 and scale_factor <= 20:
        scale_factor = 6
    elif scale_factor >20 and scale_factor<=30:
        scale_factor = 8
    elif scale_factor > 30 and scale_factor <= 60:
        scale_factor = 10
    elif scale_factor > 60:
        scale_factor = 11
    """
    return scale_factor 


if_syn = True 
args = parser.parse_args()
# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    disp = np.zeros((h, w), dtype=np.int32)
    
    # TODO: Some magic
    if if_syn:
        print('Synthesis')
        
        cmd = 'python submission.py --maxdisp 192 --model stackhourglass --KITTI cv --left_datapath '+ args.input_left +' --right_datapath '+args.input_right+' --output '+ "ori_"+args.output +' --loadmodel synthesis_tune_best_for_end.tar'
        print(cmd)
        retcode = subprocess.call(cmd, shell=True)
        print(retcode)
               
        cmd = 'python submission.py --maxdisp 192 --model stackhourglass --KITTI cv --left_datapath '+ args.input_left +' --right_datapath '+args.input_right+' --output '+ "resolu_"+args.output +' --loadmodel aug_resolutionfinetune_146.tar'
        print(cmd)
        retcode = subprocess.call(cmd, shell=True)
        print(retcode)
        
        cmd = 'python submission.py --maxdisp 192 --model stackhourglass --KITTI cv --left_datapath '+ args.input_left +' --right_datapath '+args.input_right+' --output '+ "color_"+args.output +' --loadmodel aug_colorfinetune_169.tar'
        print(cmd)
        retcode = subprocess.call(cmd, shell=True)
        print(retcode)

        p1 = readPFM("ori_"+args.output)
        p2 = readPFM("resolu_"+args.output)
        p3 = readPFM("color_"+args.output)
        final_pfm = (p1+p2+p3)/3
        writePFM(args.output, final_pfm)

    else:
        print('Real')
        scale_factor = getlrdisp(Il, Ir)
        print(scale_factor)
        #if scale_factor in [14,56,75]:
        #    cmd = 'python cencus_quick.py '+ args.input_left +' '+args.input_right+' '+ args.output + ' ' + 
        cmd = 'python cencus_transform.py '+ args.input_left +' '+args.input_right+' '+ args.output
        print(cmd)
        retcode = subprocess.call(cmd, shell=True)
        print(retcode)
        
        
        
    
    return disp


def main():
    args = parser.parse_args()
    global if_syn
    
    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    print(img_left.shape)
    if_syn = False
    for h in range(img_left.shape[0]):
        for w in range(img_left.shape[1]):
            if img_left[h,w,0] != img_left[h,w,1] != img_left[h,w,2]:
                if img_right[h,w,0] != img_right[h,w,1] != img_right[h,w,2]:
                    if_syn = True
                    break

    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    #if if_syn is False:
    #writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
