import numpy as np 
import cv2
from time import time as now
import math
from funtions import hist, fillpixel, fillpixel_I
from wmf import guidedfilter_color_precompute, guided_filter, weighted_median_filter
import logging

def show(imgs, st='show'):
    for i, im in enumerate(imgs):
        cv2.imshow(st+str(i), im.astype(np.uint8))
    cv2.waitKey(0)
    exit()
    
def getlrdisp(left_img, right_img):
    left_img = hist(left_img)
    right_img = hist(right_img)

    compress = 150
    left_img = compress//2 + (left_img/255)*(255-compress)
    right_img = compress//2 + (right_img/255)*(255-compress)
    left_img = cv2.resize(left_img,(0,0) ,fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right_img,(0,0) ,fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    left_img, right_img = equal(left_img, right_img)
    
    
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
    print(numGoodMatches)
    
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
    # import matplotlib.pyplot as plt
    # plt.figure(0)
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
    

    # plt.bar(differ, number)
    # print(differ, number)
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
    # if np.abs(maxd) > np.abs(mind)*2 or np.abs(mind) > np.abs(maxd)*2
    print(maxd, mind)

    return maxd, mind

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
        R[:,:,i] = np.clip(R[:,:,i],0,255)
        
    
    return L.astype(np.uint8), R.astype(np.uint8)


import numpy as np
import cv2

def norm(image):
    return cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

def transform(image, window_size=3):
    """
    Take a gray scale image and for each pixel around the center of the window generate a bit value of length
    window_size * 2 - 1. window_size of 3 produces bit length of 8, and 5 produces 24.

    The image gets border of zero padded pixels half the window size.

    Bits are set to one if pixel under consideration is greater than the center, otherwise zero.

    :param image: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
    :param window_size: int odd-valued
    :return: numpy.ndarray(shape=(MxN), , dtype=numpy.uint8)
    >>> image = np.array([ [50, 70, 80], [90, 100, 110], [60, 120, 150] ])
    >>> np.binary_repr(transform(image)[0, 0])
    '1011'
    >>> image = np.array([ [60, 75, 85], [115, 110, 105], [70, 130, 170] ])
    >>> np.binary_repr(transform(image)[0, 0])
    '10011'
    """
    

    window_size += 2
    half_window_size = window_size // 2

    image = cv2.copyMakeBorder(image, top=half_window_size, left=half_window_size, right=half_window_size, bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)
    rows, cols = image.shape
    census = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    census1 = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    census2 = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
    print(census.shape)
    
    center_pixels = image[half_window_size:rows - half_window_size, half_window_size:cols - half_window_size]

    # offsets = [(row, col) for row in range(half_window_size) for col in range(half_window_size) if not row == half_window_size + 1 == col]
    # print(offsets)
    # for (row, col) in offsets1:
    #     census = (census << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
    #     print(row,row + rows - half_window_size * 2)
    offsets1 = [(0,0),(0,2),(0,4),(2,0),(2,4),(4,0),(4,2),(4,4)]
    offsets2 = [(2,4),(4,0),(4,2),(4,4)]
        
    for (row, col) in offsets1:
        census1 = (census1 << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
        # print(census1[0])
        # exit()
    return (census1)
    
    print(np.max(census1))
    print(census1)
    exit()()
    for (row, col) in offsets2:
        census2 = (census2 << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
        # print(row,row + rows - half_window_size * 2)
    # exit()
    # print(np.max(census2))
def column_cost(left_col, right_col):
    """
    Column-wise Hamming edit distance
    Also see https://www.youtube.com/watch?v=kxsvG4sSuvA&feature=youtu.be&t=1032
    :param left: numpy.ndarray(shape(Mx1), dtype=numpy.uint)
    :param right: numpy.ndarray(shape(Mx1), dtype=numpy.uint)
    :return: numpy.ndarray(shape(Mx1), dtype=numpy.uint)
    >>> image = np.array([ [50, 70, 80], [90, 100, 110], [60, 120, 150] ])
    >>> left = transform(image)
    >>> image = np.array([ [60, 75, 85], [115, 110, 105], [70, 130, 170] ])
    >>> right = transform(image)
    >>> column_cost(left, right)[0, 0]
    2
    """
    return np.sum(np.unpackbits(np.bitwise_xor(left_col, right_col), axis=1), axis=1).reshape(left_col.shape[0], left_col.shape[1])

def cost(left, right, window_size=3, disparity=0):
    """
    Compute cost difference between left and right grayscale images. Disparity value can be used to assist with evaluating stereo
    correspondence.
    :param left: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
    :param right: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
    :param window_size: int odd-valued
    :param disparity: int
    :return:
    """
    ct_left = norm(transform(left, window_size=window_size))
    ct_right = norm(transform(right, window_size=window_size))
    rows, cols = ct_left.shape
    C = np.full(shape=(rows, cols), fill_value=0)
    for col in range(disparity, cols):
        C[:, col] = column_cost(
            ct_left[:, col:col + 1],
            ct_right[:, col - disparity:col - disparity + 1]
        ).reshape(ct_left.shape[0])
    return C

def hamming(l, r):
    return np.sum(np.unpackbits(np.bitwise_xor(l, r), axis=2), axis=2)


def select_wta_turn(cost_aggre, bias):
    
    d, h, w = cost_aggre.shape
    wta = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            index = np.argmin(cost_aggre[:,i,j])
            #index = index +1
            if index % 2 == 0:
                index = index/2
            else:
                index = -1*((index+1)/2)
            wta[i,j] = index
    min_ = np.min(wta)
    return wta, bias

def select_wta_turn2(cost_aggre, wta, wta_f):
    
    d, h, w = cost_aggre.shape
    pos = wta.shape[0]
    neg = wta_f.shape[0]
    outwta = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            index = np.argmin(cost_aggre[:,i,j])
            #index = index +1
            if index % 2 == 0:
                index = index/2
            else:
                index = -1*((index+1)/2)
            outwta[i,j] = index
    min_ = np.min(wta)
    wta = np.argmax(-wta, axis=0)
    wta = wta + neg
    print(np.min(wta))
    wta_f = -np.argmin(wta_f, axis=0)
    wta_f = wta_f + neg
    print(np.min(wta_f))
    outwta = outwta + neg
    print(np.min(outwta))
    mask0 = wta.copy()
    mask1 = wta_f.copy()
    mask0[(outwta).astype(np.uint8)==wta.astype(np.uint8)] = 255
    mask1[(outwta).astype(np.uint8)==wta_f.astype(np.uint8)] = 255
    _, th0 = cv2.threshold(mask0.astype(np.uint8), 254, 255, cv2.THRESH_BINARY)
    
    _, th1 = cv2.threshold(mask1.astype(np.uint8), 254, 255, cv2.THRESH_BINARY)
    AND0 = np.bitwise_and((outwta).astype(np.uint8), th0)
    AND1 = np.bitwise_and((outwta).astype(np.uint8), th1)
    AND3 = np.bitwise_and(AND0, AND1)
    # AND = np.bitwise_and(th0, th1)
    # OR = np.bitwise_or(th0, th1)
    # XOR = np.bitwise_xor(th0, th1)
    # show([outwta, AND3, wta, wta_f, mask0, mask1])

    return th0, th1

def getCosts(yhalf_windos, xhalf_windos, ct_left, ct_right, conlist, max_disp):
    yhalf_windos = ysize//2
    xhalf_windos = xsize//2
    cont_l = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)), dtype=np.uint8)
    for i, (hbias, wbias) in enumerate(conlist):
        cont_l[hbias:h+hbias,wbias:w+wbias,i] = ct_left

    cont_l = cont_l[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]
    wta = np.ones((1, h, w), dtype=np.uint8)*255
    # if not posSkip:
    for i in range(max_disp):
        tmp1 = ct_right[:,:-jump]
        ct_right[:,jump:] = tmp1
        ct_right[:,i*jump:(i+1)*jump] = ct_right[:,:jump]

        cont_r = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)), dtype=np.uint8)
        for j, (hbias, wbias) in enumerate(conlist):
            cont_r[hbias:h+hbias,wbias:w+wbias,j] = ct_right
        cont_r = cont_r[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]
        
        diff = hamming(cont_l, cont_r)

        wta = np.concatenate((wta, np.expand_dims(diff/np.mean(diff), axis=0)), axis=0)
        

    # print(wta, wta.shape)
    wta = wta[1:,:,:] 
    return  wta

def getMix(yhalf_windos, xhalf_windos, ct_left, ct_right, conlist, max_disp, fl, fr):
    yhalf_windos = ysize//2
    xhalf_windos = xsize//2
    cont_l = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)), dtype=np.uint8)
    for i, (hbias, wbias) in enumerate(conlist):
        cont_l[hbias:h+hbias,wbias:w+wbias,i] = ct_left

    cont_l = cont_l[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]

    cont_fl = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)))
    for i, (hbias, wbias) in enumerate(conlist):
        cont_fl[hbias:h+hbias,wbias:w+wbias,i] = fl

    cont_fl = cont_fl[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]
    wta = np.ones((1, h, w))*255
    # if not posSkip:
    for i in range(max_disp):
        tmp1 = ct_right[:,:-jump]
        ct_right[:,jump:] = tmp1
        ct_right[:,i*jump:(i+1)*jump] = ct_right[:,:jump]

        cont_r = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)), dtype=np.uint8)
        for j, (hbias, wbias) in enumerate(conlist):
            cont_r[hbias:h+hbias,wbias:w+wbias,j] = ct_right
        cont_r = cont_r[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]
        
        diff1 = hamming(cont_l, cont_r).astype(np.float)

        tmp2 = fr[:,:-jump]
        fr[:,jump:] = tmp2
        fr[:,i*jump:(i+1)*jump] = fr[:,:jump]

        cont_fr = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)))
        for j, (hbias, wbias) in enumerate(conlist):
            cont_fr[hbias:h+hbias,wbias:w+wbias,j] = fr
        cont_fr = cont_fr[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]

        diff2 = np.sum(np.abs(cont_l - cont_r),axis=2) 
        diff = diff1/np.mean(diff1) + diff2/np.mean(diff2)

        wta = np.concatenate((wta, np.expand_dims(diff, axis=0)), axis=0)
        

    # print(wta, wta.shape)
    wta = wta[1:,:,:] 
    return  wta

def getCostsBox(yhalf_windos, xhalf_windos, ct_left, ct_right, conlist, max_disp):
    yhalf_windos = ysize//2
    xhalf_windos = xsize//2
    cont_l = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)), dtype=np.uint8)
    for i, (hbias, wbias) in enumerate(conlist):
        cont_l[hbias:h+hbias,wbias:w+wbias,i] = ct_left

    cont_l = cont_l[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]
    wta = np.ones((1, h, w), dtype=np.float)*255
    # if not posSkip:
    for i in range(max_disp):
        tmp1 = ct_right[:,:-jump]
        ct_right[:,jump:] = tmp1
        ct_right[:,i*jump:(i+1)*jump] = ct_right[:,:jump]

        cont_r = np.zeros((h+2*yhalf_windos, w+2*xhalf_windos, len(conlist)), dtype=np.float)
        for j, (hbias, wbias) in enumerate(conlist):
            cont_r[hbias:h+hbias,wbias:w+wbias,j] = ct_right
        cont_r = cont_r[yhalf_windos:-yhalf_windos,xhalf_windos:-xhalf_windos,:]
        
        diff = np.sum(np.abs(cont_l - cont_r),axis=2)
        
        wta = np.concatenate((wta, np.expand_dims(diff/np.mean(diff), axis=0)), axis=0)
        

    # print(wta, wta.shape)
    wta = wta[1:,:,:] 
    return  wta

import sys

if __name__ == "__main__":
    begin = now()
    """
    ri = 3
    epsi = 0.00001
    imgL = cv2.imread('../CV_finalproject/data/Real/TL0.bmp')
    wta = cv2.imread('beforewmf_cen0.png',)
    wta = weighted_median_filter(wta.astype(np.uint8), imgL, [i for i in range(int(np.max(wta))+1)], ri, epsi)
    print(begin - now())
    show([wta])
    """
    # load as grayscale
    # imgL = cv2.imread('../../hw4/testdata/cones/im2.png',0)
    # imgR = cv2.imread('../../hw4/testdata/cones/im6.png',0)
    Lpath = sys.argv[1]
    Rpath = sys.argv[2]
    Opath = sys.argv[3]
    imgL = cv2.imread(Lpath)
    imgR = cv2.imread(Rpath)
    
    imgL_o = imgL.copy()
    imgR_o = imgR.copy()
    
    fh, fw, c = imgL.shape
    
    # imgL = cv2.GaussianBlur(imgL,(3,3),0)
    # imgR = cv2.GaussianBlur(imgR,(5,7),0)
    imgL, imgR = equal(imgL, imgR)
    imgL = hist(imgL)
    imgR = hist(imgR)

    # imgL = cv2.bilateralFilter(imgL, 9, 150, 80)
    # imgR = cv2.bilateralFilter(imgR, 9, 150, 80)
    neg, pos = getlrdisp(imgL, imgR)
    negSkip, posSkip = False, False
    if neg == 0:
        negSkip = True
    if pos == 0:
        posSkip = True
    # print(pos, neg)
    
    if pos != 0:
        pos_s = pos/abs(pos)
    else:
        pos_s = 1
    if neg != 0:
        neg_s = neg/abs(neg)
    else:
        neg_s = 1
    pos = abs(pos)
    neg = abs(neg)
    if abs(pos) > abs(neg) and neg != 0:
        neg = int((pos + neg)/2)
    elif abs(pos) < abs(neg) and pos != 0:
        pos = int((pos + neg)/2)
    # print(pos, neg)

    max_disp = pos + neg
    disp_todo = 64
    scale = math.ceil(np.clip(disp_todo/max_disp, 1, 255))

    pos = math.ceil(pos*scale*1.2)
    neg = math.ceil(neg*scale*1.2)
    spos = pos*pos_s
    sneg = neg*neg_s
    jump = 1#math.ceil(64/(max_disp)

    print('pos', 'neg', pos, neg)    
    print('spos', 'sneg', spos, sneg)    
    print('scale', scale)
    print('max_disp', max_disp)
    print('jump', jump)
    # imgL = cv2.imread('./PL.png',0)
    # imgR = cv2.imread('./PR.png',0)
    imgL = cv2.resize(imgL, (0,0),fx=scale, fy=1, interpolation=cv2.INTER_CUBIC)
    imgR = cv2.resize(imgR, (0,0),fx=scale, fy=1, interpolation=cv2.INTER_CUBIC)


    imgL_g = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_g = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # imgL_g = cv2.bilateralFilter(imgL_g, 51, 150, 80)
    # imgR_g = cv2.bilateralFilter(imgR_g, 51, 150, 80)
    # newnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnew
    fx_l = np.gradient(imgL_g/255.)[1]
    fx_r = np.gradient(imgR_g/255.)[1]
    fx_l = (fx_l + 0.5)*255
    fx_r = (fx_r + 0.5)*255
    fx_l_f = cv2.flip(fx_l, 1).copy()
    fx_r_f = cv2.flip(fx_r, 1).copy()
    # newnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnewnew
    compress = 150
    buffer = 1

    size = 5
    ysize = size
    xsize = math.ceil(size*(scale+1))
    print(ysize, xsize)
    # imgL_g = compress//2 + (imgL_g/255)*(255-compress)
    # imgL_g = compress//2 + (imgL_g/255)*(255-compress)

    
    
    # cv2.imshow('l',imgL.astype(np.uint8))
    # cv2.imshow('r',imgR.astype(np.uint8))
    # cv2.waitKey(0)
    # imgL = norm(transform(imgL))
    # imgR = norm(transform(imgR))
    h, w, c = imgL.shape
    # kernel = np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]])
    # imgL = cv2.filter2D(imgL,-1,kernel)
    # imgR = cv2.filter2D(imgR,-1,kernel)
    # =========================start===========================
    imgL_f = imgR_g[:,::-1].copy()
    imgR_f = imgL_g[:,::-1].copy()
    
    
    ct_left = transform(imgL_g)
    ct_right = transform(imgR_g)

    conlist = []
    for i in range(ysize):
        for j in range(xsize):
            if i%2==0 and j%2==0:
                conlist.append((i,j)) 
    print('filter point', len(conlist))
    yhalf_windos = ysize//2
    xhalf_windos = xsize//2
    
    if not posSkip:
        wta = getCosts(yhalf_windos, xhalf_windos, 
            ct_left.copy(), ct_right.copy(), conlist, pos)
        # # wtaBox = getCostsBox(yhalf_windos, xhalf_windos, 
        # #     fx_l.copy(), fx_r.copy(), conlist, pos)
        # # wta = wtaBox + wta
        # wta = getMix(yhalf_windos, xhalf_windos, 
        #     ct_left.copy(), ct_right.copy(), conlist, pos, fx_l.copy(), fx_r.copy())

    else:
        wta = np.ones((1, h, w))*255

    wta_tmp = np.zeros((wta.shape[0], fh, fw))
    for i in range(wta.shape[0]):
        for x in range(wta.shape[2]//scale):
            wta_tmp[i, :, x] = np.min(wta[i, :, x*scale:(x+1)*scale], axis=1)
    wta = wta_tmp
    hoo = np.argmin(wta,axis=0)
    
    # cv2.imwrite('cenl'+n+'.png', np.argmin(wta, axis=0).astype(np.uint8))
    # half
    ct_left_f = transform(imgR_f)
    ct_right_f = transform(imgL_f)

    if not negSkip:
        wta_f = getCosts(yhalf_windos, xhalf_windos, 
            ct_left_f.copy(), ct_right_f.copy(), conlist, neg)[:,:,::-1]
        # # wta_fBow = getCostsBox(yhalf_windos, xhalf_windos, 
        # #     fx_l_f.copy(), fx_r_f.copy(), conlist, neg)[:,:,::-1]
        # # wta_f = wta_fBow + wta_f
        # wta_f = getMix(yhalf_windos, xhalf_windos, 
        #     ct_left_f.copy(), ct_right_f.copy(), conlist, neg, fx_l_f.copy(), fx_r_f.copy())[:,:,::-1]
    else:
        wta_f = np.ones((1, h, w))*255
    wta_tmp = np.zeros((wta_f.shape[0], fh, fw))
    for i in range(wta_f.shape[0]):
        for x in range(wta_f.shape[2]//scale):
            wta_tmp[i, :, x] = np.min(wta_f[i, :, x*scale:(x+1)*scale], axis=1)
    wta_f = wta_tmp
    hoo_f = np.argmin(wta_f,axis=0)

    # cv2.imwrite('cenr'+n+'.png', np.argmin(wta_f, axis=0).astype(np.uint8))    
    print(wta.shape, wta_f.shape)

    total_d = []    

    total_d.append(wta_f[0])
    for i in range(1, max(wta.shape[0],wta_f.shape[0])):
        if i < wta_f.shape[0]:
            total_d.append(wta_f[i])
        else:
            total_d.append(np.ones(wta[0].shape)*255)
        if i < wta.shape[0]:
            total_d.append(wta[i])
        else:
            total_d.append(np.ones(wta[0].shape)*255)
    
    total_d = np.array(total_d)
    print(total_d.shape)
    # goodl, goodr = select_wta_turn2(total_d, wta, wta_f)
    # show([goodl, goodr])
    wta, l_bias = select_wta_turn(total_d, neg)
    print(np.max(wta), np.min(wta))
    wta = np.clip(wta, -sneg, -spos)
    # wta, l_bias = select_wta_turn2(total_d, wta, wta_f)

    #cv2.imwrite(Opath+'cen_.png', (wta).astype(np.uint8))
    
    # exit()
    # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    
    if not posSkip:
        r_wta = getCosts(yhalf_windos, xhalf_windos, 
            ct_right_f.copy(), ct_left_f.copy(), conlist, pos)
        # # r_wtaBox = getCostsBox(yhalf_windos, xhalf_windos, 
        # #     fx_r_f.copy(), fx_l_f.copy(), conlist, pos)
        # # r_wta = r_wtaBox + r_wta
        # r_wta = getMix(yhalf_windos, xhalf_windos, 
        #     ct_right_f.copy(), ct_left_f.copy(), conlist, pos, fx_r_f.copy(), fx_l_f.copy())
    else:
        r_wta = np.ones((1, h, w))*255

    wta_tmp = np.zeros((r_wta.shape[0], fh, fw))
    for i in range(r_wta.shape[0]):
        for x in range(r_wta.shape[2]//scale):
            wta_tmp[i, :, x] = np.min(r_wta[i, :, x*scale:(((x+1)*scale)//2)*2+1], axis=1)
    r_wta = wta_tmp
    r_hoo = np.argmin(r_wta[:,:,::-1],axis=0)
    # cv2.imwrite('cenl_f'+n+'.png', np.argmin(r_wta, axis=0).astype(np.uint8))    
    
    if not negSkip:
        r_wta_f = getCosts(yhalf_windos, xhalf_windos, 
            ct_right.copy(), ct_left.copy(), conlist, neg)[:,:,::-1]
        # r_wta_fBow = getCostsBox(yhalf_windos, xhalf_windos, 
        #     fx_l.copy(), fx_r.copy(), conlist, neg)[:,:,::-1]
        # r_wta_f = r_wta_fBow + r_wta_f
        # r_wta_f = getMix(yhalf_windos, xhalf_windos, 
        #     ct_right.copy(), ct_left.copy(), conlist, neg, fx_l.copy(), fx_r.copy())[:,:,::-1]
    else:
        r_wta_f = np.ones((1, h, w))*255

    wta_tmp = np.zeros((r_wta_f.shape[0], fh, fw))
    for i in range(r_wta_f.shape[0]):
        for x in range(r_wta_f.shape[2]//scale):
            wta_tmp[i, :, x] = np.min(r_wta_f[i, :, x*scale:(((x+1)*scale)//2)*2+1], axis=1)
    r_wta_f = wta_tmp
    r_hoo_f = np.argmin(r_wta_f[:,:,::-1],axis=0)
    # cv2.imwrite('cenr_f'+n+'.png', np.argmin(r_wta_f, axis=0).astype(np.uint8))    

    r_total_d = []    
    r_total_d.append(r_wta_f[0])
    for i in range(1,max(r_wta.shape[0],r_wta_f.shape[0])+1):
        
        if i < r_wta_f.shape[0]:
            r_total_d.append(r_wta_f[i])
        else:
            r_total_d.append(np.ones(r_wta[0].shape)*255)
        if i < r_wta.shape[0]:
            r_total_d.append(r_wta[i])
        else:
            r_total_d.append(np.ones(r_wta[0].shape)*255)
    r_total_d = np.array(r_total_d)[:,:,::-1]
    # print(r_total_d.shape)
    
    r_wta, r_bias = select_wta_turn(r_total_d, neg)
    # print(np.max(r_wta), r_wta)
    r_wta = np.clip(r_wta, -sneg, -spos)
    # print(np.max(r_wta), r_wta)
    #cv2.imwrite(Opath+'cen_f.png', (r_wta + max(spos, sneg)).astype(np.uint8))    
    # r_guide = np.zeros((fh, fw, 3))
    # r_guide[:,:,0] = r_wta + max(spos, sneg)
    # r_guide[:,:,1] = cv2.resize(imgR[:,:,0], (fw, fh))
    # r_guide[:,:,2] = (((imgR_o[:,:,0]/255.)**2)*255).astype(np.uint8)
    # r_wta = weighted_median_filter((r_wta + max(spos, sneg)).astype(np.uint8), imgR_o, [i for i in range(int(np.max(r_guide)+1))], 30, 0.0001) - max(spos, sneg)
    # cv2.imwrite('wmf_r'+n+'.png', (r_wta  + max(spos, sneg)).astype(np.uint8))
    '''
    print('wta minmax', np.min(wta), np.max(wta)) 
    print('r_wta minmax', np.min(r_wta), np.max(r_wta))  

    tmp_wta = cv2.medianBlur(np.uint8(wta + max(spos, sneg)), 3) + 1
    # wta = cv2.medianBlur(np.uint8(wta + max(spos, sneg)), 5) - max(spos, sneg)
    # r_wta = cv2.medianBlur(np.uint8(r_wta + max(spos, sneg)), 5) - max(spos, sneg)
    '''
    alll = np.zeros((fh,fw),dtype=np.uint8)
    allr = np.zeros((fh,fw),dtype=np.uint8)
    indx = wta.astype(np.int16)
    for y in range(fh):
        for x in range(fw):
            tmp = x - (indx[y, x])//scale
            if tmp < fw and x+1<fw:
                if indx[y, x] <= 0:
                    if abs(wta[y, x] - r_wta[y, tmp]) > scale:
                        alll[y, x] = 255

                else:
                    if abs(wta[y, x] - r_wta[y, tmp]) > scale:
                        allr[y, x] = 255

    alll_r = np.zeros((fh,fw),dtype=np.uint8)
    allr_r = np.zeros((fh,fw),dtype=np.uint8)
    indx = r_wta.astype(np.int16)
    for y in range(fh):
        for x in range(fw):
            if x + indx[y, x]//scale+1 < fw:
                if indx[y, x] <= 0:
                    if abs(r_wta[y, x] - wta[y, x + indx[y, x]//scale+1]) > scale:
                        alll_r[y, x] = 255

                else:
                    if abs(r_wta[y, x] - wta[y, x + indx[y, x]//scale+1]) > scale:
                        allr_r[y, x] = 255

    indx = r_wta.astype(np.int16)
    for y in range(fh):
        for x in range(fw):
            if x + indx[y, x]//scale < fw:
                if indx[y, x] <= 0:
                    if abs(r_wta[y, x] - wta[y, x + indx[y, x]//scale]) > 1:
                        pass

                else:
                    if abs(r_wta[y, x] - wta[y, x + indx[y, x]//scale]) > 1:
                        r_wta[y, x] = -max(spos, sneg) -1
                        pass

    wta = wta + max(spos, sneg)
    r_wta = r_wta + max(spos, sneg)
    r_wta = fillpixel_I(r_wta, int(np.max(wta)) + 1)
    wta = wta - max(spos, sneg)
    r_wta = r_wta - max(spos, sneg)
    r_wta[alll_r==255] = -1
    r_wta = fillpixel_I(r_wta, int(np.max(wta)) + 1)
    wta = wta + max(spos, sneg)
    r_wta = r_wta + max(spos, sneg)
    r_wta[allr_r==255] = -1
    r_wta = fillpixel(r_wta[:,::-1], int(np.max(wta)) + 1)[:,::-1]

    wta = wta - max(spos, sneg)
    r_wta = r_wta - max(spos, sneg)
    indx = r_wta.astype(np.int16)
    for y in range(fh):
        for x in range(fw):
            tmp = x - (indx[y, x])//scale-5
            if tmp < fw:
                if indx[y, x] <= 0:
                    if abs(wta[y, x] - r_wta[y, tmp]) > scale:
                        pass
                else:
                    if abs(wta[y, x] - r_wta[y, tmp]) > scale:
                        wta[y, x] = r_wta[y, x - indx[y, x]//scale]
    wta = wta + max(spos, sneg)
    r_wta = r_wta + max(spos, sneg)
    # print(now()-begin)
    # show([wta, r_wta])
    
    # allr = np.zeros((fh,fw),dtype=np.uint8)
    # wta = wta + max(spos, sneg)
    # wta = fillpixel(wta, int(np.max(wta)) + 1)
    # wta = fillpixel_I(wta, int(np.max(wta)) + 1)
    # show([wta])
    # ooo = np.zeros((fh,fw,3),dtype=np.uint8)
    # ooo[:,:,0] = wta.copy()
    # ooo[:,:,0][wta==-1] = 0
    # wta[wta!=-1] = 0
    # ooo[:,:,1] = np.bitwise_and(goodl,wta.astype(np.uint8)) + ooo[:,:,0]
    # ooo[:,:,2] = np.bitwise_and(goodr,wta.astype(np.uint8)) + ooo[:,:,0]
    

    # show([np.bitwise_and(goodl,wta.astype(np.uint8)), np.bitwise_and(goodr,wta.astype(np.uint8)), ooo])
    #cv2.imwrite(Opath+'be_filling.png', wta.astype(np.uint8))

    # r_wta = r_wta + max(spos, sneg)
    # print('after consistcy')
    # print('wta minmax', np.min(wta), np.max(wta)) 
    # print('r_wta minmax', np.min(r_wta), np.max(r_wta))   


    ri = 17
    epsi = 0.00001
    
    
    # t_wta = cv2.imread('weighted_median_filter'+n+'.png', 0)
    # if t_wta is None:
    # wtal = fillpixel(wta, int(np.max(wta)) + 1) #far
    
    # wtar = fillpixel_I(wta, int(np.max(wta)) + 1) #near
    # show([wtal, wtar, (wtal + wtar)//2, goodl, goodr])
    
    wta = cv2.medianBlur(wta.astype(np.uint8), 5)
    # cv2.imwrite('weighted_median_filter'+n+'.png', wta)
    # from roll_bi import *
    # guide = rolling_bilateral(imgL_o, 4)
    sss = wta.copy()
    ooo = sss.copy()
    ooo = cv2.normalize(imgL_o[:,:,0], ooo, 0,255,cv2.NORM_MINMAX)/255.
    # print(np.sum(ooo))
    # print(np.mean(fx_l))
    ooo = ((ooo**2)*255).astype(np.uint8)
    guide = imgL_o.copy()
    guide[:,:,0] = (cv2.normalize(wta.copy(),sss, 0,255,cv2.NORM_MINMAX).astype(np.uint8) + cv2.resize(fx_l, (fw, fh)))/2
    guide[:,:,1] = ooo
    guide[:,:,2] = (imgL_o[:,:,0]).astype(np.uint8)
    # print(np.mean(guide[:,:,0]),np.mean(guide[:,:,1]),np.mean(guide[:,:,2]))
    guide = cv2.cvtColor(guide, cv2.COLOR_HSV2RGB_FULL)
    wta = weighted_median_filter(wta.astype(np.uint8), guide, [i for i in range(int(np.max(wta)+1))], ri, epsi)
    wta = cv2.medianBlur(wta.astype(np.uint8), 5)
    cv2.imwrite(Opath+".png", cv2.normalize(wta,wta,0,255, cv2.NORM_MINMAX))
    from util import writePFM
    writePFM(Opath, wta.astype(np.float32))
    print(begin - now())

    exit()
