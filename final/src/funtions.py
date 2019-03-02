import numpy as np
import os, sys
import cv2
import time
import math
from numpy.matlib import repmat


def getlrdisp(left_img, right_img):
    left_img = cv2.equalizeHist(left_img)
    right_img = cv2.equalizeHist(right_img)

    compress = 150
    left_img = compress//2 + (left_img/255)*(255-compress)
    right_img = compress//2 + (right_img/255)*(255-compress)
    left_img = cv2.resize(left_img,(0,0) ,fx=3, fy=2, interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right_img,(0,0) ,fx=3, fy=2, interpolation=cv2.INTER_LINEAR)
    
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
        maxd = -6
    maxd = -maxd

    if len(number[:differ.index(0)]) != 0:
        mind = differ[np.argmax(number[:differ.index(0)])]
    else:
        mind = -6
    # if np.abs(maxd) > np.abs(mind)*2 or np.abs(mind) > np.abs(maxd)*2
    # print(maxd, mind)

    return maxd/2, mind/2

def guidedfilter(I, p, r, eps):
    
    h, w = p.shape
    
    N = boxfilter(np.ones((h,w)), r)
    
    mean_I_r = boxfilter(I[:,:,2], r) / N
    
    mean_I_g = boxfilter(I[:,:,1], r) / N
    mean_I_b = boxfilter(I[:,:,0], r) / N
    
    mean_p = boxfilter(p, r) / N

    mean_Ip_r = boxfilter(I[:,:,2]*p, r) / N
    mean_Ip_g = boxfilter(I[:,:,1]*p, r) / N
    mean_Ip_b = boxfilter(I[:,:,0]*p, r) / N

    conv_Ip_r = mean_Ip_r - mean_I_r * mean_p
    conv_Ip_g = mean_Ip_g - mean_I_g * mean_p
    conv_Ip_b = mean_Ip_b - mean_I_b * mean_p

    var_I_rr = boxfilter(I[:,:,2]*I[:,:,2], r) / N - mean_I_r*mean_I_r
    var_I_rg = boxfilter(I[:,:,2]*I[:,:,1], r) / N - mean_I_r*mean_I_g
    var_I_rb = boxfilter(I[:,:,2]*I[:,:,0], r) / N - mean_I_r*mean_I_b
    var_I_gg = boxfilter(I[:,:,1]*I[:,:,1], r) / N - mean_I_g*mean_I_g
    var_I_gb = boxfilter(I[:,:,1]*I[:,:,0], r) / N - mean_I_g*mean_I_b
    var_I_bb = boxfilter(I[:,:,0]*I[:,:,0], r) / N - mean_I_b*mean_I_b

    a = np.zeros((h,w,3))

    for y in range(0, h):
        for x in range(0, w):
            Sigma = [[var_I_rr[y,x], var_I_rg[y,x], var_I_rb[y,x]],
                    [var_I_rg[y,x], var_I_gg[y,x], var_I_gb[y,x]],
                    [var_I_rb[y,x], var_I_gb[y,x], var_I_bb[y,x]]]

            conv_Ip = np.array([conv_Ip_r[y,x], conv_Ip_g[y,x], conv_Ip_b[y,x]])

            a[y,x,:] = np.dot(conv_Ip, np.linalg.inv(Sigma + eps * np.eye(3)))

    b = mean_p - (a[:,:,0] * mean_I_r) - (a[:,:,1] * mean_I_g) - (a[:,:,2] * mean_I_b)

    q = (boxfilter(a[:,:,2], r) * I[:,:,0] + boxfilter(a[:,:,1], r) * I[:,:,1] + boxfilter(a[:,:,0], r) * I[:,:,2] + boxfilter(b, r)) / N
    return q

def fillPixelreference(final_label, max_disp):
    h, w = final_label.shape
    #final_label = final_label.astype(np.int32)
    fill_img = final_label.copy()
    
    fillVals = np.ones((h,)) * max_disp
    final_labels_filled = fill_img.copy()

    for i in range(w):
        curCol = fill_img[:, i].copy()
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        final_labels_filled[:,i] = curCol

    fillVals = np.ones((h,)) * max_disp
    final_labels_filled1 = fill_img.copy()
    
    for i in range(w)[::-1]:
        curCol = fill_img[:, i].copy()
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        final_labels_filled1[:,i] = curCol
    
    for i in range(h):
        for j in range(w):
            final_label[i,j] = min(final_labels_filled[i,j], final_labels_filled1[i,j])
    
    for i in range(h):
        for j in range(0,w-1)[::-1]:
            if final_label[i,j] == max_disp:
                final_label[i,j] = final_label[i,j+1]

    #cv2.imwrite("try_filllable.png", (final_label*16).astype(np.uint8))
    return final_label

def boxfilter(imSrc, r):
    h, w = imSrc.shape
    imDst = np.zeros((h,w))
    r = int(r)
    imCum = np.cumsum(imSrc, 0)
    imDst[0:r+1, :] = imCum[r:2*r+1,:]
    imDst[r+1:h-r,:] = imCum[2*r+1:h, :] - imCum[0:h-2*r-1,:]
    imDst[h-r:h, :] = np.repeat(imCum[h-1:h, :], r, axis=0) - imCum[h-2*r-1: h-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:w-r] = imCum[:, 2*r+2-1:w] - imCum[:, 0:w-2*r-1]
    imDst[:, w-r:w] = np.repeat(imCum[:, w-1:w], r, axis=1) - imCum[:, w-2*r-1:w-r-1]

    return imDst

def guidedfilter_color_precompute(I, r, eps):
    gf_obj = {}

    gf_obj['I'] = I
    gf_obj['r'] = r
    gf_obj['eps'] = eps

    hei, wid = I.shape[0], I.shape[1]

    gf_obj['N'] = boxfilter(np.ones((hei, wid), dtype=np.float), r)
    
    gf_obj['mean_I_r'] = boxfilter(I[:, :, 2], r) / gf_obj['N'] 
    gf_obj['mean_I_g'] = boxfilter(I[:, :, 1], r) / gf_obj['N'] 
    gf_obj['mean_I_b'] = boxfilter(I[:, :, 0], r) / gf_obj['N'] 
    
    gf_obj['var_I_rr'] = boxfilter(I[:, :, 2]*I[:, :, 2], r) / gf_obj['N'] - gf_obj['mean_I_r'] *  gf_obj['mean_I_r'] 
    gf_obj['var_I_rg'] = boxfilter(I[:, :, 2]*I[:, :, 1], r) / gf_obj['N'] - gf_obj['mean_I_r'] *  gf_obj['mean_I_g'] 
    gf_obj['var_I_rb'] = boxfilter(I[:, :, 2]*I[:, :, 0], r) / gf_obj['N'] - gf_obj['mean_I_r'] *  gf_obj['mean_I_b']  
    gf_obj['var_I_gg'] = boxfilter(I[:, :, 1]*I[:, :, 1], r) / gf_obj['N'] - gf_obj['mean_I_g'] *  gf_obj['mean_I_g'] 
    gf_obj['var_I_gb'] = boxfilter(I[:, :, 1]*I[:, :, 0], r) / gf_obj['N'] - gf_obj['mean_I_g'] *  gf_obj['mean_I_b'] 
    gf_obj['var_I_bb'] = boxfilter(I[:, :, 0]*I[:, :, 0], r) / gf_obj['N'] - gf_obj['mean_I_b'] *  gf_obj['mean_I_b']  
    
    gf_obj['invSigma'] = []
    
    for y in range(hei): 
        for x in range(wid):
            Sigma = [[gf_obj['var_I_rr'][y, x], gf_obj['var_I_rg'][y, x], gf_obj['var_I_rb'][y, x]],
                     [gf_obj['var_I_rg'][y, x], gf_obj['var_I_gg'][y, x], gf_obj['var_I_gb'][y, x]],
                     [gf_obj['var_I_rb'][y, x], gf_obj['var_I_gb'][y, x], gf_obj['var_I_bb'][y, x]]]
            
            gf_obj['invSigma'].append(np.linalg.inv(Sigma + eps * np.eye(3)))

    return gf_obj

def guidedfilter_color_runfilter(p, gf_obj):
    r = gf_obj['r']
    if len(p.shape) == 3:
        (hei, wid, ch) = p.shape
        p = p[:,:,0]
    else:
        (hei, wid) = p.shape
    
    tmp = np.zeros(p[:,:].shape)
    for y in range(hei):
        for x in range(wid):
            if p[y,x]:
                tmp[y,x] = 1

    mean_p = boxfilter(tmp, r) / gf_obj['N']

    mean_Ip_r = boxfilter(gf_obj['I'][:, :, 2]*tmp, r) / gf_obj['N']
    mean_Ip_g = boxfilter(gf_obj['I'][:, :, 1]*tmp, r) / gf_obj['N']
    mean_Ip_b = boxfilter(gf_obj['I'][:, :, 0]*tmp, r) / gf_obj['N']
   
    # % covariance of (I, p) in each local patch.
    cov_Ip_r = mean_Ip_r - gf_obj['mean_I_r'] * mean_p
    cov_Ip_g = mean_Ip_g - gf_obj['mean_I_g'] * mean_p
    cov_Ip_b = mean_Ip_b - gf_obj['mean_I_b'] * mean_p
    
    a = np.zeros((hei, wid, 3))
    count = 0
    for y in range(hei):
        for x in range(wid):
            cov_Ip = [cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]]
            a[y, x, :] = np.dot(cov_Ip, gf_obj['invSigma'][count])
            count += 1
          
    b = mean_p - a[:, :, 0] * gf_obj['mean_I_r'] - a[:, :, 1] * gf_obj['mean_I_g'] - a[:, :, 2] * gf_obj['mean_I_b']
    
    q = ( boxfilter(a[:, :, 0], r) * gf_obj['I'][:, :, 2]
        + boxfilter(a[:, :, 1], r) * gf_obj['I'][:, :, 1]
        + boxfilter(a[:, :, 2], r) * gf_obj['I'][:, :, 0]
        + boxfilter(b, r)) / gf_obj['N']
    
    return q

def weighted_median_filter(dispIn, imgGuide, vecDisps, r, epsilon = 0.01):
    imgGuide = imgGuide/255.

    if len(dispIn.shape) == 2:
        dispIn = np.expand_dims(dispIn, axis=2)
        dispIn = np.concatenate((dispIn, dispIn, dispIn), axis=2)
    dispOut  = np.zeros( dispIn.shape, dtype=np.float)
    imgAccum = np.zeros( dispIn.shape, dtype=np.float)

    gf_obj = guidedfilter_color_precompute(imgGuide, r, epsilon)
    
    for d in range(1,len(vecDisps)): 
        img = guidedfilter_color_runfilter((dispIn == vecDisps[d]).astype(np.float), gf_obj)
        img = np.expand_dims(img, axis=2)
        imgAccum = imgAccum + np.concatenate((img, img, img), axis=2)
        idxSelected = (imgAccum > 0.5) & (dispOut == 0)
        dispOut[idxSelected] = d
    
    dispOut = dispOut[:,:,0]
    return dispOut.astype(np.uint8)





def space_filter(sigma_s):
    size = int(round((3*sigma_s)*2 + 1))
    
    space_kernel = np.zeros((size,size))
    trans = size//2
    for i in range(space_kernel.shape[0]):
         for j in range(space_kernel.shape[1]):
             l = -((i-trans)**2+(j-trans)**2)/(2*(sigma_s**2))
             e = math.exp(l)
             space_kernel[i,j] = e
    return space_kernel

def range_filter(img,sigma_r,size):
    
    result_img = np.zeros(img.shape)
    borderType = cv2.BORDER_CONSTANT
    value = [0, 0, 0]
    pad_size = size//2
    #img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, borderType, None, value)
    img = np.pad(img, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), 'symmetric')
    if len(img.shape) == 2:
        img = np.reshape(img, (img.shape[0],img.shape[1],1))
    
    img = img/255
    all_range_kernel = []
    
    for i in range(img.shape[0]-(size-1))[:]:
        for j in range(img.shape[1]-(size-1))[:]:
            plane = img[i:i+size, j:j+size, :]
            
            neighbor_diff = plane - plane[pad_size,pad_size]
            
            neighbor_diff_square = np.square(neighbor_diff)
            
            channel_add = np.ones((size,size))
            for channel in range(neighbor_diff_square.shape[2]):
                channel_add = channel_add * np.exp((-1)*neighbor_diff_square[:,:,channel]/(2 * (sigma_r**2)))
            
            all_range_kernel.append(channel_add)

    all_range_kernel = np.array(all_range_kernel)
    
    return all_range_kernel
    
    
    
def bilateral_filter(space_kernel, all_range_kernel, img):
    size = space_kernel.shape[0]
    pad_size = size//2
    if len(img.shape) != 3:
        img = np.expand_dims(img, axis=2)
    result_img = np.zeros(img.shape)
    borderType = cv2.BORDER_CONSTANT
    value = [0, 0, 0]
    #img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, borderType, None, value)
    img = np.pad(img, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), 'symmetric')
    img = img/255
    count = 0
    
    for i in range(img.shape[0]-(size-1))[:]:
        for j in range(img.shape[1]-(size-1))[:]:
            plane = img[i:i+size, j:j+size, :]
            range_kernel = all_range_kernel[count]
            count += 1
            G_filter = space_kernel * range_kernel
            bilateral_kernel = G_filter/np.sum(G_filter) 
            bilateral_kernel = bilateral_kernel.reshape((bilateral_kernel.shape[0], bilateral_kernel.shape[1],1))
            mul = np.multiply(plane, bilateral_kernel)
            kernel_sum = np.sum(np.sum(mul, axis=0), axis=0)
            result_img[i,j] = kernel_sum
            
    result_img = result_img * 255
    
    return result_img


def equal(L, R):
    L = L.astype(np.float)
    R = R.astype(np.float)
    if len(L.shape) == 3:
        ch = 3
    else:
        ch = 1
    L = np.reshape(L, L.shape + (ch,))
    R = np.reshape(R, R.shape + (ch,))
    for i in range(ch):
        R[:,:,i] = R[:,:,i] - (np.mean(R[:,:,i]) - np.mean(L[:,:,i]))
        R[:,:,i] = np.clip(R[:,:,i], 0, 255)
        veo = np.var(L[:,:,i])/np.var(R[:,:,i])
        R[:,:,i] = R[:,:,i]*(veo)**0.5
        
    
    return L.astype(np.uint8), R.astype(np.uint8)

def hist(img):
    ch = 1
    if len(img.shape) == 3:
        ch = 3
    img = img.reshape(img.shape[:2]+(ch,))
    for i in range(ch):
        tmpl = img[:,:,i]
        # l = (l - np.min(l))/(np.max(l) - np.min(l))
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # cl1 = clahe.apply(tmpl)
        # l[:,:,i] = cl1
        # tmpl = cv2.medianBlur(tmpl, 3)
        equl = cv2.equalizeHist(tmpl)
        img[:,:,i] = equl
        
    return img

def fillpixel(label, max_disp): #far
    h, w = label.shape
    fill_img = label.copy()
    fill_imgR = cv2.flip(label, 1).copy()

    fillVals = np.ones((h,)) * max_disp
    fill_labels = fill_img.copy()
    # scan from left to right
    for i in range(w):
        curCol = fill_img[:, i].copy()
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        fill_labels[:,i] = curCol

    return fill_labels
    
    fillVals = np.ones((h,)) * max_disp
    fill_labelsR = fill_imgR.copy()
    # scan from right to left
    for i in range(w):
        curCol = fill_imgR[:, i].copy()      
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        fill_labelsR[:,i] = curCol
    fill_labelsR = cv2.flip(fill_labelsR, 1) # flip back to do lr consistancy
    
    # do left right consistancy
    for y in range(h):
        for x in range(w):
            label[y,x] = max(fill_labels[y,x]*0, fill_labelsR[y,x])
            # label[y,x] = min(fill_labels[y,x], fill_labelsR[y,x])
    
    # hole filling
    for y in range(h):
        for x in range(1,w)[::-1]:
            if label[y,x] == max_disp and x < w - 1:
                label[y,x] = label[y,x+1]

    return label

def fillpixel_I(label, max_disp):
    h, w = label.shape
    fill_img = label.copy()
    fill_imgR = cv2.flip(label, 1).copy()
    """
    fillVals = np.ones((h,)) * max_disp
    fill_labels = fill_img.copy()
    # scan from left to right
    for i in range(w):
        curCol = fill_img[:, i].copy()
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        fill_labels[:,i] = curCol
    """
    fillVals = np.ones((h,)) * max_disp
    fill_labelsR = fill_imgR.copy()
    # scan from right to left
    for i in range(w):
        curCol = fill_imgR[:, i].copy()
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]


        fill_labelsR[:,i] = curCol
    fill_labelsR = cv2.flip(fill_labelsR, 1) # flip back to do lr consistancy
    return fill_labelsR
    # do left right consistancy
    for y in range(h):
        for x in range(w):
            label[y,x] = max(fill_labels[y,x], fill_labelsR[y,x]*0)
            # label[y,x] = min(fill_labels[y,x], fill_labelsR[y,x])
    
    # hole filling
    for y in range(h):
        for x in range(1,w)[::-1]:
            if label[y,x] == max_disp and x < w - 1:
                label[y,x] = label[y,x+1]

    return label

class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)