import cv2
import numpy as np

def p(ssss):
    print(ssss)
    exit()

def boxfilter(imSrc, r):
    (hei, wid) = imSrc.shape
    imDst = np.zeros(imSrc.shape)
    # print(r)
    r = int(r)
    imCum = np.cumsum(imSrc, axis=0)
    imDst[:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:hei-r, :] = imCum[2*r+1:hei, :] - imCum[:hei-2*r-1, :]
    imDst[hei-r:hei, :] = np.tile(imCum[hei-1, :], (r, 1)) - imCum[hei-2*r-1:hei-r-1, :] #repmat(imCum[hei-1, :], [r-1, 0]) - imCum[hei-2*r-1:hei-r-2, :]

    imCum = np.cumsum(imDst, axis=1)
    imDst[:, :r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, :wid-2*r-1]
    imDst[:, wid-r:wid] = np.tile(imCum[:, wid-1], (r, 1)).T - imCum[:, wid-2*r-1:wid-r-1] # np.matlib.repmat(imCum[:, wid], [0, r-1]) - imCum[:, wid-2*r:wid-r-2]
    return imDst



def guidedfilter_color_precompute(I, r, eps):
    gfobj = {}

    gfobj['I'] = I
    gfobj['r'] = r
    gfobj['eps'] = eps

    hei = I.shape[0]
    wid = I.shape[1]
    gfobj['N'] = boxfilter(np.ones((hei, wid), dtype=np.float), r)# - 1.0e-10
    N = gfobj['N']

    R = boxfilter(I[:, :, 2], r)
    G = boxfilter(I[:, :, 1], r)
    B = boxfilter(I[:, :, 0], r)
    gfobj['mean_I_r'] = R = R / N #R = boxfilter(I[:, :, 0], r) / gfobj['N']
    gfobj['mean_I_g'] = G = G / N #G = boxfilter(I[:, :, 1], r) / gfobj['N']
    gfobj['mean_I_b'] = B = B / N #B = boxfilter(I[:, :, 2], r) / gfobj['N']

    RR =  boxfilter(I[:, :, 2]*I[:, :, 2], r)
    RG =  boxfilter(I[:, :, 2]*I[:, :, 1], r)
    RB =  boxfilter(I[:, :, 2]*I[:, :, 0], r)
    GG =  boxfilter(I[:, :, 1]*I[:, :, 1], r)
    GB =  boxfilter(I[:, :, 1]*I[:, :, 0], r)
    BB =  boxfilter(I[:, :, 0]*I[:, :, 0], r)
    gfobj['var_I_rr'] = RR = RR / N - gfobj['mean_I_r'] *  gfobj['mean_I_r'] #  / gfobj['N'] - gfobj['mean_I_r'] *  gfobj['mean_I_r']
    gfobj['var_I_rg'] = RG = RG / N - gfobj['mean_I_r'] *  gfobj['mean_I_g'] #  / gfobj['N'] - gfobj['mean_I_r'] *  gfobj['mean_I_g'] 
    gfobj['var_I_rb'] = RB = RB / N - gfobj['mean_I_r'] *  gfobj['mean_I_b'] #  / gfobj['N'] - gfobj['mean_I_r'] *  gfobj['mean_I_b'] 
    gfobj['var_I_gg'] = GG = GG / N - gfobj['mean_I_g'] *  gfobj['mean_I_g'] #  / gfobj['N'] - gfobj['mean_I_g'] *  gfobj['mean_I_g'] 
    gfobj['var_I_gb'] = GB = GB / N - gfobj['mean_I_g'] *  gfobj['mean_I_b'] #  / gfobj['N'] - gfobj['mean_I_g'] *  gfobj['mean_I_b'] 
    gfobj['var_I_bb'] = BB = BB / N - gfobj['mean_I_b'] *  gfobj['mean_I_b'] #  / gfobj['N'] - gfobj['mean_I_b'] *  gfobj['mean_I_b'] 

    gfobj['invSigma'] = []
    
    for y in range(hei):
        for x in range(wid):
            Sigma = [[gfobj['var_I_rr'][y, x], gfobj['var_I_rg'][y, x], gfobj['var_I_rb'][y, x]],
                     [gfobj['var_I_rg'][y, x], gfobj['var_I_gg'][y, x], gfobj['var_I_gb'][y, x]],
                     [gfobj['var_I_rb'][y, x], gfobj['var_I_gb'][y, x], gfobj['var_I_bb'][y, x]]]
            gfobj['invSigma'].append(np.linalg.inv(Sigma + eps * np.eye(3)))
    return gfobj

def guidedfilter_color_runfilter(p, gfobj, d):
    r = gfobj['r']
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

    mean_p = boxfilter(tmp, r) / gfobj['N']

    I = gfobj['I']
    N = gfobj['N']

    p = tmp
    R = boxfilter(I[:, :, 2]*p, r)
    G = boxfilter(I[:, :, 1]*p, r)
    B = boxfilter(I[:, :, 0]*p, r)
    mean_Ip_r = R = R / N
    mean_Ip_g = G = G / N
    mean_Ip_b = B = B / N

    cov_Ip_r = R = mean_Ip_r - gfobj['mean_I_r'] * mean_p
    cov_Ip_g = G = mean_Ip_g - gfobj['mean_I_g'] * mean_p
    cov_Ip_b = B = mean_Ip_b - gfobj['mean_I_b'] * mean_p

    a = np.zeros((hei, wid, 3))
    count = 0
    for y in range(hei):#y=1:hei
        for x in range(wid):#x=1:wid
            cov_Ip = [cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]]
            a[y, x, :] = np.dot(cov_Ip, gfobj['invSigma'][count])
            count += 1
    
    b = mean_p - a[:, :, 0] * gfobj['mean_I_r'] - a[:, :, 1] * gfobj['mean_I_g'] - a[:, :, 2] * gfobj['mean_I_b']
    
    q = ( boxfilter(a[:, :, 0], r) * I[:, :, 2]
        + boxfilter(a[:, :, 1], r) * I[:, :, 1]
        + boxfilter(a[:, :, 2], r) * I[:, :, 0]
        + boxfilter(b, r)) / N

    return q

def weighted_median_filter(dispIn, imgGuide, vecDisps, r, epsilon = 0.01):
    imgGuide = imgGuide/255.

    if len(dispIn.shape) == 2:
        dispIn = np.expand_dims(dispIn, axis=2)
        dispIn = np.concatenate((dispIn, dispIn, dispIn), axis=2)
    dispOut  = np.zeros( dispIn.shape, dtype=np.float)
    imgAccum = np.zeros( dispIn.shape, dtype=np.float)

    gfobj = guidedfilter_color_precompute(imgGuide, r, epsilon)
    
    for d in range(1,np.size(vecDisps)): 
        
        img01 = guidedfilter_color_runfilter((dispIn == vecDisps[d]).astype(np.float), gfobj, d)
   
        img01 = np.expand_dims(img01, axis=2)
        
        imgAccum = imgAccum + np.concatenate((img01, img01, img01), axis=2)
        
        idxSelected = (imgAccum > 0.5) & (dispOut == 0)
        dispOut[idxSelected] = d
    
    
    dispOut = dispOut[:,:,0]
    return dispOut.astype(np.uint8)
    
def guided_filter(guideImg, p, gfobj):    
    h, w = p.shape
    N = gfobj['N'] # boxfilter(np.ones((h,w)), r)    
    r = gfobj['r']
    I = gfobj['I']

    gfobj['mean_I_r'] = boxfilter(I[:,:,2], r) / N    
    gfobj['mean_I_g'] = boxfilter(I[:,:,1], r) / N
    gfobj['mean_I_b'] = boxfilter(I[:,:,0], r) / N
    
    gfobj['mean_p'] = boxfilter(p, r) / N
    gfobj['mean_Ip_r'] = boxfilter(I[:,:,2] * p, r) / N
    gfobj['mean_Ip_g'] = boxfilter(I[:,:,1] * p, r) / N
    gfobj['mean_Ip_b'] = boxfilter(I[:,:,0] * p, r) / N

    gfobj['conv_Ip_r'] = gfobj['mean_Ip_r'] - gfobj['mean_I_r'] * gfobj['mean_p']
    gfobj['conv_Ip_g'] = gfobj['mean_Ip_g'] - gfobj['mean_I_g'] * gfobj['mean_p']
    gfobj['conv_Ip_b'] = gfobj['mean_Ip_b'] - gfobj['mean_I_b'] * gfobj['mean_p']

    var_I_rr = boxfilter(I[:,:,2] * I[:,:,2], r) / N - gfobj['mean_I_r'] * gfobj['mean_I_r']
    var_I_rg = boxfilter(I[:,:,2] * I[:,:,1], r) / N - gfobj['mean_I_r'] * gfobj['mean_I_g']
    var_I_rb = boxfilter(I[:,:,2] * I[:,:,0], r) / N - gfobj['mean_I_r'] * gfobj['mean_I_b']
    var_I_gg = boxfilter(I[:,:,1] * I[:,:,1], r) / N - gfobj['mean_I_g'] * gfobj['mean_I_g']
    var_I_gb = boxfilter(I[:,:,1] * I[:,:,0], r) / N - gfobj['mean_I_g'] * gfobj['mean_I_b']
    var_I_bb = boxfilter(I[:,:,0] * I[:,:,0], r) / N - gfobj['mean_I_b'] * gfobj['mean_I_b']
    
    a = np.zeros((h,w,3))

    for y in range(0, h):
        for x in range(0, w):
            Sigma = [[var_I_rr[y,x], var_I_rg[y,x], var_I_rb[y,x]],
                    [var_I_rg[y,x], var_I_gg[y,x], var_I_gb[y,x]],
                    [var_I_rb[y,x], var_I_gb[y,x], var_I_bb[y,x]]]

            conv_Ip = np.array([gfobj['conv_Ip_r'][y,x], gfobj['conv_Ip_g'][y,x], gfobj['conv_Ip_b'][y,x]])

            a[y,x,:] = np.dot(conv_Ip, np.linalg.inv(Sigma + gfobj['eps'] * np.eye(3)))

    b = gfobj['mean_p'] - a[:,:,0] * gfobj['mean_I_r'] - a[:,:,1] * gfobj['mean_I_g'] - a[:,:,2] * gfobj['mean_I_b']

    q = (boxfilter(a[:,:,2], r) * guideImg[:,:,0] + boxfilter(a[:,:,1], r) * guideImg[:,:,1] + boxfilter(a[:,:,0], r) * guideImg[:,:,2] + boxfilter(b, r)) / N
    return q

if __name__ == '__main__':
        
        

    num_disp = 15
    disp_scale = 16

    imgGuide = cv2.imread('./tsukuba_left.png')
    dispMapInput = cv2.imread('./tsukuba_boxagg.png')/disp_scale

    eps = 0.01**2
    # r = ceil(max(size(imgGuide, 1), size(imgGuide, 2)) / 40)
    r = np.ceil(max(imgGuide.shape[0], imgGuide.shape[1]) / 40)

    dispMapOutput = weighted_median_filter(dispMapInput.astype(np.uint8), imgGuide.astype(np.uint8), [i for i in range(num_disp+1)], r, eps)
    print(dispMapOutput)
    dispMapOutput = cv2.medianBlur(dispMapOutput*16,5)
    cv2.imwrite('wwww.png', dispMapOutput)
    # dispMapOutput = medfilt2(dispMapOutput,[3,3])




