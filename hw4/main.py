import numpy as np
import cv2
import time
from wmf import boxfilter, guidedfilter_color_runfilter, guidedfilter_color_precompute, guided_filter

def fillpixel(Il, label, max_disp):
    h, w = label.shape
    fill_img = label.copy()
    fill_imgR = cv2.flip(label, 1).copy()

    fillVals = np.ones((h,)) * max_disp
    fill_labels = fill_img.copy()
    # scan from left to right
    for i in range(w):
        curCol = fill_img[:, i]
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        fill_labels[:,i] = curCol
    
    fillVals = np.ones((h,)) * max_disp
    fill_labelsR = fill_imgR.copy()
    # scan from right to left
    for i in range(w):
        curCol = fill_imgR[:, i]        
        curCol[curCol==-1] = fillVals[curCol ==-1]
        fillVals[curCol !=-1] = curCol[curCol!=-1]
        fill_labelsR[:,i] = curCol
    fill_labelsR = cv2.flip(fill_labelsR, 1) # flip back to do lr consistancy
    
    # do left right consistancy
    for y in range(h):
        for x in range(w):
            label[y,x] = min(fill_labels[y,x], fill_labelsR[y,x])
    
    # hole filling
    for y in range(h):
        for x in range(1,w)[::-1]:
            if label[y,x] == max_disp:
                label[y,x] = label[y,x+1]

    return label



def computeDisp(Il, Ir, max_disp):
    from wmf import weighted_median_filter

    scale_factor = 256 // max_disp
    h, w, ch = Il.shape
    print(h, w, ch)
    Il = Il.astype(np.float32)/255
    Ir = Ir.astype(np.float32)/255
    Il_f = cv2.flip(Il, 1)
    Ir_f = cv2.flip(Ir, 1)
    Il_g = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_g = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    

    fx_l = np.gradient(Il_g)[1]
    fx_r = np.gradient(Ir_g)[1]
    fx_l = fx_l + 0.5
    fx_r = fx_r + 0.5    
    fx_l_f = cv2.flip(fx_l, 1)
    fx_r_f = cv2.flip(fx_r, 1)    

    max_color = 7/255
    max_grad = 2/255
    max_Border = 3/255
    
    gamma = 0.17
    gamma_c = 0.1
    gamma_d = 9
    r_median = 19
    eps = 0.00001
    r = 6
    # >>> Cost computation
    tic = time.time()

    # TODO: Compute matching cost from Il and Ir
    # ======================================l-r
    labels = np.zeros((h, w), dtype=np.uint8)
    disparity = []

    for i in range(1, max_disp+2):
        # get disp of Ir
        tmp = np.ones((h, w+(i), ch)) * max_Border
        tmp[:,i:,:] = Ir
        tmp[:,:i,:] = Ir[:,0,:].reshape(h, 1, ch)
        tmp = tmp[:,:w,:]
        # compute cost of Il
        p_color = np.sum(np.abs(Il - tmp), axis=2)/3
        p_color = np.clip(p_color, 0, max_color)

        # get disp of fx_r
        tmpg = np.zeros((h, w+(i)))
        tmpg[:,i:] = fx_r
        tmpg[:,:i] = fx_r[:,0].reshape(h, 1)
        tmpg = tmpg[:,:w]
        # compute cost of fx_l
        p_grad = np.abs(fx_l - tmpg)
        p_grad = np.clip(p_grad, 0, max_grad)
        
        p = gamma*p_color + (1-gamma)*p_grad

        disparity.append(p)
    
    disparity = np.array(disparity)
    # ======================================r-l
    labelsR = np.zeros((h, w), dtype=np.uint8)
    disparityR = []
    
    for i in range(1, max_disp+2):
        # get disp of Il_f
        tmp = np.ones((h, w+(i), ch)) * max_Border
        tmp[:,i:,:] = Il_f
        tmp[:,:i,:] = Il_f[:,0,:].reshape(h, 1, ch)
        tmp = tmp[:,:w,:]
        # compute cost of Ir_f
        p_color = np.sum(np.abs(Ir_f - tmp), axis=2)/3
        p_color = np.clip(p_color, 0, max_color)

        # get disp of fx_l_f
        tmpg = np.zeros((h, w+(i)))
        tmpg[:,i:] = fx_l_f
        tmpg[:,:i] = fx_l_f[:,0].reshape(h, 1)
        tmpg = tmpg[:,:w]
        # compute cost of fx_r_f
        p_grad = np.abs(fx_r_f - tmpg)
        p_grad = np.clip(p_grad, 0, max_grad)

        p = gamma*p_color + (1-gamma)*p_grad
        
        disparityR.append(p)
    disparityR = np.array(disparityR)
    
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    labels = None
    labelsR = None
    
    # precompute for guided filter
    gfobj = guidedfilter_color_precompute(Il, r, eps)
    gfobjR = guidedfilter_color_precompute(Ir_f, r, eps)

    # Refine cost by doing guided filter
    for i in range(max_disp+1):
        p = disparity[i]
        q = guided_filter(Il, p, gfobj)
        disparity[i] = q
        
        pR = disparityR[i]
        qR = guidedfilter_color_runfilter((pR == i).astype(np.uint8), gfobjR, i)
        qR = guided_filter(Ir_f, pR, gfobjR)
        disparityR[i] = qR[:,::-1]
        
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    # Just do winner-takes-all
    labels = np.zeros((h, w))
    for j in range(h):
        for i in range(w):
            labels[j, i] = np.argmin(disparity[:,j, i])+1
    
    labelsR = np.zeros((h, w))
    for j in range(h):
        for i in range(w):
            labelsR[j, i] = np.argmin(disparityR[:,j, i])+1

    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()

    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    # Before Left-right consistency, use median filter to remove pepper and solt noises
    # And do a larg windows weighted_median_filter to make a hole filling reference pic.
    # And filling the holes under threshold with reference pic.
    ri = 30
    epsi = 0.00001
    minth = max_disp//6 # threshold of hole
    labels = cv2.medianBlur(np.uint8(labels)*scale_factor, 3)/scale_factor
    # Left reference pic
    t_labels = weighted_median_filter(labels, Il*255, [i for i in range(max_disp+1)], ri, epsi)
    # Filling left
    labels[labels<=minth] = t_labels[labels<=minth]

    # cv2.imwrite(name + '_ls.png', np.uint8(labels*scale_factor))

    labelsR = cv2.medianBlur(np.uint8(labelsR)*scale_factor, 3)/scale_factor
    # Right reference pic
    t_labelsR = weighted_median_filter(labelsR, Ir*255, [i for i in range(max_disp+1)], ri, epsi)
    # Filling Right
    labelsR[labelsR<=minth] = t_labelsR[labelsR<=minth]

    # cv2.imwrite(name + '_rs.png', np.uint8(labelsR*scale_factor))

    Y = np.repeat(np.array([[i for i in range(1,h+1)]]).T , w, axis=1)
    X = np.repeat(np.array([[i for i in range(1,w+1)]]) , h, axis=0)
    X = X - labels

    for y in range(X.shape[0]):
        for x in range(X.shape[1]):
            if X[y, x] < 1:
                X[y, x] = 1
    
    indices = Y + (X - 1) * h
    right_ind = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            index = indices[y, x]
            tmpx = int(index/h)
            tmpy = int(index%h)-1
            
            right_ind[y, x] = labelsR[tmpy, tmpx]
    
    # Do lr consistance check
    for y in range(h):
        for x in range(w):
            if np.abs(labels[y, x] - right_ind[y, x]) >= 1:
                labels[y, x] = -1

    # Filling the -1 pixels of lr consistance
    labels = fillpixel(Il, labels, max_disp)

    # cv2.imwrite(name + '_o.png', np.uint8(labels * scale_factor))
    
    
    epsi = 0.001
    ri = np.ceil(max(Il.shape[0], Il.shape[1]) / 50)
    # refine by weighted median filter
    labels = weighted_median_filter(labels.astype(np.uint8), (Il*255).astype(np.uint8), [i for i in range(max_disp+1)], 7, epsi)
    labels = cv2.medianBlur(labels.astype(np.uint8),3)
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))

if __name__ == '__main__':
    main()
