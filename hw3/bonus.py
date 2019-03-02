import cv2
import numpy as np 
from time import time as now 

k = 3
tra = None
orixy = None
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print(v.shape[0], N)
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # --- version 2
    A = []
    for u_i, v_i in zip(u, v):
        A.append([u_i[0], u_i[1], 1, 0, 0, 0, -u_i[0]*v_i[0], -u_i[1]*v_i[0], -v_i[0]])
        A.append([0, 0, 0, u_i[0], u_i[1], 1, -u_i[0]*v_i[1], -u_i[1]*v_i[1], -v_i[1]])
    A = np.array(A)
    ATA = np.dot(A.T, A)
    U, z, V = np.linalg.svd(ATA, full_matrices=True)
    H = U[:, -1]
    return H.reshape((3,3))/H[-1]

pantapoint = None
# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    global tra
    global orixy
    global pantapoint
    
    
    
    print(corners)
    
    # m = solve_homography(pantapoint, corners)
    m = solve_homography(corners, pantapoint)
    # print(corners)
    inv = solve_homography(pantapoint, corners)
    # print(np.concatenate(()))
    # print(np.dot(inv, np.array([[0,0,1],[0,img.shape[1],1],[img.shape[0],img.shape[1],1],[img.shape[0],0,1]]).T))
    bound = np.dot(inv, np.array([[0,0,1],[0,img.shape[1]-1,1],[img.shape[0]-1,img.shape[1]-1,1],[img.shape[0]-1,0,1]]).T)
    print(bound/bound[2,:])
    bound = bound/bound[2,:]
    bound = bound[:2,:].astype(np.int).T
    corners = bound.reshape(4,1,2)
    # print(corners)
    # print(bound)
    # exit()
    
    # print(m)
    beg = now()
    
    tra1 = np.dot(m,tra)
    # print(tra)
    tra1 = (tra1/tra1[2,:]).astype(np.int16)
    print(now() - beg)

    tra1[0,:] = np.clip(tra1[0,:], 0, canvas.shape[1]-1)
    tra1[1,:] = np.clip(tra1[1,:], 0, canvas.shape[0]-1)
    tra1 = tra1[:2,:].T
    mask = np.zeros(canvas.shape, dtype=np.uint8)
    cv2.drawContours(mask, [corners], -1, (255, 255, 255), thickness=-1)
    white = 255*3

    mask = mask[:,:,0]
    invmask = cv2.bitwise_not(mask)
    front = cv2.warpPerspective(img, inv, (canvas.shape[1], canvas.shape[0]))
    front = cv2.bitwise_and(front, front, mask=mask)
    back = cv2.bitwise_and(canvas, canvas, mask=invmask)
    result = cv2.add(front, back)
    for i in range(np.min(bound[:,1], axis=0), np.max(bound[:,1], axis=0)):
        break
        for j in range(np.min(bound[:,0], axis=0), np.max(bound[:,0], axis=0)):
            if np.sum(mask[i,j,:]) == white:
                # print(np.sum(mask[i,j]))
                x, y, c = np.dot(m,np.array([j,i,1]))
                x, y = np.rint(x/c).astype(np.int16), np.rint(y/c).astype(np.int16)
                # print(img.shape)
                if x < img.shape[0] and y < img.shape[1]:
                    canvas[i,j,:] = img[y,x,:]
                # exit()
    # return canvas
    return result#cv2.warpPerspective(img, inv, (canvas.shape[1], canvas.shape[0]))
    # exit()
    # print(now() - beg)
    # TODO: some magic
def cross(l1_p1, l1_p2, l2_p1, l2_p2):
    x1 = np.linalg.solve(np.vstack([l1_p1, l1_p2]), np.array([1,1]))
    bc, c = x1
    c1 = 1/c
    b1 = -bc * c1
    print(b1, c1)
    x2 = np.linalg.solve(np.vstack([l2_p1, l2_p2]), np.array([1,1]))
    bc, c = x2
    c2 = 1/c
    b2 = -bc * c2
    print(b2, c2)
    xy, y = np.linalg.solve(np.matrix([[b1, c1], [b2, c2]]), np.array([1,1]))
    y = 1/y
    x = xy * y
    return np.array([[[x,y]]]).reshape((1,1,2))
    # return x, y
def recoverImg(image):
    global kernel
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,110,255,0)

    thresh = cv2.morphologyEx(255-thresh, cv2.MORPH_GRADIENT, kernel)

    img_fc ,contours, hierarchy = cv2.findContours(255-thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    found = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c >= 5:
            found.append(i)
    print(found)
    panta = np.array([[[0,0]]])
    for i in found:
        area = cv2.approxPolyDP(contours[i],20,True)
        panta = np.concatenate((panta,area),axis=0)
    # print(panta[1:,])
    # area = cv2.approxPolyDP(panta[1:,],100,True)
    area = cv2.convexHull(panta[1:,])
    area = cv2.approxPolyDP(area,50,True)
    print(area)
    lineLen = []
    for i in range(-1, 4):
        area[i] - area[i+1]
        print(np.sum((area[i] - area[i+1])**2))
        lineLen.append(np.sum((area[i] - area[i+1])**2))
    min2 = np.argsort(lineLen)[:2]
    # print(np.sum((area[min2[0]] - cc)**2), np.sum((area[min2[1]-1] - cc)**2))
    # exit()
    # print(area[min2[0]-2], c)
    # exit()
    # cv2.circle(image,(area[min2[0]-2][:,0],area[min2[0]-2][:,1]),10,(0,0,255),5)
    # print(c1,c2)
 
    outpoint = np.rint(cross(area[min2[0]-1], area[min2[0]], area[min2[1]-1], area[min2[1]])).astype(int)
    tmp = np.zeros(image.shape, dtype=np.uint8)
    croy = outpoint[:,:,0][0,0]
    crox = outpoint[:,:,1][0,0]
    print(crox,croy)
    bias = 50
    imgray = cv2.cvtColor(image[crox-bias:crox+bias,croy-bias:croy+bias,:],cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,110,255,0)

    # thresh = cv2.morphologyEx(255-thresh, cv2.MORPH_GRADIENT, kernel)

    # contours = cv2.approxPolyDP(contours,20,True)
    
    # exit()
    tmp[crox-bias:crox+bias,croy-bias:croy+bias,:] = cv2.cvtColor(255-thresh, cv2.COLOR_GRAY2BGR)

    img_fc ,contours, hierarchy = cv2.findContours(tmp[:,:,0].astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    farestpoint = None
    farest = 0
    for cnt in contours:
        cnt = cv2.approxPolyDP(cnt,3,True)
        for p in cnt:
            if np.sum((area[min2[0]-2] - p)**2) >= farest:
                farestpoint = p
                farest = np.sum((area[min2[0]-2] - p)**2)
            print(np.sum((area[min2[0]-2] - p)**2))
        # exit()
        # cv2.drawContours(tmp, [cnt], 0, (0, 0, 255), 3)    
        # cv2.drawContour()
    # cv2.circle(tmp,(farestpoint[:,0],farestpoint[:,1]),10,(55,255,155),4)
    
    # print(len(contours))

    # print(outpoint)

    related = (area - outpoint).reshape(area.shape[0],2)
    

    # print(np.sum(related**2, axis = 1))
    related = np.argsort(np.sum(related**2, axis = 1))
    # print(related)
    # area = np.vstack([area, farestpoint.reshape((1,1,2))])
    # area = cv2.convexHull(area)
    # area = cv2.approxPolyDP(area,50,True)
    
    # print(area.shape,farestpoint.shape)
    
    area = np.vstack([[area[related[4]]], [area[related[4]-1]], [farestpoint.astype(int)], [area[related[4]-4]]])
    # print(area.shape)

    # area = cv2.approxPolyDP(area,150,True)
    # area = cv2.contourArea(panta[1:,])
    # cv2.drawContours(image, [area], 0, (0, 255, 0), 3)
    # cv2.drawContours(image, [area], 0, (0, 255, 0), 3)
    # mask = np.zeros(image.shape, dtype=np.uint8)
    # cv2.drawContours(mask, [area], -1, (255, 255, 255), thickness=-1)
    
    return image, np.array([co[0] for co in area])

cap = cv2.VideoCapture('./input/ar_marker.mp4')
cap2 = cv2.VideoCapture('./NAO.mp4')

# ret, frame = cap.read()
# cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('frame', frame.shape[1]//4, frame.shape[0]//4)
# print(frame.shape[1], frame.shape[0])
# exit()

larg = 2
qrcode = cv2.imread('marker.png', cv2.IMREAD_COLOR)
sticker = cv2.imread('./input/1.png', cv2.IMREAD_COLOR)
qrcode = cv2.resize(qrcode, (int(qrcode.shape[1]*larg),int(qrcode.shape[0]*larg)))
sticker = cv2.resize(sticker, (int(qrcode.shape[1]),int(qrcode.shape[0])))

h, w, ch = qrcode.shape
pixelNum = h*w
orixy = np.zeros((pixelNum,2)).astype(int)
d0 = np.arange(h)
d1 = np.arange(w)

orixy[:,0] = np.tile(d1, h)
orixy[:,1] = np.repeat(d0, w)

trax = orixy[:,0].reshape(1,pixelNum)
tray = orixy[:,1].reshape(1,pixelNum)
tra1 = np.ones((1,pixelNum))

tra = np.concatenate((trax,tray,tra1), axis = 0)
# tra = np.matrix(tra)
_, pantapoint = recoverImg(qrcode)
fourcc = cv2.VideoWriter_fourcc('m','p', '4', 'v')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (3840//4, 2160//4))

while(cap.isOpened()):
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if ret2:
        frame2 = cv2.resize(frame2[:,(frame2.shape[1]-frame2.shape[0])//2:frame2.shape[1] - (frame2.shape[1]-frame2.shape[0])//2,:], (qrcode.shape[0],qrcode.shape[1]))
    # print(frame2.shape)
    # exit()

    if ret:
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame.shape)
    # frame = cv2.resize(frame, (frame.shape[1]//8,frame.shape[0]//8))
    # print(frame.shape)
        frame, point = recoverImg(frame)
        frame = transform(frame2, frame, point)
        
        out.write(cv2.resize(frame, (frame.shape[1]//4,frame.shape[0]//4)))
        cv2.imshow('frame',frame)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()