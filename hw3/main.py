import numpy as np
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print(v.shape[0], N)
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # --- version 1
    """
    # A = np.zeros((2*N, 8))
    A = []
    for u_i, v_i in zip(u, v):
        A.append([u_i[0], u_i[1], 1, 0, 0, 0, -u_i[0]*v_i[0], -u_i[1]*v_i[0]])
        A.append([0, 0, 0, u_i[0], u_i[1], 1, -u_i[0]*v_i[1], -u_i[1]*v_i[1]])
    A = np.array(A)
	# if you take solution 2:
	# A = np.zeros((2*N, 9))

    # b = np.zeros((2*N, 1))

    b = v.reshape(v.shape[0]*2,1)
    x = np.linalg.solve(A, b)
    x = np.concatenate((x, np.array([[1]])), axis = 0)

    # H = np.zeros((3, 3))
    H = x.reshape((3,3))
    # print(H)
    return H
    """
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

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    pixelNum = h*w
    orixy = np.zeros((pixelNum,2)).astype(int)
    d0 = np.arange(h)
    d1 = np.arange(w)
    orixy[:,0] = np.tile(d1, h)
    orixy[:,1] = np.repeat(d0, w)

    trax = orixy[:,0].reshape(1,pixelNum)
    tray = orixy[:,1].reshape(1,pixelNum)
    tra1 = np.ones((1,pixelNum))
    # print(trax.shape,tray.shape,tra1.shape)
    tra = np.concatenate((trax,tray,tra1), axis = 0)
    # print(tra)
    img4C = np.array([[0,0],[img.shape[1]-1,0],[0,img.shape[0]-1],[img.shape[1]-1,img.shape[0]-1]])
    m = solve_homography(img4C, corners)
    print(m)
    
    tra = np.dot(m,tra)
    # print(tra)
    tra = np.rint(tra/tra[2,:]).astype(int)
    
    canvas[tra[1,:],tra[0,:],:] = img[orixy[:,1],orixy[:,0],:]
    # TODO: some magic


def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/1.png')
    img2 = cv2.imread('./input/2.png')
    img3 = cv2.imread('./input/3.png')
    img4 = cv2.imread('./input/4.png')
    img5 = cv2.imread('./input/5.jpg')

    # canvas,_,_ = recoverImg(canvas)
    # cv2.imshow('ii', ii)
    # cv2.waitKey(0)
    corners1 = np.array([[820, 352], [880, 352], [817, 407], [883, 408]])
    transform(img1, canvas, corners1)
    # for xy in [[818, 352], [884, 352], [818, 407], [885, 408]]:
    #     break
    #     cv2.circle(canvas,(xy[0], xy[1]), 2, (255, 0, 255), -1)

    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    transform(img2, canvas, corners2)
    # for xy in [[311, 14], [402, 150], [157, 152], [278, 315]]:
    #     break
    #     cv2.circle(canvas,(xy[0], xy[1]), 10, (255, 255, 0), -1)

    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    transform(img3, canvas, corners3)
    # for xy in [[364, 674], [430, 725], [279, 864], [369, 885]]:
    #     break
    #     cv2.circle(canvas,(xy[0], xy[1]), 10, (0, 255, 255), -1)

    corners4 = np.array([[808, 494], [892, 494], [802, 609], [896, 609]])
    transform(img4, canvas, corners4)
    # for xy in [[808, 495], [892, 495], [802, 609], [896, 609]]:
    #     break
    #     cv2.circle(canvas,(xy[0], xy[1]), 13, (255, 255, 255), -1)

    corners5 = np.array([[1024, 608], [1119, 593], [1032, 664], [1135, 651]])
    transform(img5, canvas, corners5)
    # for xy in [[1024, 608], [1118, 593], [1032, 664], [1134, 651]]:
    #     break
    #     cv2.circle(canvas,(xy[0], xy[1]), 13, (200, 200, 200), -1)

    # TODO: some magic
    cv2.imwrite('part1.png', canvas)
    
    # Part 2
    img = cv2.imread('./input/screen.jpg')
    # TODO: some magic
    qrco = np.array([[1039, 368], [1101, 394], [982, 554], [1035, 600]])
    # for xy in [[1039, 368], [1101, 394], [982, 554], [1035, 600]]:
    #     # break
    #     cv2.circle(img,(xy[0], xy[1]), 1, (0, 255, 255), -1)

    transedcode = np.zeros((400,400,3))
    h, w, ch = transedcode.shape
    pixelNum = h*w
    orixy = np.zeros((pixelNum,2)).astype(int)
    d0 = np.arange(h)
    d1 = np.arange(w)
    orixy[:,0] = np.tile(d1, h)
    orixy[:,1] = np.repeat(d0, w)

    trax = orixy[:,0].reshape(1,pixelNum)
    tray = orixy[:,1].reshape(1,pixelNum)
    tra1 = np.ones((1,pixelNum))
    # print(trax.shape,tray.shape,tra1.shape)
    tra = np.concatenate((trax,tray,tra1), axis = 0)
    print(tra)
    img4C = np.array([[0,0],[h-1,0],[0,w-1],[h-1,w-1]])
    m = solve_homography(img4C, qrco)
    print(m)
    # print(np.dot(m, np.array([0,0,1]).T)) 
    # print(np.dot(m, np.array([900,900,1]).T)/9.02523680e-01) 
    tra = np.dot(m,tra)
    # print(tra)
    tra = tra/tra[2,:]
    for [ox, oy], [x, y] in zip(orixy[:,:2], tra[:2,:].T):
        transedcode[oy,ox,:] = img[np.rint(y).astype(int),np.rint(x).astype(int),:]
    # np.dot(m, np.array([i,j,1]).T)
    # transedcode = transedcode[:,:,0].astype(np.int16)
    
    def custom_blur_demo(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        dst = cv2.filter2D(image, -1, kernel=kernel)
        return dst
    transedcode = cv2.blur(transedcode, (5, 5))
    transedcode = cv2.resize(transedcode,(200,200),transedcode)
    transedcode = cv2.blur(transedcode, (3, 3))
    transedcode = custom_blur_demo(transedcode)
    transedcode = cv2.resize(transedcode,(400,400),transedcode)
    transedcode = custom_blur_demo(transedcode)

    transedcode = ((transedcode - np.min(transedcode))/(np.max(transedcode) - np.min(transedcode)))*255
    cv2.imwrite('part2.png', transedcode)
    
    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    bad_point = np.array([[135, 164], [587, 159], [62, 238], [660, 228]])
    street = np.array([[205,159], [587, 159], [154, 232], [660, 228]])
    # [135, 164], [62, 238]
    # for xy in street:
    #     cv2.circle(img_front,(xy[0], xy[1]), 1, (0, 255, 255), -1)
    lstreet = np.array([[135, 164], [174,160], [62, 238], [109, 235]])
    # for xy in lstreet:
    #     cv2.circle(img_front,(xy[0], xy[1]), 1, (0, 0, 255), -1)
    # cv2.imshow('d', img_front)
    # cv2.waitKey(0)
    # exit()
    # transedcode = np.zeros((600,300,3))
    # ltransedstreet = np.zeros((55,295,3))
    badh, badw = 700, 300
    h, w = 600, 300
    lh, lw = 55, 295

    badtra = np.zeros((750,1050,3))
    badimg4C = np.array([[0,0], [badh-1,0],[0,badw-1], [badh-1,badw-1]])
    badm = solve_homography(badimg4C, bad_point)
    

    badbiasi = 180 - 1 # lr
    badbiasj = 180 - 1 # ud
    for i in range(-badbiasi,badtra.shape[1] - 1 - badbiasi):
        for j in range(-badbiasj,badtra.shape[0] - 1 - badbiasj):
            x, y, c = np.dot(badm,np.array([i,j,1]))
            x, y = int(round(x/c)), int(round(y/c))
            if x >= 0 and y >= 0:
                try:
                    badtra[j+badbiasj,i+badbiasi,:] = img_front[y,x,:]
                except:
                    badtra[j+badbiasj,i+badbiasi,:] = [0,0,0]
                    pass

    newtra = np.zeros((750,1050,3))
    img4C = np.array([[0,0], [h-1,0],[0,w-1], [h-1,w-1]])
    m = solve_homography(img4C, street)
    limg4C = np.array([[0,0], [lh-1,0],[0,lw-1], [lh-1,lw-1]])
    lm = solve_homography(limg4C, lstreet)

    biasi = 280 - 1 # lr
    biasj = 180 - 1 # ud
    lbiasi = 280 - 1 # lr
    lbiasj = 180 - 1 # ud
    for i in range(-biasi,newtra.shape[1] - 1 - biasi):
        for j in range(-biasj,newtra.shape[0] - 1 - biasj):
            x, y, c = np.dot(m,np.array([i,j,1]))
            lx, ly, lc = np.dot(lm,np.array([i+100,j+6,1]))
            x, y = int(round(x/c)), int(round(y/c))
            lx, ly = int(round(lx/lc)), int(round(ly/lc))
            if x >= 0 and y >= 0:
                try:
                    if i >= -biasi + 280:
                        newtra[j+biasj,i+biasi,:] = img_front[y,x,:]
                    else:
                        newtra[j+lbiasj,i+lbiasi,:] = img_front[ly,lx,:]
                except:
                    newtra[j+biasj,i+biasi,:] = [0,0,0]
                    pass
    print(newtra.shape)

    # TODO: some magic
    # cv2.imshow('part3.png', badtra.astype(np.uint8))
    # cv2.waitKey(0)
    cv2.imwrite('part3.png', badtra)


if __name__ == '__main__':
    main()
