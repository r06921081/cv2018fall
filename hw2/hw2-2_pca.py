import sys, os
import numpy as np
import cv2
from skimage import io, transform
dirname = sys.argv[1]
if dirname[-1] != '/':
    dirname = dirname + '/'
print(dirname)

# person = [None] * 10
peoples = {}
peoples_test = {}
knn_train_x = []
knn_train_y = []
for pn in os.listdir(dirname):
    if pn.split('.')[1] == 'png':
        
        # print(dirname + pn)
        personid, picid = int(pn.split('_')[0]), int((pn.split('_')[1]).split('.')[0])
        # print(personid, picid)
        img = cv2.imread(dirname + pn)[:,:,0].reshape(56*46)
        # print(img.shape)
        if picid - 1 >= 7:
            try:
                # peoples[personid] = [None] * 10
                peoples_test[personid][picid - 8] = img
            except:
                peoples_test[personid] = [None] * 3
                # print(picid - 7)
                peoples_test[personid][picid - 8] = img
        else:
            try:
                # peoples[personid] = [None] * 10
                peoples[personid][picid - 1] = img
                knn_train_x.append(img)
                knn_train_y.append(personid)
            except:
                peoples[personid] = [None] * 7
                peoples[personid][picid - 1] = img
                knn_train_x.append(img)
                knn_train_y.append(personid)

# for i in peoples[1]:
#     print(type(i), i.shape)
#     # for p in i:
#     #     pass
#     # print(type(p),p.shape)
#     cv2.imshow(str(i[0][0]), i)
#     cv2.waitKey(0)
allin = []
for i in range(len(peoples.items())):
    for pic in peoples[i+1]:
        allin.append(pic)
allin = np.array(allin)
print(allin.shape)

pic_num = len(allin)
# pics.append(transform.resize(pic,(size,size), mode='constant'))
flat_pics = allin
print(flat_pics.shape)
print(allin.shape)
# print(allin)
avgface = np.sum(allin, axis=0)/pic_num
show = avgface
# cv2.imwrite('avgface.png', show.reshape(56,46))

picMid = flat_pics - avgface # x - u
U, s, V = np.linalg.svd(picMid.T, full_matrices=False)
print(U.shape, V.shape)
print(np.max(U[:,0]), np.min(U[:,0]))
for i in range(5):
    show = 255 - ((U[:,i] - np.min(U[:,i]))/(np.max(U[:,i]) - np.min(U[:,i]))) * 255
    # cv2.imwrite('eigenface_' + str(i+1) + '.png', show.reshape(56,46))
# show = 255 - ((U[:,0] - np.min(U[:,0]))/(np.max(U[:,0]) - np.min(U[:,0]))) * 255
# # cv2.imshow('ef', show.reshape(56,46))
# # cv2.imwrite('sss.png', show.reshape(56,46))
# # print(np.abs(U[:,0].reshape(56,46))*255)
# # cv2.waitKey(0)
# # exit()
all_sigma = np.sum(s)

coordinate = np.dot(picMid, U) 

for i in range(1,281):
    continue
    first = 1*i
    U_f = U[:, :first]
    avgface = avgface
    # coordinate = coordinate[:, :first]
    # targetImg = transform.resize(io.imread(os.path.join(image_dir, recon_img)), (size, size), mode='constant')
    targetImg = img = cv2.imread(sys.argv[2])[:,:,0].reshape(56*46)

    # print(coordinate.shape)
    P = np.dot(U_f.T, targetImg - avgface)
    # x = np.dot(P, U.T) + avgface
    # plt.figure()
    # rcs = avgface + np.dot(coordinate[imgNo], U_f.T)
    rcs = avgface + np.dot(P, U_f.T)
    if i in [5, 50, 150, 280]:
        print(np.mean(np.square(targetImg-rcs)))
        # cv2.imwrite(str(i) + '_recon.png', rcs.reshape(56,46))
    rcs = (rcs - np.min(rcs))/(np.max(rcs)- np.min(rcs))
    # rcs /= (np.max(rcs)- np.min(rcs))
    rcs = (rcs * 255).astype(np.uint8)
    from time import sleep
    sleep(0.05)
    cv2.namedWindow("r",0)
    cv2.resizeWindow("r", 640, 480)
    cv2.imshow('r', rcs.reshape(56, 46))
    # cv2.resizeWindow("r", 56, 46)
    if cv2.waitKey(1) & 0xFF != 255 :
    #   if cv2.waitKey(1) & 0xFF != ord('q') :
        # video_service.unsubscribe(videoClient)
        break
    # print(i)
    # io.imsave('./reconstruction.png', rcs.reshape(56, 46))
# cv2.imwrite('sss.png', avgface)
# cv2.imshow('sss', avgface/255.)
# cv2.waitKey(0)

first = 280
U_f = U[:, :first]
avgface = avgface
targetImg = img = cv2.imread(sys.argv[2])[:,:,0].reshape(56*46)
P = np.dot(U_f.T, targetImg - avgface)
rcs = avgface + np.dot(P, U_f.T)
print(np.mean(np.square(targetImg-rcs)))
cv2.imwrite(sys.argv[3], rcs.reshape(56,46))

toTSNE = []
lable = []
for i in range(len(peoples_test.items())):
    for pic in peoples_test[i + 1]:    
        first = 100
        U_f = U[:, :first]
        avgface = avgface
        # coordinate = coordinate[:, :first]
        # targetImg = transform.resize(io.imread(os.path.join(image_dir, recon_img)), (size, size), mode='constant')
        targetImg = pic

        # print(coordinate.shape)
        # P = np.dot(U_f.T, targetImg - avgface)
        # x = np.dot(P, U.T) + avgface
        # plt.figure()
        # rcs = avgface + np.dot(coordinate[imgNo], U_f.T)
        # rcs = avgface + np.dot(P, U_f.T)
        toTSNE.append(targetImg)
        lable.append(i + 1)
        # print(np.mean(np.square(targetImg-rcs)))
        # cv2.imwrite(str(i) + 'test_recon.png', rcs.reshape(56,46))
        # rcs = (rcs - np.min(rcs))/(np.max(rcs)- np.min(rcs))
        # rcs /= (np.max(rcs)- np.min(rcs))
        # rcs = (rcs * 255).astype(np.uint8)
        # from time import sleep
        # sleep(0.05)
        # cv2.namedWindow("r",0)
        # cv2.resizeWindow("r", 640, 480)
        # cv2.imshow('r', rcs.reshape(56, 46))
        # cv2.resizeWindow("r", 56, 46)
        # if cv2.waitKey(1) & 0xFF != 255 :
        # #   if cv2.waitKey(1) & 0xFF != ord('q') :
        #     # video_service.unsubscribe(videoClient)
        #     break
    # print(i)
    # io.imsave('./reconstruction.png', rcs.reshape(56, 46))
first = 100
U_f = U[:, :first]
toKNN = np.array(toTSNE).copy()
toTSNE = np.array(toTSNE)
print(toTSNE.shape, U_f[:,:100].shape)
toTSNE = np.dot(toTSNE, U_f)
print(toTSNE.shape)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=30, n_iter=500000, perplexity=20)
dim2 = tsne.fit_transform(toTSNE)
ploted = [0] * 40
for x, y, l in zip(dim2[:,0], dim2[:,1], lable):
    color = l%8
    mark = (l//8)%5
    print(color, mark)
    mm = ["o", "^", "s", "P", "d"]
    colorA = ['b', 'g', 'r', 'c', 'gray', 'y', 'k', 'purple']
    ploted[l-1] += 1
    if ploted[l-1] == 3:
        plt.scatter(x, y, c=colorA[color], marker=mm[mark],  cmap='jet', label=l, s = 80)
    else:
        plt.scatter(x, y, c=colorA[color], marker=mm[mark],  cmap='jet', s = 80)
# plt.scatter(dim2[:,0], dim2[:,1], marker=lable,  cmap='jet', s = 4)#, cmap=plt.cm.get_cmap("jet", 10))
plt.legend(prop={'size':7})
# plt.show()

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 

#加载iris数据集
# iris = load_iris()
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
for k in [1, 3, 5]:
    for n in [3, 10, 39]:
        X_train = np.dot(np.array(knn_train_x), U[:, :n])
        y_train = np.array(knn_train_y)
        X_test = np.dot(toKNN, U[:, :n])
        y_test = np.array(lable)
        # X = np.concatenate((X_train, X_test), axis=0)
        # y = np.concatenate((y_train, y_test), axis=0)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        # exit()
        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, y_train)

        print()
        from sklearn.cross_validation import cross_val_score 

        scores = cross_val_score(knn, X_test, y_test, cv=3, scoring='accuracy')

        print(k, n, scores)
        # [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

        #将5次的预测准确平均率打印出
        print(scores.mean(), 'test score:', knn.score(X_test, y_test))
        print('----------------------------')
        # 0.973333333333
exit()
avgscore = {}
for k in [1, 3, 5]:
    for n in [3, 10, 39]:
        avgscore[str(k)+str(n)] = 0
        for valid_index in [[0,1,2,3,4], [0,1,2,5,6], [0,3,4,5,6]]:
            fold_train_x = []
            fold_train_y = []
            fold_valid_x = []
            fold_valid_y = []
            for x_index in range(len(knn_train_x)):
                if x_index%7 in valid_index:
                    fold_train_x.append(knn_train_x[x_index])
                    fold_train_y.append(knn_train_y[x_index])
                else:
                    fold_valid_x.append(knn_train_x[x_index])
                    fold_valid_y.append(knn_train_y[x_index])
            # X_train = np.dot(np.array(knn_train_x), U[:, :n])
            # y_train = np.array(knn_train_y)
            # X_test = np.dot(toKNN, U[:, :n])
            # y_test = np.array(lable)
            X_train = np.dot(np.array(fold_train_x), U[:, :n])
            y_train = np.array(fold_train_y)
            X_test = np.dot(np.array(fold_valid_x), U[:, :n])
            y_test = np.array(fold_valid_y)
            # X = np.concatenate((X_train, X_test), axis=0)
            # y = np.concatenate((y_train, y_test), axis=0)
            # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
            # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            # exit()
            knn = KNeighborsClassifier(n_neighbors=k)

            knn.fit(X_train, y_train)

            print(k, n, knn.score(X_test, y_test))
            avgscore[str(k)+str(n)] += knn.score(X_test, y_test)
            # from sklearn.cross_validation import cross_val_score 

            # scores = cross_val_score(knn, X_test, y_test, cv=3, scoring='accuracy')

            # print(k, n, scores)
            # [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

            #将5次的预测准确平均率打印出
            # print(scores.mean())
            # 0.973333333333
for k in [1, 3, 5]:
    for n in [3, 10, 39]:
        print(avgscore[str(k)+str(n)]/3)