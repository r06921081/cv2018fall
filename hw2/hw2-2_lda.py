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
        img = cv2.imread(dirname + pn)[:,:,0].reshape(56*46)/255
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

picMid = flat_pics - avgface # x - u
U, s, V = np.linalg.svd(picMid.T, full_matrices=False)
print(U.shape, V.shape)

first = 240
U_f = U[:, :first]
to240 = []

toTSNE = []
lable = []

proj = np.dot(flat_pics, U_f)

mu = np.mean(proj, axis=0)
# to240 = to240 - mu
# lda = np.dot(to240,U_f)
# print(lda.shape)
# print(mu.shape)

mu_class = [None]*40
# print(to240.shape)
for c in range(40):
    mu_class[c] = np.mean(proj[c*7:c*7+7], axis = 0).reshape(-1, 1)
    
new_mu = np.mean(mu_class, axis = 0)
# print(np.sum(new_mu - mu))
# exit()
sb = np.zeros((240, 240), dtype=np.float64)
print(new_mu.shape)
for u in mu_class:
    tmp = (u - mu.reshape(-1, 1))
    sb += np.dot(tmp, tmp.T)
print(sb.shape)

sw = np.zeros((240, 240), dtype=np.float64)
for c in range(40):
    for j in np.array(proj[c*7:c*7+7]):
    # for j in range(7):
        # print(proj[c*7:c*7+7].shape)
        tmp = (mu_class[c] - j.reshape(240, -1))
        sw += np.dot(tmp, tmp.T)
# print(sw.shape)
# print(np.sum(sw-s), 'ffff')
sw_ = np.linalg.inv(sw)
fisher = np.dot(sw_, sb)
evalue, evec = np.linalg.eig(fisher)
evalue = np.real(evalue)
evec = np.real(evec)

idx = evalue.argsort()[::-1]
evalue = evalue[idx]
evec = evec[:,idx]

print(fisher.shape)
# print(evalue)
print(evalue.shape, evec.shape)
# exit()
#fisherface = np.dot(U_f, f_v[:39].T)
print(evec[:39,:].shape)
# exit()
fisherface = np.dot(U_f, evec[:,:39])
print(fisherface.shape)
for i in range(5):
    ff = fisherface[:,i].reshape(56,46)
    ff = (ff - np.min(ff))/(np.max(ff)- np.min(ff))
    # rcs /= (np.max(rcs)- np.min(rcs))

    ff = (ff * 255).astype(np.uint8)
    # cv2.imshow('ssssf', ff)
    if i == 0:
        cv2.imwrite(sys.argv[2], ff)
    # cv2.imwrite(str(i) + 'fisherface.png', ff)
    # cv2.resizeWindow("ssssf", 640, 480)
cv2.waitKey(0)

toTSNE = []
tsnelabel = []
lable = []
for i in range(len(peoples_test.items())):
    for j in peoples_test[i + 1]:
        toTSNE.append(j)
        tsnelabel.append(i + 1)
        lable.append(i + 1)
toKNN = np.array(toTSNE).copy()
toTSNE = np.array(toTSNE)
tsnelabel = np.array(tsnelabel)
print(toTSNE.shape, tsnelabel.shape)

toTSNE = np.dot(toTSNE, fisherface[:,:30])
# print(toTSNE.shape)
# exit()
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=30, n_iter=500000, perplexity=20)
dim2 = tsne.fit_transform(toTSNE)
ploted = [0] * 40
for x, y, l in zip(dim2[:,0], dim2[:,1], tsnelabel):
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
plt.show()

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
for k in [1, 3, 5]:
    for n in [3, 10, 39]:
        X_train = np.dot(np.array(knn_train_x), fisherface[:, :n])
        y_train = np.array(knn_train_y)
        X_test = np.dot(toKNN, fisherface[:, :n])
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