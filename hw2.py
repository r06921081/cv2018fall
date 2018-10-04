import cv2
import numpy as np

def h_s(i, j, sigma_s):
    return np.exp(-(i**2 + j**2)/ (2 * sigma_s ** 2))

def h_r(i, j, sigma_r):
    return np.exp(-(i**2 + j**2)/ (2 * sigma_r ** 2))

def gaussian(self, x, sigma=1500, u=0):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y

def h(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2 * sigma ** 2))

def w(r, sigma):
    result = 0
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            h(i, j, sigma)


def jfioio(x, y, sigma_s, sigma_r, r):
    sum = 0
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            h_s = np.exp(-(i**2 + j**2)/ (2 * sigma_s ** 2))
            h_r = np.exp(-(_f(x - i, y - j) - _f(x, y)) **2 / (2 * sigma_r ** 2))
            sum += h_s * h_r * f(x, y)
    result = sum / w(r, sigma)

img = cv2.imread('testdata/1a.png')
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
# cv2.imshow('R', r)
# cv2.imshow('G', g)
# cv2.imshow('B', b)
gray = np.array(0.299 * r + 0.587 * g + 0.114 * b, dtype=np.uint8)
x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
absX = cv2.convertScaleAbs(x)   # 转回uint8
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
print(absX)

# cv2.imshow("Result", dst)
sym = cv2.copyMakeBorder(dst,10,10,10,10,cv2.BORDER_REFLECT_101)
cv2.imshow("101", sym)

img = cv2.imread("testdata/1a.png", 0)

ox = cv2.Sobel(img,cv2.CV_16S,1,0)
oy = cv2.Sobel(img,cv2.CV_16S,0,1)

oabsX = cv2.convertScaleAbs(ox)
oabsY = cv2.convertScaleAbs(oy)

ol = cv2.addWeighted(oabsX,0.5,oabsY,0.5,0)

# cv2.imshow("rgbline", ol)


# cv2.imshow('Gray', gray)
for sigma_s in [1, 2, 3]:
    for sigma_r in [0.05, 0.1, 0.2]:
        print(sigma_s, sigma_r)


kernel_size = 3

img = cv2.imread('testdata/1a.png')
img = gray
print(type(img))
img1 = img#np.array(img, dtype=np.float32) #np.array(img, dtype=cv2.CV_16S) #转化数值类型
print(type(img1))
# exit()
# kernel = np.ones((kernel_size, kernel_size), np.float32)/kernel_size**2
kernel = np.array([[-1,0,1],
[-2,0,2],
[-1,0,1]])
print(type(kernel))

# print(gray)
dst = cv2.filter2D(gray, -1, kernel)#, dtype=np.uint8)
print(dst)
# exit()
#cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
#输出图像与输入图像大小相同


cv2.imshow('Gray', img1)

cv2.imshow('gray', dst)
print('fffffff')
print(dst)
print(np.sum(dst - dst))
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(dst)