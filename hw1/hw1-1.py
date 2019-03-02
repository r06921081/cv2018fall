import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
    print('try with: "python hw1-1.py <input/path> <output/path>"')
    exit()
img = cv2.imread(sys.argv[1].replace('//', '/'))
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
# print(b.shape)
# gray = np.array(0.299 * r + 0.587 * g + 0.114 * b, dtype=np.uint8)
weight = (0.114, 0.587, 0.299)
gray = np.array(weight[2] * r + weight[1] * g + weight[0] * b, dtype=np.uint8)


cv2.imwrite(sys.argv[2].replace('//', '/'), gray)
cv2.imshow('Gray', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(gray)