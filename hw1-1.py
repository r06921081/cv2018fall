import cv2
import numpy as np
img = cv2.imread('testdata/1b.png')
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
print(b.shape)
# gray = np.array(0.299 * r + 0.587 * g + 0.114 * b, dtype=np.uint8)
weight = (0.4, 0.2, 0.4)
gray = np.array(weight[2] * r + weight[1] * g + weight[0] * b, dtype=np.uint8)

cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(gray)