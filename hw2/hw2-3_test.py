import numpy as np
import sys
import os
import cv2
from keras.models import load_model
import csv
from keras.utils import to_categorical
mnistdir = sys.argv[1]
if mnistdir[-1] != '/':
    mnistdir = mnistdir + '/'
# print(os.listdir(mnistdir))
x_test_org = []
x_name = []
namelist = os.listdir(mnistdir)
print(namelist)
namelist = sorted(namelist)
for pic_name in namelist:
    if pic_name.split('.')[-1] == 'png':
        pic = cv2.imread(mnistdir + pic_name)
        x_test_org.append(pic)
        x_name.append(pic_name.split('.')[0])
    # y_test_org.append(int(mnistdir[-2]))


x_test_org = np.array(x_test_org).reshape(-1,28,28,3)
# y_test_org = np.array(y_test_org).reshape(-1,1)
# print(x_test_org.shape, y_test_org.shape)
model = load_model('./CNN_model_e30_submit')
model.summary()
# prediction = model.predict(x_test)
result = model.predict(x_test_org)
# print(result)
# print(np.argsort(result)[:,-1])
# accuracy = model.evaluate(x_test_org, to_categorical(y_test_org, num_classes=10), verbose=0, batch_size=128)
# print('Testing accuracy: {}'.format(accuracy))
data2write = [['id', 'label']]
for pic_id, row in zip(x_name, result):
    # if row > 0.5:
    #     row = 1
    data2write.append([pic_id, int(np.argmax(result))])
text = open(sys.argv[2], 'w+')
s = csv.writer(text, delimiter=',', lineterminator='\n')
for i in data2write:
    s.writerow(i) 
text.close()