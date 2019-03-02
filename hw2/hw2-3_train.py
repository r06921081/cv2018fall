import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

mnistdir = sys.argv[1]
if mnistdir[-1] != '/':
    mnistdir = mnistdir + '/'
print(os.listdir(mnistdir))
os.listdir(mnistdir)
tv = os.listdir(mnistdir)
kindlist = os.listdir(mnistdir+tv[0])
x_train_org = []
y_train_org = []
x_test_org = []
y_test_org = []
for md in tv:
    if md != 'valid':
        for dn in kindlist:
            for pn in os.listdir(mnistdir+md+'/'+dn):
                if (pn.split('.')[1] == 'png'):
                    pic = cv2.imread(mnistdir+md+'/'+dn+'/'+pn)
                    x_train_org.append(pic)
                    y_train_org.append(int(dn.split('_')[1]))
    else:
        for dn in kindlist:
            for pn in os.listdir(mnistdir+md+'/'+dn):
                if (pn.split('.')[1] == 'png'):
                    pic = cv2.imread(mnistdir+md+'/'+dn+'/'+pn)
                    x_test_org.append(pic)            
                    y_test_org.append(int(dn.split('_')[1]))

x_train_org = np.array(x_train_org).reshape(-1,28,28,3)
x_test_org = np.array(x_test_org).reshape(-1,28,28,3)
y_train_org = np.array(y_train_org).reshape(-1,1)
y_test_org = np.array(y_test_org).reshape(-1,1)
# print(len(x_train_org), len(x_test_org))
# exit()
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, MaxPool2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical


num_classes = 10


# (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
print('orginal training images shape: {}\norginal training labels shape: {}\n'.format(x_train_org.shape, y_train_org.shape))
print('testing images shape: {}\ntesting labels shape: {}'.format(x_test_org.shape, y_test_org.shape))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train_org[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train_org[i]))

x_train_org = x_train_org / 255.
x_test_org = x_test_org / 255.

x_train = x_train_org
y_train = to_categorical(y_train_org, num_classes)
x_valid = x_test_org
y_valid = to_categorical(y_test_org, num_classes)

x_test = x_test_org
y_test = to_categorical(y_test_org)

print('x_train shape: {} // y_train.shape: {}'.format(x_train.shape, y_train.shape))
print('x_valid shape: {} // y_valid.shape: {}'.format(x_valid.shape, y_valid.shape))

def myCNN():
    img_input = Input(shape=(28, 28, 3))
    co1 = Conv2D(64, (7, 7), padding='valid', activation='relu', name='co1')(img_input)
    # bo1 = BatchNormalization()(co1)
    # mo1 = AveragePooling2D((2,2))(bo1)
    # do1 = Dropout(0.2, name='do1')(bo1)
    co2 = Conv2D(64, (5, 5), strides=2, padding='valid', activation='relu', name='co2')(co1)
    # bo2 = BatchNormalization()(co2)
    # mo2 = AveragePooling2D((2,2))(bo2)
    # do2 = Dropout(0.3, name='do2')(bo2)
    co3 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='co3')(co2)
    # bo3 = BatchNormalization()(co3)
    # mo3 = AveragePooling2D((2,2))(bo3)
    # do3 = Dropout(0.4, name='do3')(mo3)
    flat = Flatten()(co3)
    fc1 = Dense(64, activation='relu', name='de1')(flat)
    # fb1 = BatchNormalization()(fc1)
    # dof1 = Dropout(0.4, name='dof1')(fb1)
    fc2 = Dense(64, activation='relu', name='de2')(fc1)
    # fb2 = BatchNormalization()(fc2)
    # dof2 = Dropout(0.5, name='dof2')(fb2)
    fc3 = Dense(num_classes, activation='softmax', name='de3')(fc1)
    # model = Sequential()
    # model.add(img_input)
    # model.add(Conv2D(32, kernel_size=(15, 15), activation='relu', 
    #     name='co1', padding='valid', kernel_initializer='glorot_normal'))
    # model.add(Conv2D(64, kernel_size=(9, 9), activation='relu', 
    #     name='co2', padding='valid', kernel_initializer='glorot_normal'))
    # model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', 
    #     name='co3', padding='valid', kernel_initializer='glorot_normal'))

    # model.add(Flatten())

    # model.add(Dense(120, activation='relu', name='de1'))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.5))
    # model.add(Dense(84, activation='relu', name='de2'))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax', name='de3', kernel_initializer='glorot_normal'))

    model = Model(img_input, fc3)
    L_model = 1#Model(img_input, co1)
    H_model = 1#Model(img_input, co3)
    
    return model, L_model, H_model

model, L_model, H_model = myCNN()
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

ckpt = ModelCheckpoint('CNN_model_e{epoch:02d}', # CNN_model_e{epoch:02d}_a{val_acc:.4f}
                       monitor='val_acc',
                       save_best_only=False,
                       save_weights_only=False,
                       verbose=1)
cb = [ckpt]

epochs = 30
batch_size = 256

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_valid, y_valid),
                    callbacks=cb,
                    verbose=1)


l = history.history['loss']
vl = history.history['val_loss']
acc = history.history['acc']
vacc = history.history['val_acc']

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(np.arange(epochs)+1, l, 'b', label='train loss')
plt.plot(np.arange(epochs)+1, vl, 'r', label='valid loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss curve")
plt.legend(loc='best')

plt.subplot(122)
plt.plot(np.arange(epochs)+1, acc, 'b', label='train accuracy')
plt.plot(np.arange(epochs)+1, vacc, 'r', label='valid accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy curve")
plt.legend(loc='best')
plt.tight_layout()

# plt.show()

model_name = 'CNN_model_e' + str(epochs).zfill(2)
model.load_weights(model_name)

# prediction = model.predict(x_test)
accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Testing accuracy: {}'.format(accuracy))