import os
import numpy as np
import glob
from PIL import Image
current_path = os.getcwd()
def load_data(img_path):
    # 가위 : 0, 바위 : 1, 보 : 2
    r_path = current_path + '/image/r'
    p_path = current_path + '/image/p'
    s_path = current_path + '/image/s'

    def getFileCount(forder_path):
        return len(os.listdir(forder_path))

    number_of_data = getFileCount(r_path) + getFileCount(p_path) + getFileCount(s_path)

    img_size=224
    color=4
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/s/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 0   # 가위 : 0
        idx += 1

    for file in glob.iglob(img_path+'/r/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 1   # 바위 : 1
        idx += 1
    
    for file in glob.iglob(img_path+'/p/*.png'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 2   # 보 : 2
        idx += 1
        
    print("학습데이터(x_train)의 이미지 개수는",idx,"입니다.")
    return imgs, labels

image_dir_path = os.getcwd() + "/image"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# learning

import tensorflow as tf
from tensorflow import keras
import numpy as np

n_channel_1=32
n_channel_2=64
n_dense=128
n_train_epoch=5

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(224,224,4)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.summary()
model.save(current_path + '/model')


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print('라벨: ', y_train[0])


index=50
plt.imshow(x_train[index],cmap=plt.cm.binary)
plt.show()
print((index+1), '번째 이미지는 바로 ',  y_train[index], '입니다.')