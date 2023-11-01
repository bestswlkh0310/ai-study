from tensorflow import keras
import os
import numpy as np
import glob
from PIL import Image
current_path = os.getcwd()
model = keras.models.load_model(current_path + '/model')

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
        
    print("학습데이터(x_test)의 이미지 개수는",idx,"입니다.")
    return imgs, labels

image_dir_path = os.getcwd() + "/image"
(x_test, y_test)=load_data(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))

