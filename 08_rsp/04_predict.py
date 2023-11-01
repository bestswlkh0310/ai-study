from tensorflow import keras
import os
import numpy as np
import glob
from PIL import Image
current_path = os.getcwd()
model = keras.models.load_model(current_path + '/model')

# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getcwd() + "/test"
images = glob.glob(image_dir_path + "/*.jpg")
# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(224, 224)
for img in images:
    print(img)
    try:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.LANCZOS)
        new_img.save(img,"PNG")
    except Exception as e:
        print(e)

img_size=224
color=4
file = glob.glob(current_path + '/test/test_image.png')[0]
alpha_channel = np.ones((224, 224, 1), dtype=np.int32)
data= np.array(Image.open(file),dtype=np.int32)

dd = np.concatenate((data, alpha_channel), axis=2)
# print()
d = np.expand_dims(dd, axis=0)
# print(load_data().shape)
result = model.predict(d)
print(result)