from PIL import Image
import os, glob
import os
# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getcwd() + "/image/s"
print("이미지 디렉토리 경로: ", image_dir_path)

images = glob.glob(image_dir_path + "/*.png")

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(224, 224)
for img in images:
    try:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.LANCZOS)
        new_img.save(img,"PNG")
    except:
        pass

print("가위 이미지 resize 완료!")

# 바위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getcwd() + "/image/r"
print("이미지 디렉토리 경로: ", image_dir_path)

images=glob.glob(image_dir_path + "/*.png")

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(224, 224)
for img in images:
    old_img=Image.open(img)
    new_img=old_img.resize(target_size,Image.LANCZOS)
    new_img.save(img,"PNG")

print("바위 이미지 resize 완료!")

# 보 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getcwd() + "/image/p"
print("이미지 디렉토리 경로: ", image_dir_path)

images=glob.glob(image_dir_path + "/*.png")

# 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
target_size=(224, 224)
for img in images:
    old_img=Image.open(img)
    new_img=old_img.resize(target_size,Image.LANCZOS)
    new_img.save(img,"PNG")

print("보 이미지 resize 완료!")
