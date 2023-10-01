import matplotlib.pyplot as plt  # 그림으로 보기 위한 matplotlib 라이브러리 import
from tensorflow.keras.datasets import mnist  # 라이브러리가 기본으로 제공하는 mnist 데이터셋
from tensorflow.keras.utils import to_categorical  # one-hot encoding 을 위한 함수
from tensorflow.keras.models import Sequential  # 레이어를 층층히 쌓아가는 연쇄 모델
from tensorflow.keras.layers import Dense  # 완전연결층
from tensorflow.keras.models import load_model  # 저장된 모델 불러오기
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f"X_train_shape: {X_train.shape}")  # X_train_shape: (60000, 28, 28)
# print(f"y_train_shape: {y_train.shape}")  # y_train_shape: (60000,)
# print(f"X_test_shape: {X_test.shape}")  # X_test_shape: (10000, 28, 28)
# print(f"y_test_shape: {y_test.shape}")  # y_test_shape: (10000,)

# print(X_train)
# plt.imshow(X_train[0])
# plt.show()

print(type(x_test), type(x_train))

input_shape = x_train.shape[1] * x_train.shape[2]  # 그림의 크기: 28 * 28

number_of_classes = len(set(y_train))  # 레이블의 종류. 0~9로 10개

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, input_shape)  # 3차원 -> 2차원
x_test = x_test.reshape(-1, input_shape)

print(f"X_train_shape: {x_train.shape}")  # X_train_shape: (60000, 784)
print(f"X_test_shape: {x_test.shape}")  # X_test_shape: (10000, 784)

y_train = to_categorical(y_train, number_of_classes)  # 원-핫 인코딩. 1차원 -> 2차원
y_test = to_categorical(y_test, number_of_classes)

print(f"y_train_shape: {y_train.shape}")  # y_train_shape: (60000, 10)
print(f"y_test_shape: {y_test.shape}")  # y_test_shape: (10000, 10)

model = Sequential()  # 모델 선언

# 완전연결층 추가. 처음 쌓는 레이어는 input_shape: 데이터 차원(개수 제외)을 적어줘야함.
print(x_train.shape[1:])
model.add(Dense(128, activation="relu", input_shape=x_train.shape[1:]))

# 출력하는 완전연결층 추가. 다중분류이므로, softmax 활성화함수 사용
model.add(Dense(y_train.shape[1], activation="softmax"))

# 모델 컴파일. 다중분류이므로 categorical_crossentropy, 정확도 표기
model.compile(loss="categorical_crossentropy",
              optimizer="nadam",
              metrics=["acc"])

model.summary()  # 간단하게 요약해 출력

history = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)  # 학습 완료 후 검증
print("손실률:", loss)  # 손실률: 0.08662549406290054
print("정확도:", acc)  # 정확도: 0.9779999852180481

plt.figure(figsize=(18, 6))

# 에포크별 정확도
plt.subplot(1,2,1)
plt.plot(history.history["acc"], label="accuracy")
plt.plot(history.history["val_acc"], label="val_accuracy")
plt.title("accuracy")
plt.legend()

# 에포크별 손실률
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("loss")
plt.legend()

plt.show()

plt.imshow(x_test[0].reshape(28, 28))  # 데이터 일자로 펴주기
plt.show()

pred = model.predict(x_test[:1])[0]  # 다중분류이므로, predict_classes

print("real:", y_test[0].argmax())  # 7
# print("predict:", pred)  # 7

for i in pred:
    print(round(i, 5))