from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# tensorflow setting
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train_shape', x_train.shape)
print('y_train_shape', y_train.shape)
print('x_test_shape', x_test.shape)
print('y_test_shape', y_test.shape)

# reshape
x_train = x_train.reshape(60000, 28 * 28).astype('float32') / 255
x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255
print(x_train.shape)
print(x_test.shape)

# pre
pre_y_train = y_train
pre_y_test = y_test

# categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train.shape)
print(y_test.shape)

# making model
model = Sequential(name='sexy_text_classification_model')
model.add(Dense(512, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# learning model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)

# get score
score = model.evaluate(x_test, y_test)
print('score -', score[0])
print('accuracy -', score[1])

# visualize result
predicted_classes = np.argmax(model.predict(x_test), axis=1)
correct_indices = np.nonzero(predicted_classes == pre_y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != pre_y_test)[0]

# correct
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray')
    plt.title(f'predicted: {predicted_classes[correct]}, class: {pre_y_test[correct]}')

plt.tight_layout()
plt.show()

# incorrect
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray')
    plt.title(f'predicted: {predicted_classes[incorrect]}, class: {pre_y_test[incorrect]}')

plt.tight_layout()
plt.show()
