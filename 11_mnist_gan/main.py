from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = (x_test.astype(np.float32) - 127.5) / 127.5
mnist_data = x_test.reshape(10000, 28 * 28)
print(mnist_data.shape)
print(len(mnist_data))

