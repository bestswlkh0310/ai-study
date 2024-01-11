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


def create_generator():
    generator = Sequential(name='create_text')
    generator.add(Dense(256, input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(784, activation='tanh'))
    return generator


def create_discriminator():
    discriminator = Sequential(name='text_discriminator')
    discriminator.add(Dense(512, input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return discriminator


def create_gan(discriminator: Sequential, generator: Sequential):
    discriminator.trainable = False
    gan_input = Input(shape=(100, ))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


generator = create_generator()
generator.summary()

discriminator = create_discriminator()
discriminator.summary()

gan = create_gan(discriminator, generator)
gan.summary()