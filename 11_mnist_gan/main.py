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
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def plot_generated_images(generator: Sequential):
    noise = np.random.normal(loc=0, scale=1, size=(100, 100))
    generated_images = generator.predict(noise).reshape(100, 28, 28)
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    # plt.tight_layout()


generator = create_generator()
generator.summary()

discriminator = create_discriminator()
discriminator.summary()

gan = create_gan(discriminator, generator)
gan.summary()

plot_generated_images(generator)

batch_size = 128
epochs = 5000

for e in tqdm(range(epochs)):
    noise = np.random.normal(0, 1, [batch_size, 100])
    generated_images = generator.predict(noise)
    image_batch = mnist_data[np.random.randint(low=0, high=mnist_data.shape[0], size=batch_size)]
    X = np.concatenate([image_batch, generated_images])
    y_dis = np.zeros(2 * batch_size)
    y_dis[:batch_size] = 1

    discriminator.trainable = True
    discriminator.train_on_batch(X, y_dis)

    noise = np.random.normal(0, 1, [batch_size, 100])
    y_gen = np.ones(batch_size)
    discriminator.trainable = False
    gan.train_on_batch(noise, y_gen)

    if e == 0 or e % 1000 == 0:
        plot_generated_images(generator)

plt.show()