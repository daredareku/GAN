import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize the images
x_train = x_train / 127.5 - 1.
x_train = np.expand_dims(x_train, axis=3)

# Set the dimensions of the noise vector
noise_dim = 100

# Generator model
generator = Sequential()
generator.add(Dense(128 * 7 * 7, input_dim=noise_dim))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Reshape((7, 7, 128)))
generator.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
generator.add(LeakyReLU(alpha=0.2))
generator.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
generator.add(LeakyReLU(alpha=0.2))
generator.add(tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same'))

# Discriminator model
discriminator = Sequential()
discriminator.add(tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(28,28,1)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(tf.keras.layers.Flatten())
discriminator.add(Dense(1, activation='sigmoid'))

# Combined model
gan = Sequential([generator, discriminator])

# Set the optimizer and compile the models
adam = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=adam)

# Train the GAN
epochs = 100
batch_size = 128
steps_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Train the discriminator
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_images = generator.predict(noise)
        x = np.concatenate((real_images, fake_images))
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x, y)

        # Train the generator
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        y = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y)

    # Print the losses
    print(f"Epoch {epoch+1}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

    # Save generated images every 10 epochs
    if epoch % 10 == 0:
        noise = np.random.normal(0, 1, size=(25, noise_dim))
        generated_images = generator.predict(noise)
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        plt.show()
