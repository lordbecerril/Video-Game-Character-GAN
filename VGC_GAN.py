'''
 AUTHORS: (in no particular order)
        Eric Becerril-Blas    <Github: https://github.com/lordbecerril>
        Itzel Becerril        <Github: https://github.com/HadidBuilds>
        Erving Marure Sosa    <Github: https://eems20.github.io/>
        
 PURPOSE:
        We process the data and generate video game characters.
        Using a Deep Convolutional Generative Adversial Network.
'''
# Keras Libraries
import keras
from keras import layers
from keras.optimizers import Adam, RMSprop
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img ,img_to_array
from keras.preprocessing import image
# Tensorflow libraries
import tensorflow as tf
# skimage libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
# matplot lib
import matplotlib.pyplot as plt
# NumPy libraries
import numpy as np
# Libraries for file management
from shutil import copyfile
import os
# Progress bar library
from tqdm import tqdm

def main():
    # Global Variables below
    latent_dim = 32 # 32 dimesnion of latent variables... user for linear algebra operation
    height = 32 # 32 Pixel Height
    width = 32 # 32 Pixel Width
    channels = 3 # Every image has 3 channels: Red, Green, and Blue (RGB)
    # Generator
    print("\nLet us build our generator\n")
    '''
        Let us begin creating the GENERATOR!!! (https://www.youtube.com/watch?v=AJtedULP6fQ)
        Generator Architecture:
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_1 (InputLayer)         (None, 32)                0
        _________________________________________________________________
        dense_1 (Dense)              (None, 32768)             1081344
        _________________________________________________________________
        leaky_re_lu_1 (LeakyReLU)    (None, 32768)             0
        _________________________________________________________________
        reshape_1 (Reshape)          (None, 16, 16, 128)       0
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 16, 16, 256)       819456
        _________________________________________________________________
        leaky_re_lu_2 (LeakyReLU)    (None, 16, 16, 256)       0
        _________________________________________________________________
        conv2d_transpose_1 (Conv2DTr (None, 32, 32, 256)       1048832
        _________________________________________________________________
        leaky_re_lu_3 (LeakyReLU)    (None, 32, 32, 256)       0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 32, 32, 256)       1638656
        _________________________________________________________________
        leaky_re_lu_4 (LeakyReLU)    (None, 32, 32, 256)       0
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 32, 32, 256)       1638656
        _________________________________________________________________
        leaky_re_lu_5 (LeakyReLU)    (None, 32, 32, 256)       0
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 32, 32, 3)         37635
        =================================================================
        Total params: 6,264,579
        Trainable params: 6,264,579
        Non-trainable params: 0
        _________________________________________________________________
    '''
    generator_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()
    """
        Now we build our Discriminator:
        Discriminator Architecture:
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_2 (InputLayer)         (None, 32, 32, 3)         0
        _________________________________________________________________
        conv2d_5 (Conv2D)            (None, 30, 30, 128)       3584
        _________________________________________________________________
        leaky_re_lu_6 (LeakyReLU)    (None, 30, 30, 128)       0
        _________________________________________________________________
        conv2d_6 (Conv2D)            (None, 14, 14, 128)       262272
        _________________________________________________________________
        leaky_re_lu_7 (LeakyReLU)    (None, 14, 14, 128)       0
        _________________________________________________________________
        conv2d_7 (Conv2D)            (None, 6, 6, 128)         262272
        _________________________________________________________________
        leaky_re_lu_8 (LeakyReLU)    (None, 6, 6, 128)         0
        _________________________________________________________________
        conv2d_8 (Conv2D)            (None, 2, 2, 128)         262272
        _________________________________________________________________
        leaky_re_lu_9 (LeakyReLU)    (None, 2, 2, 128)         0
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 512)               0
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 512)               0
        _________________________________________________________________
        dense_2 (Dense)              (None, 1)                 513
        =================================================================
        Total params: 790,913
        Trainable params: 790,913
        Non-trainable params: 0
        _________________________________________________________________
    """
    print("\n")
    print("Let us build our discriminator")
    print("\n")
    discriminator_input = layers.Input(shape = (height, width, channels))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)
    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()
    print("\n")
    print("Let us build our actual GAN now")
    print("\n")
    '''
        To make the backpropagation possible for the Generator, we create new
        network in Keras, which is Generator followed by Discriminator. In this
        network, we freeze all the weights so that its weight do not changes.
        GAN Architecture:
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_3 (InputLayer)         (None, 32)                0
        _________________________________________________________________
        model_1 (Model)              (None, 32, 32, 3)         6264579
        _________________________________________________________________
        model_2 (Model)              (None, 1)                 790913
        =================================================================
        Total params: 7,055,492
        Trainable params: 6,264,579
        Non-trainable params: 790,913
        _________________________________________________________________
    '''
    discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
    gan.summary()
    print("\n")
    print("Train Model")
    print("\n")
    list_file = os.listdir(os.fsencode("./Data/train_data/rgb_images"))
    data_train_gan = np.array([resize(imread(os.path.join('./Data/train_data/rgb_images',file_name.decode("utf-8"))), (32, 32,3)) for file_name in list_file])
    x_train = data_train_gan
    # We will do 10000 iterations. Every iteration we process 128 batches
    iterations = 10000
    batch_size = 128
    save_dir = './train_output/'
    start = 0
    # Iterate and make a cool progress bar
    for step in tqdm(range(iterations)):
        # Generate random noise in a probability distribution such as normal distribution.
        random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
        generated_images = generator.predict(random_latent_vectors)
        # Combine the generated fake data with the data that is sampled from the dataset
        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size, 1))])
        # Add noise to the label of the input
        labels += 0.05 * np.random.random(labels.shape)
        # Train the Discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)
        # Train the Generator
        random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        # Update the start index of the real dataset
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        # Print the loss and also save the faces generated by generator into train_output
        if step % 10 == 0:
            print('discriminator loss:', d_loss)
            print('advesarial loss:', a_loss)
            fig, axes = plt.subplots(2, 2)
            fig.set_size_inches(2,2)
            count = 0
            for i in range(2):
              for j in range(2):
                  axes[i, j].imshow(resize(generated_images[count], (32,32)))
                  axes[i, j].axis('off')
                  count += 1
            plt.savefig(save_dir+str(step)+'.png') # Every step % 10 iteration save the images
        # We save weight every 100 steps
        if step % 100 == 0:
            gan.save_weights('model.h5') # pretty large file, comment out if running locally
    print("Finished")
if __name__== "__main__":
    main()
