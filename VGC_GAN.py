'''
 AUTHORS: (in no particular order)
        Eric Becerril-Blas    <Github: https://github.com/lordbecerril>
        Itzel Becerril        <Github: https://github.com/HadidBuilds>
        Erving Marure Sosa    <Github: https://eems20.github.io/>

 PURPOSE:
        We process the data and generate video game characters.
        Using a Deep Convolutional Generative Adversial Network.
'''
import pyfiglet
from pyfiglet import Figlet


from keras.optimizers import Adam, RMSprop

from shutil import copyfile
import os
from keras.preprocessing.image import load_img ,img_to_array
import matplotlib.pyplot as plt

import keras
from keras import layers
import os
import tensorflow as tf
from keras.layers import BatchNormalization
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.utils.vis_utils import plot_model
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rescale
import numpy as np
from tqdm import tqdm
import os
from keras.preprocessing import image


'''
    Model below
'''
latent_dim = 80 # 80 dimesnion of latent variables
height = 80
width = 80
channels = 3 # Every image has 3 channels: Red, Green, and Blue (RGB)
import keras
from keras import layers
import os
import tensorflow as tf
from keras.layers import BatchNormalization


def main():
    custom_fig = Figlet(font='starwars')
    print(custom_fig.renderText('Hello World From VGC_GAN.py!!'))

    print("\n")
    print("Let us build our generator")
    print("\n")
    '''
        Let us begin creating the GENERATOR!!! (https://www.youtube.com/watch?v=AJtedULP6fQ)
        Generator Architecture:
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_1 (InputLayer)         (None, 80)                0
        _________________________________________________________________
        dense_1 (Dense)              (None, 256000)            20736000
        _________________________________________________________________
        leaky_re_lu_1 (LeakyReLU)    (None, 256000)            0
        _________________________________________________________________
        reshape_1 (Reshape)          (None, 40, 40, 160)       0
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 40, 40, 320)       1280320
        _________________________________________________________________
        leaky_re_lu_2 (LeakyReLU)    (None, 40, 40, 320)       0
        _________________________________________________________________
        conv2d_transpose_1 (Conv2DTr (None, 80, 80, 320)       1638720
        _________________________________________________________________
        leaky_re_lu_3 (LeakyReLU)    (None, 80, 80, 320)       0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 80, 80, 320)       2560320
        _________________________________________________________________
        leaky_re_lu_4 (LeakyReLU)    (None, 80, 80, 320)       0
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 80, 80, 320)       2560320
        _________________________________________________________________
        leaky_re_lu_5 (LeakyReLU)    (None, 80, 80, 320)       0
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 80, 80, 3)         47043
        =================================================================
        Total params: 28,822,723
        Trainable params: 28,822,723
        Non-trainable params: 0
        _________________________________________________________________
    '''
    generator_input = keras.Input(shape=(latent_dim,))

    x = layers.Dense(160 * 40 * 40)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((40, 40, 160))(x)

    x = layers.Conv2D(320, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(320, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(320, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(320, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()
    """
    Now we build our Discriminator:
    Discriminator Architecture:
    Model: "model_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_2 (InputLayer)         (None, 80, 80, 3)         0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 78, 78, 160)       4480
    _________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)    (None, 78, 78, 160)       0
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 38, 38, 160)       409760
    _________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)    (None, 38, 38, 160)       0
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 18, 18, 160)       409760
    _________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)    (None, 18, 18, 160)       0
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 8, 8, 160)         409760
    _________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)    (None, 8, 8, 160)         0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 10240)             0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 10240)             0
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 10241
    =================================================================
    Total params: 1,244,001
    Trainable params: 1,244,001
    Non-trainable params: 0
    _________________________________________________________________
    """
    print("\n")
    print("Let us build our discriminator")
    print("\n")
    discriminator_input = layers.Input(shape = (height, width, channels))
    x = layers.Conv2D(160, 3)(discriminator_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(160, 4, strides = 2)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(160, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(160, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(1, activation = 'sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()
    print("\n")
    print("Let us build our actual GAN now")
    print("\n")
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
    print("list file is")
    print(list_file)


    data_train_gan = np.array([resize(imread(os.path.join('./Data/train_data/rgb_images',file_name.decode("utf-8"))), (80, 80,3)) for file_name in list_file])

    x_train = data_train_gan
    iterations = 10000
    batch_size = 40
    save_dir = './train_output/'


    start = 0
    dummy = 1
    for step in tqdm(range(iterations)):
      random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
      generated_images = generator.predict(random_latent_vectors)
      #print("generated images is")
      #print(generated_images)
      stop = start + batch_size
      real_images = x_train[start: stop]
      #print("reaml images is")
      #print(real_images)
      combined_images = np.concatenate([generated_images, real_images])
      labels = np.concatenate([np.ones((batch_size,1)),
                                        np.zeros((batch_size, 1))])
      labels += 0.05 * np.random.random(labels.shape)

      d_loss = discriminator.train_on_batch(combined_images, labels)

      random_latent_vectors = np.random.normal(size=(batch_size,
                                                     latent_dim))
      misleading_targets = np.zeros((batch_size, 1))
      a_loss = gan.train_on_batch(random_latent_vectors,
                                  misleading_targets)
      start += batch_size

      if start > len(x_train) - batch_size:
        start = 0

      if step % 10 == 0:
        print('discriminator loss:', d_loss)
        print('advesarial loss:', a_loss)
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(2,2)
        count = 0
        for i in range(2):
          for j in range(2):
            axes[i, j].imshow(resize(generated_images[count], (80,80)))
            axes[i, j].axis('off')
            count += 1
        plt.savefig(save_dir+str(dummy)+'.png')

      if step % 100 == 0:
        # Save the weight. If you want to train for a long time, make sure to save
        # in your drive. Mount your drive here. Google on how to mount your drive
        # into colab
        #gan.save_weights('model.h5')

        print('discriminator loss:', d_loss)
        print('advesarial loss:', a_loss)
    plt.show()

if __name__== "__main__":
    main()
