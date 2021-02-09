"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from kes_data_loader import DataLoader
import numpy as np
import os

import math
import scipy

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from keras.backend import tensorflow_backend as K
config = tf.compat.v1.ConfigProto() #tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=config))#tf.Session
K.set_learning_phase(1)

old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class SRGAN():
    def __init__(self):
        # Input shape
        self.scale_factor = 2 # 2x
        self.channels = 3
        self.hr_height = 256                 # High resolution height
        self.hr_width = 256                  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.lr_height = int(self.hr_height / self.scale_factor)   # Low resolution height
        self.lr_width = int(self.hr_width / self.scale_factor)     # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        ## PARAMETERS
        # Number of residual blocks in the generator
        self.model_name = "201231_DHGLe3,1"
        self.dataset_name = 'Linnaeus5'
        self.n_residual_blocks = 8
        self.Epochs = 0
        self.d_pair = "0" # 0 : hr, 1 : lr
        self.g_pair = "1" # 0 : hr, 1 : lr
        self.weights = [1e-3, 1]

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Configure data loader
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))
        self.total_sample = self.data_loader.get_total()

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        if self.Epochs != 0:
            self.discriminator = load_model("./saved_model/{}/Discriminator_{}.h5".format(self.model_name, self.Epochs))
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        if self.Epochs != 0:
            self.generator = load_model("./saved_model/{}/Generator_{}.h5".format(self.model_name, self.Epochs))

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.hr_shape) # not lr_shape

        # Generate high res. version from low res.
        fake_lr = self.generator(img_hr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_lr)

        # For the combined model we will only train the generator
        #self.discriminator.trainable = False
        self.discriminator_frozen = Model(self.discriminator.input, self.discriminator.output)
        self.discriminator_frozen.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator_frozen(fake_lr)

        self.combined = Model([img_hr, img_lr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=self.weights, #1e-3, 1
                              optimizer=optimizer)


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=(256, 256, 3))

        # Extract image features
        img_features = vgg(img)

        model = Model(inputs=[img], outputs=[img_features], name='vgg')
        return model
        #return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = Activation('relu')(d)
            #d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        # Low resolution image input
        img_hr = Input(shape=self.hr_shape)

        # Pre-residual block
        c1 = Conv2D(self.gf, kernel_size=9, strides=1, padding='same')(img_hr)
        c1 = Activation('relu')(c1)
        c2 = c1
        for i in range(int(math.log2(self.scale_factor))):
            c2 = d_block(c2, self.gf, strides=2)
            c2 = d_block(c2, self.gf, strides=1)

        # Propogate through residual blocks
        r = residual_block(c2, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c3 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(r)
        c3 = BatchNormalization(momentum=0.8)(c3)
        c3 = Add()([c3, c2])
        c3 = Conv2D(self.gf * self.scale_factor, kernel_size=3, strides=1, padding='same')(c3)

        # Generate high resolution output
        u1 = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(c3)

        # Upsampling
        gen_lr = UpSampling2D(size=self.scale_factor)(u1)

        return Model(img_hr, gen_lr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        print("train start")
        start_time = datetime.datetime.now()
        #minibatch = self.data_loader.get_batch(batch_size)

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        t_epoch = []
        v_epoch = []
        t_D_Loss = []
        t_G_Loss = []
        v_D_Loss = []
        v_G_Loss = []
        for epoch in range(epochs):
            for batch_i, (imgs_hr, imgs_lr) in enumerate(self.data_loader.load_batch(batch_size)):
            #for batch_i in range(minibatch):

                # ----------------------
                #  Train Discriminator
                # ----------------------

                # Sample images and their conditioning counterparts
                #imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

                # From low res. image generate high res. version
                fake_lr = self.generator.predict(imgs_hr)

                #valid = np.ones((batch_size,) + self.disc_patch)
                #fake = np.zeros((batch_size,) + self.disc_patch)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                if self.d_pair == 1:
                    d_loss_real = self.discriminator.train_on_batch(imgs_lr, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_lr, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ------------------
                #  Train Generator
                # ------------------

                # Sample images and their conditioning counterparts
                #imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

                # The generators want the discriminators to label the generated images as real
                #valid = np.ones((batch_size,) + self.disc_patch)

                # Extract ground truth image features using pre-trained VGG19 model
                image_features = self.vgg.predict(imgs_hr)
                if self.g_pair == 1:
                    image_features = self.vgg.predict(imgs_lr)

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_hr, imgs_lr], [valid, image_features])
                # Plot the progress
                elapsed_time = datetime.datetime.now() - start_time
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f] time: %s " \
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100 * d_loss[1], g_loss[0],
                       elapsed_time))

                if batch_i % sample_interval == 0 and batch_i != 0:
                    self.sample_images(epoch, batch_i)
                    self.sample_images_save(epoch, batch_i)

                if batch_i == (self.total_sample / batch_size) - 1:
                #if batch_i % int((6000 / batch_size) / 10) == 0 and not(batch_i == epoch == 0):
                    new_imgs_hr, new_imgs_lr = self.data_loader.load_data(batch_size)
                    fake_lr = self.generator.predict(new_imgs_hr)
                    d_loss_real = self.discriminator.test_on_batch(new_imgs_hr, valid)
                    if self.d_pair == 1:
                        d_loss_real = self.discriminator.test_on_batch(new_imgs_lr, valid)
                    d_loss_fake = self.discriminator.test_on_batch(fake_lr, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    new_image_features = self.vgg.predict(new_imgs_hr)
                    if self.g_pair == 1:
                        new_image_features = self.vgg.predict(new_imgs_lr)
                    g_loss = self.combined.test_on_batch([new_imgs_hr, new_imgs_lr], [valid, new_image_features])

                    t_epoch.append(epoch)# + (batch_i / (6000 / batch_size)))
                    t_D_Loss.append(d_loss[0] * 100)
                    t_G_Loss.append(g_loss[2])

                if batch_i == (self.total_sample / batch_size) - 1:
                #if batch_i % int((6000 / batch_size) / 4) == 0 and not(batch_i == epoch == 0):
                    test_imgs_hr, test_imgs_lr = self.data_loader.load_data(batch_size, is_testing=False)
                    test_fake_lr = self.generator.predict(test_imgs_hr)
                    test_d_loss_real = self.discriminator.test_on_batch(test_imgs_hr, valid)
                    if self.d_pair == 1:
                        test_d_loss_real = self.discriminator.test_on_batch(test_imgs_lr, valid)
                    test_d_loss_fake = self.discriminator.test_on_batch(test_fake_lr, fake)
                    test_d_loss = 0.5 * np.add(test_d_loss_real, test_d_loss_fake)
                    test_image_features = self.vgg.predict(test_imgs_hr)
                    if self.g_pair == 1:
                        test_image_features = self.vgg.predict(test_imgs_lr)
                    test_g_loss = self.combined.test_on_batch([test_imgs_hr, test_imgs_lr], [valid, test_image_features])

                    v_epoch.append(epoch)# + (batch_i / (6000 / batch_size)))
                    v_D_Loss.append(test_d_loss[0] * 100)
                    v_G_Loss.append(test_g_loss[2])

                if batch_i == (self.total_sample / batch_size) - 1:
                #if (batch_i % int((6000 / batch_size) / 10) == 0 and not(batch_i == epoch == 0)) or (batch_i % int((6000 / batch_size) / 4) == 0 and not(batch_i == epoch == 0)):
                    plt.clf()
                    plt.subplot(211)
                    plt.plot(t_epoch, t_D_Loss, c='red', label="train_D*100") #marker='^',
                    plt.plot(v_epoch, v_D_Loss, c='orange', label="test_D*100") #marker='v',
                    plt.legend(loc='upper right')
                    plt.grid()
                    plt.xlabel('epoch')
                    plt.ylabel('D_Loss')
                    plt.subplot(212)
                    plt.plot(t_epoch, t_G_Loss, c='blue', label="train_G") #marker='^',
                    plt.plot(v_epoch, v_G_Loss, c='cyan', label="test_G") #marker='v',
                    plt.legend(loc='upper right')
                    plt.grid()
                    plt.xlabel('epoch')
                    plt.ylabel('G_Loss')
                    plt.savefig('images/%s/fig.png' % self.model_name)

            if epoch % 10 == 0 and epoch != 0:
                self.save_models(epoch)

    def save_models(self, epoch):
        os.makedirs('saved_model/%s' % self.model_name, exist_ok=True)
        # os.makedirs('saved_model/%s/%d' % (self.model_name, epoch), exist_ok=True)
        self.discriminator.save("./saved_model/{}/Discriminator_{}.h5".format(self.model_name, epoch))
        self.generator.save("./saved_model/{}/Generator_{}.h5".format(self.model_name, epoch))

    def sample_images_save(self, epoch, batch_i, batch_size = 1):
        os.makedirs('images/%s' % self.model_name, exist_ok=True)
        os.makedirs('images/%s/%d' % (self.model_name, epoch), exist_ok=True)

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size, is_testing=True)
        fake_lr = self.generator.predict(imgs_lr)
        for idx in range(batch_size):

            a = np.array(imgs_hr[idx])
            b = np.array(fake_lr[idx])
            c = np.array(imgs_lr[idx])
            d = np.hstack((a, b, c))

            # Rescale images 0 - 1
            imgs_lr = 0.5 * imgs_lr + 0.5
            fake_hr = 0.5 * fake_lr + 0.5
            imgs_hr = 0.5 * imgs_hr + 0.5

            #scipy.misc.imsave("results/{}/epochs_{}/img_input_{}_{}.png".format(folder_name, epoch, epoch, index), input_images[index])
            scipy.misc.imsave('images/%s/%d/%d_compare%d.png' % (self.model_name, epoch, batch_i, idx), d)
            scipy.misc.imsave('images/%s/%d/%d_hr%d.png' % (self.model_name, epoch, batch_i, idx), imgs_hr[idx])
            scipy.misc.imsave('images/%s/%d/%d_nearest_lr%d.png' % (self.model_name, epoch, batch_i, idx), imgs_lr[idx])
            scipy.misc.imsave('images/%s/%d/%d_gen_lr%d.png' % (self.model_name, epoch, batch_i, idx), fake_lr[idx])


    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.model_name, exist_ok=True)
        os.makedirs('images/%s/%d' % (self.model_name, epoch), exist_ok=True)
        r, c = 2, 3

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_lr = self.generator.predict(imgs_hr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_lr = 0.5 * fake_lr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Original', 'Generated', 'Nearest']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([imgs_hr, fake_lr, imgs_lr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d/%d.png" % (self.model_name, epoch, batch_i))
        plt.close()

        # Save low resolution images for comparison
        # for i in range(r):
        #     fig = plt.figure()
        #     plt.imshow(imgs_lr[i])
        #     fig.savefig('images/%s/%d/%d_lowres%d.png' % (self.model_name, epoch, batch_i, i))
        #     plt.close()

if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=20, batch_size=8, sample_interval=50)
