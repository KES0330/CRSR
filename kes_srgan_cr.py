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
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, AveragePooling2D, Cropping2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, Lambda, Subtract
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from kes_data_loader_cr import DataLoader
import numpy as np

from PIL import Image
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
        self.dataset_name = 'Linnaeus5'
        self.model_name = "210205_BiBiCycle_e3,1,1"
        self.n_residual_blocks = 8
        self.Epochs = 0
        self.weights = [1e-3, 1, 1e-1, 1, 1e-1]
        self.interpolation = Image.BILINEAR #compare
        self.up_interpolation = Image.BILINEAR #up

        self.interpolation_str = 'Nearest'
        if self.interpolation == Image.BILINEAR:
            self.interpolation_str = 'Bilinear'
        elif self.interpolation == Image.BICUBIC:
            self.interpolation_str = 'Bicubic'
        elif self.interpolation == Image.HAMMING:
            self.interpolation_str = 'Hamming'
        elif self.interpolation == Image.LANCZOS:
            self.interpolation_str = 'Lanczos'

        self.up_interpolation_str = 'Nearest'
        if self.up_interpolation == Image.BILINEAR:
            self.up_interpolation_str = 'Bilinear'
        elif self.up_interpolation == Image.BICUBIC:
            self.up_interpolation_str = 'Bicubic'
        elif self.up_interpolation == Image.HAMMING:
            self.up_interpolation_str= 'Hamming'
        elif self.up_interpolation == Image.LANCZOS:
            self.up_interpolation_str = 'Lanczos'

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])


        self.vgg2 = self.build_vgg2()
        self.vgg2.trainable = False
        self.vgg2.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Configure data loader
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width),
                                      low_img_res=(self.lr_height, self.lr_width),
                                      interpolation=self.interpolation,
                                      up_interpolation=self.up_interpolation)
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
        self.generator = self.build_generator() # build_generator
        if self.Epochs != 0:
            self.generator = load_model("./saved_model/{}/Generator_{}.h5".format(self.model_name, self.Epochs))

        # High res. and low res. images
        img = Input(shape=self.hr_shape)
        img_crop = Input(shape=self.lr_shape)
        img_crop_hr = Input(shape=self.hr_shape)

        # Generate high res. version from low res.
        fake_lr, fake_hr = self.generator(img)
        fake_lrlr, fake_hrhr = self.generator(fake_hr)

        tmp, lrhr = self.generator(img)
        #cr1, cr2, cr3, cr4 = self.Crop4(img_lr_hr)
        cr1 = Cropping2D(((0, self.lr_height), (0, self.lr_width)))(lrhr)
        cr2 = Cropping2D(((0, self.lr_height), (self.lr_width, 0)))(lrhr)
        cr3 = Cropping2D(((self.lr_height, 0), (0, self.lr_width)))(lrhr)
        cr4 = Cropping2D(((self.lr_height, 0), (self.lr_width, 0)))(lrhr)
        #img_lr_hr2 = self.Concatenate4(cr1, cr2, cr3, cr4)
        cr12 = Concatenate(axis=2)([cr1, cr2])
        cr34 = Concatenate(axis=2)([cr3, cr4])
        img_lr_hr2 = Concatenate(axis=1)([cr12, cr34])
        cr1_hr = Lambda(self.UpScale)(cr1)
        cr2_hr = Lambda(self.UpScale)(cr2)
        cr3_hr = Lambda(self.UpScale)(cr3)
        cr4_hr = Lambda(self.UpScale)(cr4)
        cr1_hr_lr, tmp1 = self.generator(cr1_hr)
        cr2_hr_lr, tmp2 = self.generator(cr2_hr)
        cr3_hr_lr, tmp3 = self.generator(cr3_hr)
        cr4_hr_lr, tmp4 = self.generator(cr4_hr)
        #lrhrhrlr = self.Concatenate4(cr1_hr_lr, cr2_hr_lr, cr3_hr_lr, cr4_hr_lr)
        cr12_hr_lr = Concatenate(axis=2)([cr1_hr_lr, cr2_hr_lr])
        cr34_hr_lr = Concatenate(axis=2)([cr3_hr_lr, cr4_hr_lr])
        lrhrhrlr = Concatenate(axis=1)([cr12_hr_lr, cr34_hr_lr])



        fake_lr2, fake_hr2 = self.generator(img_crop_hr)

        #crop1, crop2, crop3, crop4 = self.Crop4(img)
        crop1 = Cropping2D(((0, self.lr_height), (0, self.lr_width)))(img)
        crop2 = Cropping2D(((0, self.lr_height), (self.lr_width, 0)))(img)
        crop3 = Cropping2D(((self.lr_height, 0), (0, self.lr_width)))(img)
        crop4 = Cropping2D(((self.lr_height, 0), (self.lr_width, 0)))(img)
        crop1_hr = Lambda(self.UpScale)(crop1)
        crop2_hr = Lambda(self.UpScale)(crop2)
        crop3_hr = Lambda(self.UpScale)(crop3)
        crop4_hr = Lambda(self.UpScale)(crop4)
        crop1_hr_lr, tmp11 = self.generator(crop1_hr)
        crop2_hr_lr, tmp22 = self.generator(crop2_hr)
        crop3_hr_lr, tmp33 = self.generator(crop3_hr)
        crop4_hr_lr, tmp44 = self.generator(crop4_hr)
        #crop_ing = self.Concatenate4(crop1_hr_lr, crop2_hr_lr, crop3_hr_lr, crop4_hr_lr)
        crop12_hr_lr = Concatenate(axis=2)([crop1_hr_lr, crop2_hr_lr])
        crop34_hr_lr = Concatenate(axis=2)([crop3_hr_lr, crop4_hr_lr])
        hrlr = Concatenate(axis=1)([crop12_hr_lr, crop34_hr_lr])


        #crop_ing = Activation('tanh')(crop_ing)
        tm, hrlrlrhr = self.generator(hrlr)

        # Extract image features of the generated img
        fake_features = self.vgg(lrhr)
        fake_features_2 = self.vgg(lrhrhrlr)
        fake_features2 = self.vgg(hrlr)
        fake_features2_2 = self.vgg(hrlrlrhr)
        #fake_features = self.vgg(fake_hr)
        #fake_features2 = self.vgg2(fake_lr2)

        # For the combined model we will only train the generator
        #self.discriminator.trainable = False
        self.discriminator_frozen = Model(self.discriminator.input, self.discriminator.output)
        self.discriminator_frozen.trainable = False

        # Discriminator determines validity of generated high res. images
        #validity = self.discriminator_frozen(fake_hr)
        #validity = self.discriminator_frozen(fake_hr)
        validity = self.discriminator_frozen(lrhr)

        self.combined = Model([img, img_crop_hr], [validity, fake_features, fake_features_2, fake_features2, fake_features2_2])#fake_hr])
        self.combined.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse', 'mse'],
                              loss_weights=self.weights, #1e-3, 1, 1
                              optimizer=optimizer)

    def Crop4(self, input):
        width = (input.shape[2]//2)
        height = (input.shape[1]//2)
        crop1 = Cropping2D(((0, height), (0, width)))(input)
        crop2 = Cropping2D(((height, 0), (0, width)))(input)
        crop3 = Cropping2D(((0, height), (width, 0)))(input)
        crop4 = Cropping2D(((height, 0), (width, 0)))(input)
        return crop1, crop2, crop3, crop4

    def Concatenate4(self, crop1, crop2, crop3, crop4):
        cr1 = Concatenate(axis=2)([crop1, crop2])
        cr2 = Concatenate(axis=2)([crop3, crop4])
        cr = Concatenate(axis=1)([cr1, cr2])
        return cr

    def DownScale(self, input):
        width = (input.shape[1]//2)
        height = (input.shape[2]//2)
        return tf.image.resize(input, (width, height), method='bilinear')
    def UpScale(self, input):
        width = (input.shape[1]*2)
        height = (input.shape[2]*2)
        return tf.image.resize(input, (width, height), method='bilinear')

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=(self.hr_width, self.hr_height, 3))

        # Extract image features
        img_features = vgg(img)

        model = Model(inputs=[img], outputs=[img_features], name='vgg')
        return model

    def build_vgg2(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=(self.lr_width, self.lr_height, 3))

        # Extract image features
        img_features = vgg(img)

        model = Model(inputs=[img], outputs=[img_features], name='vgg2')
        return model

    def build_generator(self):
        def DownScale(input):
            width = (input.shape[1]//2)
            height = (input.shape[2]//2)
            return tf.image.resize(input, (width, height), method='bilinear')
        def UpScale(input):
            width = (input.shape[1]*2)
            height = (input.shape[2]*2)
            return tf.image.resize(input, (width, height), method='bilinear')

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = Add()([d, layer_input])
            return d

        def d_block(layer_input, filters, strides=1, bn=False):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = Activation('relu')(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        # Low resolution image input
        img_hr = Input(shape=self.hr_shape)#
        filter_hr = Conv2D(3, kernel_size=3, strides=1, padding='same')(img_hr)
        filter_hr = Conv2D(3, kernel_size=3, strides=1, padding='same')(filter_hr)
        filter_hr = Conv2D(3, kernel_size=3, strides=1, padding='same')(filter_hr)
        filter_hr = Activation('tanh')(filter_hr)

        img_lr = Lambda(DownScale)(filter_hr)
        #img_lr = AveragePooling2D(strides=4, pool_size=(4, 4), padding='same')(img_hr)
        # Pre-residual block
        c1 = Conv2D(self.gf, kernel_size=4, strides=2, padding='same')(img_hr)
        c1 = Activation('relu')(c1)
        c2 = c1

        # Propogate through residual blocks
        # r = residual_block(c2, self.gf)
        # for _ in range(self.n_residual_blocks - 1):
        #     r = residual_block(r, self.gf)
        for _ in range(self.n_residual_blocks):
            c2 = d_block(c2, self.gf, strides=1)
        # Post-residual block
        c3 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(c2)
        c3 = Activation('relu')(c3)

        # Generate high resolution output
        c3 = Conv2D(self.channels, kernel_size=3, strides=1, padding='same')(c3)#, activation='tanh')(c3)
        u1 = Add()([c3, img_lr])

        u1 = Activation('tanh')(u1)

        # cr1 = Cropping2D(((0, self.lr_height//2), (0, self.lr_width//2)))(u1)
        # cr2 = Cropping2D(((0, self.lr_height//2), (self.lr_width//2, 0)))(u1)
        # cr3 = Cropping2D(((self.lr_height//2, 0), (0, self.lr_width//2)))(u1)
        # cr4 = Cropping2D(((self.lr_height//2, 0), (self.lr_width//2, 0)))(u1)
        # crop1 = Concatenate(axis=2)([cr1, cr2])
        # crop2 = Concatenate(axis=2)([cr3, cr4])
        # cr = Concatenate(axis=1)([crop1, crop2])

        gen_lr = Lambda(UpScale)(u1)

        model = Model(inputs=img_hr, outputs=[u1, gen_lr])
        model.summary()
        return model

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

        model = Model(d0, validity)
        model.summary()
        return model

    def train(self, epochs, batch_size=1, sample_interval=50):
        print("train start")
        start_time = datetime.datetime.now()
        #minibatch = self.data_loader.get_batch(batch_size)

        t_epoch = []
        v_epoch = []
        t_D_Loss = []
        t_G_Loss = []
        t_D_Acc = []
        v_D_Acc = []
        v_D_Loss = []
        v_G_Loss = []
        D_Loss = 0.
        D_Acc = 0.
        G_Loss = 0.
        best_test_loss = 999
        for epoch in range(epochs):
            for batch_i, (imgs, imgs_crop, imgs_crop_hr) in enumerate(self.data_loader.load_batch(batch_size)):
                valid = np.ones((batch_size,) + self.disc_patch)
                fake = np.zeros((batch_size,) + self.disc_patch)
            #for batch_i in range(minibatch):

                # ----------------------
                #  Train Discriminator
                # ----------------------

                # From low res. image generate high res. version
                fake_lr, fake_hr = self.generator.predict(imgs)
                fake_lr2, fake_hr2 = self.generator.predict(imgs_crop_hr)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ------------------
                #  Train Generator
                # ------------------

                # Extract ground truth image features using pre-trained VGG19 model
                image_features = self.vgg.predict(imgs)
                image_features2 = self.vgg2.predict(imgs_crop)

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs, imgs_crop_hr], [valid, image_features, image_features, image_features, image_features])

                # Plot the progress
                elapsed_time = datetime.datetime.now() - start_time
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f] time: %s " \
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100 * d_loss[1], g_loss[0],
                       elapsed_time))

                g_loss_val = g_loss[0]
                if(g_loss_val < best_test_loss):
                    best_test_loss = g_loss_val
                    print("Best Model Epoch:%d, batch_i:%d, G loss: %f" % (epoch, batch_i, g_loss_val))
                    self.generator.save("best_%s_%4f.h5" % (self.model_name, best_test_loss))
                    self.test(epoch, batch_i, best_test_loss)

                if batch_i % sample_interval == 0 and batch_i != 0:
                    self.sample_images(epoch, batch_i)
                    self.sample_images_save(epoch, batch_i)

            test_size = batch_size
            valid = np.ones((test_size,) + self.disc_patch)
            fake = np.zeros((test_size,) + self.disc_patch)
            new_imgs, new_imgs_crop, new_imgs_crop_hr = self.data_loader.load_data(test_size)
            fake_lr, fake_hr = self.generator.predict(new_imgs)
            d_loss_real = self.discriminator.test_on_batch(new_imgs, valid)
            d_loss_fake = self.discriminator.test_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            new_image_features = self.vgg.predict(new_imgs)
            new_image_features2 = self.vgg2.predict(new_imgs_crop)
            g_loss = []
            g_loss = self.combined.test_on_batch([new_imgs, new_imgs_crop_hr],
                                                 [valid, new_image_features, new_image_features, new_image_features, new_image_features])

            t_epoch.append(epoch)  # + (batch_i / (6000 / batch_size)))
            t_D_Loss.append(d_loss[0] * 100)
            t_D_Acc.append(d_loss[1] * 100)
            t_G_Loss.append(g_loss[0])

            test_imgs, test_imgs_crop, test_imgs_crop_hr = self.data_loader.load_data(test_size, is_testing=False)
            test_fake_lr, test_fake_hr = self.generator.predict(test_imgs)
            test_d_loss_real = self.discriminator.test_on_batch(test_imgs, valid)
            test_d_loss_fake = self.discriminator.test_on_batch(test_fake_hr, fake)
            test_d_loss = 0.5 * np.add(test_d_loss_real, test_d_loss_fake)

            test_image_features = self.vgg.predict(test_imgs)
            test_image_features2 = self.vgg2.predict(test_imgs_crop)

            test_g_loss = self.combined.test_on_batch([test_imgs, test_imgs_crop_hr],
                                                      [valid, test_image_features, test_image_features, test_image_features, test_image_features])

            v_epoch.append(epoch)  # + (batch_i / (6000 / batch_size)))
            v_D_Loss.append(test_d_loss[0] * 100)
            v_D_Acc.append(test_d_loss[1] * 100)
            v_G_Loss.append(test_g_loss[0])
            plt.clf()
            plt.subplot(311)
            plt.plot(t_epoch, t_D_Loss, c='red', label="train_D_loss*100")  # marker='^',
            plt.plot(v_epoch, v_D_Loss, c='orange', label="test_D_loss*100")  # marker='v',
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('D_Loss')
            plt.subplot(312)
            plt.plot(t_epoch, t_D_Acc, c='red', label="train_D_Acc*100")  # marker='^',
            plt.plot(v_epoch, v_D_Acc, c='orange', label="test_D_Acc*100")  # marker='v',
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('D_Acc')
            plt.subplot(313)
            plt.plot(t_epoch, t_G_Loss, c='blue', label="train_G_loss")  # marker='^',
            plt.plot(v_epoch, v_G_Loss, c='cyan', label="test_G_loss")  # marker='v',
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('G_Loss')
            plt.savefig('images/%s/fig.png' % self.model_name)
            if epoch % 5 == 0 and epoch != 0:
                self.save_models(epoch)

    def save_models(self, epoch):
        os.makedirs('saved_model/%s' % self.model_name, exist_ok=True)
        self.discriminator.save("./saved_model/{}/Discriminator_{}.h5".format(self.model_name, epoch))
        self.generator.save("./saved_model/{}/Generator_{}.h5".format(self.model_name, epoch))

    def sample_images_save(self, epoch, batch_i, batch_size=1):
        os.makedirs('images/%s' % self.model_name, exist_ok=True)
        os.makedirs('images/%s/%d' % (self.model_name, epoch), exist_ok=True)

        imgs, imgs_crop, imgs_crop_hr = self.data_loader.load_data(batch_size=batch_size, is_testing=True)
        fakes_lr, fakes_hr = self.generator.predict_on_batch(imgs)
        for idx in range(batch_size):
            # Rescale images 0 - 1
            fakes_lr = 0.5 * fakes_lr + 0.5
            fakes_hr = 0.5 * fakes_hr + 0.5
            imgs = 0.5 * imgs + 0.5

            img_hr = np.asarray(self.np2img(imgs[idx]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            img_lr = np.asarray(self.np2img(imgs[idx]).resize((self.lr_height, self.lr_height), self.interpolation))
            img_lrhr = np.asarray(Image.fromarray(img_lr).resize((self.hr_height, self.hr_height), self.up_interpolation))
            fake_lr = np.asarray(self.np2img(fakes_lr[idx]).resize((self.lr_height, self.lr_height), self.up_interpolation))
            fake_hr = np.asarray(self.np2img(fakes_lr[idx]).resize((self.hr_height, self.hr_height), self.up_interpolation))

            compare1 = np.hstack((fake_lr, img_lr))
            compare1 = np.vstack((img_hr, compare1))

            compare2 = np.hstack((img_hr, fake_hr))
            compare2 = np.hstack((compare2, img_lrhr))

            Image.fromarray(compare1).save('images/%s/%d/%d_0compare1%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(compare2).save('images/%s/%d/%d_0compare2%d.png' % (self.model_name, epoch, batch_i, idx))


            Image.fromarray(img_hr).save('images/%s/%d/%d_1hr%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(fake_hr).save('images/%s/%d/%d_2gen_hr%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(img_lrhr).save('images/%s/%d/%d_3%s_lrup%d.png' % (self.model_name, epoch, batch_i, self.interpolation_str, idx))
            Image.fromarray(fake_lr).save('images/%s/%d/%d_4gen_lr%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(img_lr).save('images/%s/%d/%d_5%s_lr%d.png' % (self.model_name, epoch, batch_i, self.interpolation_str, idx))

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.model_name, exist_ok=True)
        os.makedirs('images/%s/%d' % (self.model_name, epoch), exist_ok=True)
        r, c = 2, 3

        imgs, imgs_crop, imgs_crop_hr = self.data_loader.load_data(batch_size=2, is_testing=True)

        fakes_lr, fakes_hr = self.generator.predict(imgs)

        # Rescale images 0 - 1
        fakes_lr = 0.5 * fakes_lr + 0.5
        fakes_hr = 0.5 * fakes_hr + 0.5
        imgs = 0.5 * imgs + 0.5
        imgs_lrhr = []

        for i in range(len(imgs)):
            img_lr = np.asarray(self.np2img(imgs[i]).resize((self.lr_height, self.lr_height), self.interpolation))
            img_lrhr = np.asarray(Image.fromarray(img_lr).resize((self.hr_height, self.hr_height), self.up_interpolation))
            imgs_lrhr.append(img_lrhr)

        # Rescale images 0 - 255
        fakes_lr = (fakes_lr * 255).astype(np.uint8)
        fakes_hr = (fakes_hr * 255).astype(np.uint8)
        imgs = (imgs * 255).astype(np.uint8)
        imgs_lrhr = np.array(imgs_lrhr)
        #imgs_lrhr = 0.5 * imgs_lrhr + 0.5
        #imgs_lrhr = (imgs_lrhr * 255).astype(np.uint8)

        # Save generated images and the high resolution originals
        titles = ['Original', 'Generated', self.interpolation_str]
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([imgs, fakes_hr, imgs_lrhr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1

        fig.savefig("images/%s/%d/%d.png" % (self.model_name, epoch, batch_i))
        plt.close()


    def test(self, epoch, batch_i, best_test_loss=0.):
        os.makedirs('images/%s/best/%d/%d_%4f' % (self.model_name, epoch, batch_i, best_test_loss), exist_ok=True)
        imgs, imgs_crop, imgs_crop_hr = self.data_loader.load_select_data(dataset='./test/*')
        len_hr = len(imgs)
        fakes_lr, fakes_hr = self.generator.predict_on_batch(imgs)
        fakes_hr = (0.5 * fakes_hr + 0.5)
        fakes_lr = (0.5 * fakes_lr + 0.5)
        imgs = (0.5 * imgs + 0.5)
        for i in range(len_hr):
            img_hr = np.asarray(self.np2img(imgs[i]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            img_lr = np.asarray(self.np2img(imgs[i]).resize((self.lr_height, self.lr_height), self.interpolation))
            img_lrhr = np.asarray(Image.fromarray(img_lr).resize((self.hr_height, self.hr_height), self.up_interpolation))
            fake_hr = np.asarray(self.np2img(fakes_lr[i]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            fake_lr = np.asarray(self.np2img(fakes_lr[i]).resize((self.lr_height, self.lr_height), self.up_interpolation))

            compare1 = np.hstack((fake_lr, img_lr))
            compare1 = np.vstack((img_hr, compare1))
            Image.fromarray(compare1).save('images/%s/best/%d/%d_%4f/%d_%d_%dcompare1.png' % (self.model_name, epoch, batch_i, best_test_loss, epoch, batch_i, i))
            compare2 = np.hstack((img_hr, fake_hr))
            compare2 = np.hstack((compare2, img_lrhr))
            Image.fromarray(compare2).save('images/%s/best/%d/%d_%4f/%d_%d_%dcompare2.png' % (self.model_name, epoch, batch_i, best_test_loss, epoch, batch_i, i))
        print("finish")

    def np2img(self, array):
            return Image.fromarray((array*255).astype(np.uint8))

if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=100, batch_size=2, sample_interval=50)
