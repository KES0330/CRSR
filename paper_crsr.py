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
from kes_data_loader_cycle import DataLoader
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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
#출처: https://mellowlee.tistory.com/entry/Python-Keras-InternalError-GPU-sync-failed [잠토의 잠망경]
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.framework.config import set_memory_growth
tf.compat.v1.disable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
        self.dataset_name = 'DIV2K'
        self.model_name = "210209_CRSR_DIV2K_Paper_VGG_0.7,1"
        self.n_residual_blocks = 7
        self.Epochs = 0
        self.weights = [0.7, 1]
        self.interpolation = Image.BICUBIC #compare
        self.up_interpolation = Image.BICUBIC #up

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
                                      interpolation=self.interpolation)
        #self.total_sample = self.data_loader.get_total()

        # Calculate output shape of D (PatchGAN)
        # patch = int(self.hr_height / 2**4)
        # self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        #self.df = 64

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        # if self.Epochs != 0:
        #     self.discriminator = load_model("./saved_model/{}/Discriminator_{}.h5".format(self.model_name, self.Epochs))
        # self.discriminator.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])
        #
        # self.discriminator2 = self.build_discriminator()
        # self.discriminator2.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])

        # Build the generator
        self.SRG = self.build_SRG() # build_generator
        if self.Epochs != 0:
            self.generator = load_model("./saved_model/{}/SRG_{}.h5".format(self.model_name, self.Epochs))

        self.CRG = self.build_CRG()  # build_generator
        if self.Epochs != 0:
            self.generator = load_model("./saved_model/{}/CRG_{}.h5".format(self.model_name, self.Epochs))


        img = Input(shape=self.hr_shape)

        CR = self.CRG(img)
        CRSR = self.SRG(CR)
        CRUP = Lambda(self.UpScale)(CR)
        # CRSRCR = self.CRG(CRSR)
        # CRSRCRSR = self.SRG(CRSRCR)

        # crop1 = Cropping2D(((0, self.lr_height), (0, self.lr_width)))(img)
        # crop2 = Cropping2D(((0, self.lr_height), (self.lr_width, 0)))(img)
        # crop3 = Cropping2D(((self.lr_height, 0), (0, self.lr_width)))(img)
        # crop4 = Cropping2D(((self.lr_height, 0), (self.lr_width, 0)))(img)
        # SR1 = self.SRG(crop1)
        # SR2 = self.SRG(crop2)
        # SR3 = self.SRG(crop3)
        # SR4 = self.SRG(crop4)
        # CR1 = self.CRG(SR1)
        # CR2 = self.CRG(SR2)
        # CR3 = self.CRG(SR3)
        # CR4 = self.CRG(SR4)
        # CR12 = Concatenate(axis=2)([CR1, CR2])
        # CR34 = Concatenate(axis=2)([CR3, CR4])
        # SRCR = Concatenate(axis=1)([CR12, CR34])

        # Extract image features of the generated img
        CR_features = self.vgg2(CR)
        CRSR_features = self.vgg(CRSR)
        # CRUP_features = self.vgg(CRUP)
        # SRCR_features = self.vgg(SRCR)
        # CRSRCRSR_feature = self.vgg(CRSRCRSR)

        # For the combined model we will only train the generator
        #self.discriminator.trainable = False
        # self.discriminator_frozen = Model(self.discriminator.input, self.discriminator.output)
        # self.discriminator_frozen.trainable = False
        # self.discriminator2_frozen = Model(self.discriminator2.input, self.discriminator2.output)
        # self.discriminator2_frozen.trainable = False

        # Discriminator determines validity of generated high res. images
        # validity2 = self.discriminator_frozen(CRSR)
        # validity = self.discriminator2_frozen(CRUP)

        # self.combined = Model([img], [validity, validity2, CRUP_features, CRSR_features, CR_features]) #[validity, validity2, CRUP_features, CRSR_features, CR_features]
        # self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'mse', 'mse', 'mse'],
        #                       loss_weights=self.weights,
        #                       optimizer=optimizer)
        self.combined = Model([img], [CR_features, CRSR_features])  # [validity, validity2, CRUP_features, CRSR_features, CR_features]
        self.combined.compile(loss=['mse', 'mse'],
                              loss_weights=self.weights,
                              optimizer=optimizer)

    def UpScale(self, input):
        width = input.shape[1] * 2
        height = input.shape[2] * 2
        return tf.image.resize(input, (width, height), method='bicubic')

    def DownScale(self, input):
        width = input.shape[1] // 2
        height = input.shape[2] // 2
        return tf.image.resize(input, (width, height), method='bicubic')

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

    def build_CRG(self):
        def DownScale(input):
            width = (input.shape[1]//2)
            height = (input.shape[2]//2)
            return tf.image.resize(input, (width, height), method='bicubic')

        def d_block(layer_input, filters, strides=1, bn=False):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = Activation('relu')(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        # Low resolution image input
        img_hr = Input(shape=self.hr_shape)
        filter_hr = Conv2D(3, kernel_size=3, strides=1, padding='same')(img_hr)
        filter_hr = Conv2D(3, kernel_size=3, strides=1, padding='same')(filter_hr)
        filter_hr = Conv2D(3, kernel_size=3, strides=1, padding='same')(filter_hr)
        filter_hr = Activation('tanh')(filter_hr)

        img_lr = Lambda(DownScale)(filter_hr)

        c1 = Conv2D(self.gf, kernel_size=3, strides=2, padding='same')(img_hr)
        c1 = Activation('relu')(c1)
        c2 = c1

        for _ in range(self.n_residual_blocks + 1):
            c2 = d_block(c2, self.gf, strides=1)

        # Generate high resolution output
        c3 = Conv2D(self.channels, kernel_size=3, strides=1, padding='same')(c2)#, activation='tanh')(c3)
        u1 = Add()([c3, img_lr])
        output = Activation('tanh')(u1)

        model = Model(inputs=[img_hr], outputs=[output])
        model.summary()
        return model

    def build_SRG(self):
        def UpScale(input):
            width = (input.shape[1] * 2)
            height = (input.shape[2] * 2)
            return tf.image.resize(input, (width, height), method='bicubic')

        def d_block(layer_input, filters, strides=1, bn=False):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = Activation('relu')(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)  #
        filter_lr = Conv2D(3, kernel_size=3, strides=1, padding='same')(img_lr)
        filter_lr = Conv2D(3, kernel_size=3, strides=1, padding='same')(filter_lr)
        filter_lr = Conv2D(3, kernel_size=3, strides=1, padding='same')(filter_lr)
        filter_lr = Activation('tanh')(filter_lr)

        img_hr = Lambda(UpScale)(filter_lr)

        c1 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)
        c2 = c1

        for _ in range(self.n_residual_blocks):
            c2 = d_block(c2, self.gf, strides=1)

        c2 = Lambda(UpScale)(c2)
        c3 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(c2)
        c3 = Activation('relu')(c3)

        # Generate high resolution output
        c3 = Conv2D(self.channels, kernel_size=3, strides=1, padding='same')(c3)
        u1 = Add()([c3, img_hr])

        output = Activation('tanh')(u1)

        model = Model(inputs=[img_lr], outputs=[output])
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

        t_epoch = []
        v_epoch = []
        t_D_Loss = []
        t_G_Loss = []
        t_D_Acc = []
        v_D_Acc = []
        v_D_Loss = []
        v_G_Loss = []
        best_test_loss = 999
        for epoch in range(epochs):
            for batch_i, (imgs_down, imgs) in enumerate(self.data_loader.load_batch(batch_size)):
                # valid = np.ones((batch_size,) + self.disc_patch)
                # fake = np.zeros((batch_size,) + self.disc_patch)
            #for batch_i in range(minibatch):

                # ----------------------
                #  Train Discriminator
                # ----------------------

                # From low res. image generate high res. version
                fakes_cr = self.CRG.predict(imgs)
                fakes_crsr = self.SRG.predict(fakes_cr)
                fakes_crup = self.UpScale(fakes_cr)

                # Train the discriminators (original images = real / generated = Fake)
                # d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                # d_loss_fake = self.discriminator.train_on_batch(fakes_crsr, fake)
                # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #
                # d_loss2_real = self.discriminator2.train_on_batch(imgs, valid)
                # d_loss2_fake = self.discriminator2.train_on_batch(fakes_crup, fake)
                # d_loss2 = 0.5 * np.add(d_loss2_real, d_loss2_fake)
                # ------------------
                #  Train Generator
                # ------------------

                # Extract ground truth image features using pre-trained VGG19 model
                image_features = self.vgg.predict(imgs)
                image_down_features = self.vgg2.predict(imgs_down)

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs], [image_down_features, image_features])

                # Plot the progress
                elapsed_time = datetime.datetime.now() - start_time
                # print(
                #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D loss2 : %f, acc: %3d%%] [G loss: %05f] time: %s " \
                #     % (epoch, epochs,
                #        batch_i, self.data_loader.n_batches,
                #        d_loss2[0], d_loss[0], 100 * d_loss[1], g_loss[0],
                #        elapsed_time))
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %05f] time: %s " \
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       g_loss[0], elapsed_time))
                g_loss_val = g_loss[0]
                if(g_loss_val < best_test_loss):
                    best_test_loss = g_loss_val
                    print("Best Model Epoch:%d, batch_i:%d, G loss: %f" % (epoch, batch_i, g_loss_val))
                    os.makedirs('./best/%s' % self.model_name, exist_ok=True)
                    self.CRG.save("./best/%s/bestCR_%s_%4f.h5" % (self.model_name, self.model_name, best_test_loss))
                    self.SRG.save("./best/%s/bestSR_%s_%4f.h5" % (self.model_name, self.model_name, best_test_loss))
                    self.test(epoch, batch_i, best_test_loss)

                if batch_i % sample_interval == 0 and batch_i != 0:
                    self.sample_images(epoch, batch_i)
                    self.sample_images_save(epoch, batch_i)

            test_size = batch_size
            # valid = np.ones((test_size,) + self.disc_patch)
            # fake = np.zeros((test_size,) + self.disc_patch)
            new_imgs_down, new_imgs = self.data_loader.load_data(test_size)
            new_fakes_cr = self.CRG.predict(new_imgs)
            new_fakes_crsr = self.SRG.predict(new_fakes_cr)
            new_fakes_crup = self.UpScale(new_fakes_cr)
            # d_loss_real = self.discriminator.test_on_batch(new_imgs, valid)
            # d_loss_fake = self.discriminator.test_on_batch(new_fakes_crsr, fake)
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # d_loss2_real = self.discriminator2.test_on_batch(new_imgs, valid)
            # d_loss2_fake = self.discriminator2.test_on_batch(new_fakes_crup, fake)
            # d_loss2 = 0.5 * np.add(d_loss2_real, d_loss2_fake)

            new_image_features = self.vgg.predict(new_imgs)
            new_image_down_features = self.vgg2.predict(new_imgs_down)
            g_loss = self.combined.test_on_batch([new_imgs],
                                                 [new_image_down_features, new_image_features])

            t_epoch.append(epoch)  # + (batch_i / (6000 / batch_size)))
            # t_D_Loss.append(d_loss[0] * 100)
            # t_D_Acc.append(d_loss[1] * 100)
            t_G_Loss.append(g_loss[0])

            test_imgs_down, test_imgs = self.data_loader.load_data(test_size, is_testing=True)
            test_fakes_cr = self.CRG.predict(test_imgs)
            test_fakes_crsr = self.SRG.predict(test_fakes_cr)
            test_fakes_crup = self.UpScale(test_fakes_cr)
            # test_d_loss_real = self.discriminator.test_on_batch(test_imgs, valid)
            # test_d_loss_fake = self.discriminator.test_on_batch(test_fakes_crsr, fake)
            # test_d_loss = 0.5 * np.add(test_d_loss_real, test_d_loss_fake)
            # test_d_loss2_real = self.discriminator2.test_on_batch(test_imgs, valid)
            # test_d_loss2_fake = self.discriminator2.test_on_batch(test_fakes_crup, fake)
            # test_d_loss2 = 0.5 * np.add(test_d_loss2_real, test_d_loss2_fake)

            test_image_features = self.vgg.predict(test_imgs)
            test_image_down_features = self.vgg2.predict(test_imgs_down)

            test_g_loss = self.combined.test_on_batch([test_imgs],
                                                      [test_image_down_features, test_image_features])

            v_epoch.append(epoch)  # + (batch_i / (6000 / batch_size)))
            # v_D_Loss.append(test_d_loss[0] * 100)
            # v_D_Acc.append(test_d_loss[1] * 100)
            v_G_Loss.append(test_g_loss[0])
            plt.clf()
            # plt.subplot(311)
            # plt.plot(t_epoch, t_D_Loss, c='red', label="train_D_loss*100")  # marker='^',
            # plt.plot(v_epoch, v_D_Loss, c='orange', label="test_D_loss*100")  # marker='v',
            # plt.legend(loc='upper right')
            # plt.grid()
            # plt.xlabel('epoch')
            # plt.ylabel('D_Loss')
            # plt.subplot(312)
            # plt.plot(t_epoch, t_D_Acc, c='red', label="train_D_Acc*100")  # marker='^',
            # plt.plot(v_epoch, v_D_Acc, c='orange', label="test_D_Acc*100")  # marker='v',
            # plt.legend(loc='upper right')
            # plt.grid()
            # plt.xlabel('epoch')
            # plt.ylabel('D_Acc')
            plt.subplot(111)
            plt.plot(t_epoch, t_G_Loss, c='blue', label="train_G_loss")  # marker='^',
            plt.plot(v_epoch, v_G_Loss, c='cyan', label="test_G_loss")  # marker='v',
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('G_Loss')
            plt.savefig('images/%s/fig.png' % self.model_name)
            if epoch % 5 == 0 and epoch != 0:
                self.save_models(epoch)

            if epoch % 10 == 0:
                batch_size = batch_size // 2
                sample_interval = sample_interval * 2

    def save_models(self, epoch):
        os.makedirs('saved_model/%s' % self.model_name, exist_ok=True)
        # self.discriminator.save("./saved_model/{}/Discriminator_{}.h5".format(self.model_name, epoch))
        self.CRG.save("./saved_model/{}/CRG_{}.h5".format(self.model_name, epoch))
        self.SRG.save("./saved_model/{}/SRG_{}.h5".format(self.model_name, epoch))

    def sample_images_save(self, epoch, batch_i, batch_size=1):
        os.makedirs('images/%s' % self.model_name, exist_ok=True)
        os.makedirs('images/%s/%d' % (self.model_name, epoch), exist_ok=True)

        imgs_down, imgs = self.data_loader.load_data(batch_size=batch_size, is_testing=True)
        fakes_cr = self.CRG.predict_on_batch(imgs)
        fakes_crsr = self.SRG.predict_on_batch(fakes_cr)

        imgs = 0.5 * imgs + 0.5
        imgs = (imgs * 255).astype(np.uint8)
        imgs_down = 0.5 * imgs_down + 0.5
        imgs_down = (imgs_down * 255).astype(np.uint8)
        fakes_cr = 0.5 * fakes_cr + 0.5
        fakes_cr = (fakes_cr * 255).astype(np.uint8)
        fakes_crsr = 0.5 * fakes_crsr + 0.5
        fakes_crsr = (fakes_crsr * 255).astype(np.uint8)

        for idx in range(batch_size):
            # Rescale images -1 - 1 => 0 - 1 => 0 - 255

            #img_lr = np.asarray(Image.fromarray(imgs_down[idx]).resize((self.lr_height, self.lr_height), self.interpolation))
            img_lrhr = np.asarray(Image.fromarray(imgs_down[idx]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            fake_crup = np.asarray(Image.fromarray(fakes_cr[idx]).resize((self.hr_height, self.hr_height), self.up_interpolation))

            compare1 = np.hstack((fakes_cr[idx], imgs_down[idx]))
            compare1 = np.vstack((imgs[idx], compare1))

            compare4 = np.hstack((imgs[idx], fakes_crsr[idx]))
            compare4 = np.hstack((compare4, fake_crup))
            compare4 = np.hstack((compare4, img_lrhr))

            Image.fromarray(compare1).save('images/%s/%d/%d_0compare1%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(compare4).save('images/%s/%d/%d_0compare4%d.png' % (self.model_name, epoch, batch_i, idx))

            Image.fromarray(imgs[idx]).save('images/%s/%d/%d_1hr%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(fake_crup).save('images/%s/%d/%d_2gen_crup%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(fakes_crsr[idx]).save('images/%s/%d/%d_3gen_crsr%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(img_lrhr).save('images/%s/%d/%d_4%s_lrhr%d.png' % (self.model_name, epoch, batch_i, self.interpolation_str, idx))
            Image.fromarray(fakes_cr[idx]).save('images/%s/%d/%d_5gen_lr%d.png' % (self.model_name, epoch, batch_i, idx))
            Image.fromarray(imgs_down[idx]).save('images/%s/%d/%d_6%s_lr%d.png' % (self.model_name, epoch, batch_i, self.interpolation_str, idx))

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.model_name, exist_ok=True)
        os.makedirs('images/%s/%d' % (self.model_name, epoch), exist_ok=True)
        r, c = 2, 4

        imgs_down, imgs = self.data_loader.load_data(batch_size=2, is_testing=True)
        fakes_cr = self.CRG.predict_on_batch(imgs)
        fakes_crsr = self.SRG.predict_on_batch(fakes_cr)

        # Rescale images -1 - 1 => 0 - 1 => 0 - 255
        imgs = 0.5 * imgs + 0.5
        imgs = (imgs * 255).astype(np.uint8)
        imgs_down = 0.5 * imgs_down + 0.5
        imgs_down = (imgs_down * 255).astype(np.uint8)
        fakes_cr = 0.5 * fakes_cr + 0.5
        fakes_cr = (fakes_cr * 255).astype(np.uint8)
        fakes_crsr = 0.5 * fakes_crsr + 0.5
        fakes_crsr = (fakes_crsr * 255).astype(np.uint8)

        imgs_lrhr = []
        fakes_crup = []
        for i in range(len(imgs)):
            # img_lr = np.asarray(Image.fromarray(imgs[i]).resize((self.lr_height, self.lr_height), self.interpolation))
            img_lrhr = np.asarray(Image.fromarray(imgs_down[i]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            imgs_lrhr.append(img_lrhr)
            fake_crup = np.asarray(Image.fromarray(fakes_cr[i]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            fakes_crup.append(fake_crup)

        # Save generated images and the high resolution originals
        titles = ['Original', 'CR->SR', 'CR->%s' % self.interpolation_str, 'Only %s' % self.interpolation_str]
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([imgs, fakes_crsr, fakes_crup, imgs_lrhr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1

        fig.savefig("images/%s/%d/%d.png" % (self.model_name, epoch, batch_i))
        plt.close()


    def test(self, epoch, batch_i, best_test_loss=0.):
        os.makedirs('images/%s/best/%d/%d_%4f' % (self.model_name, epoch, batch_i, best_test_loss), exist_ok=True)
        imgs_down, imgs = self.data_loader.load_select_data(dataset='./test/*')
        fakes_cr = self.CRG.predict_on_batch(imgs)
        fakes_crsr = self.SRG.predict_on_batch(fakes_cr)

        # Rescale images -1 - 1 => 0 - 1 => 0 - 255
        imgs = 0.5 * imgs + 0.5
        imgs = (imgs * 255).astype(np.uint8)
        imgs_down = 0.5 * imgs_down + 0.5
        imgs_down = (imgs_down * 255).astype(np.uint8)
        fakes_cr = 0.5 * fakes_cr + 0.5
        fakes_cr = (fakes_cr * 255).astype(np.uint8)
        fakes_crsr = 0.5 * fakes_crsr + 0.5
        fakes_crsr = (fakes_crsr * 255).astype(np.uint8)

        for i in range(len(imgs)):
            #img_lr = np.asarray(Image.fromarray(imgs[i]).resize((self.lr_height, self.lr_height), self.interpolation))
            img_lrhr = np.asarray(Image.fromarray(imgs_down[i]).resize((self.hr_height, self.hr_height), self.up_interpolation))
            fake_crup = np.asarray(Image.fromarray(fakes_cr[i]).resize((self.hr_height, self.hr_height), self.up_interpolation))

            compare1 = np.hstack((fakes_cr[i], imgs_down[i]))
            compare1 = np.vstack((imgs[i], compare1))
            Image.fromarray(compare1).save('images/%s/best/%d/%d_%4f/%d_%d_%dcompare1.png' % (self.model_name, epoch, batch_i, best_test_loss, epoch, batch_i, i))

            compare4 = np.hstack((imgs[i], fakes_crsr[i]))
            compare4 = np.hstack((compare4, fake_crup))
            compare4 = np.hstack((compare4, img_lrhr))
            Image.fromarray(compare4).save('images/%s/best/%d/%d_%4f/%d_%d_%dcompare4.png' % (self.model_name, epoch, batch_i, best_test_loss, epoch, batch_i, i))
        print("finish")

if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=50, batch_size=32, sample_interval=5)
