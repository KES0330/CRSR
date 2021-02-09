from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
from keras.utils import np_utils
import matplotlib.pyplot as plt
import sys
from kes_data_loader import DataLoader
import numpy as np
import os
from keras.preprocessing.image import array_to_img

import math
import scipy

from PIL import Image
from SSIM_PIL import compare_ssim
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

config = tf.compat.v1.ConfigProto()  # tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=config))  # tf.Session
K.set_learning_phase(1)

old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Bicubic(layers.Layer):

    def __init__(self, width=128, height=128):
        super(Bicubic, self).__init__()
        self.width = width
        self.height = height

    def call(self, inputs):
        return tf.image.resize(inputs, (self.width, self.height), method='nearest')

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return (input_shape[0], self.width, self.height, input_shape[3])


class Clip(layers.Layer):

    def __init__(self):
        super(Clip, self).__init__()

    def call(self, inputs):
        out = tf.keras.backend.clip(inputs, -1, 1)
        return out

class KNN():
    def __init__(self):
        # Input shape
        self.scale_factor = 2  # 2x
        self.channels = 3
        self.hr_height = 256  # High resolution height
        self.hr_width = 256  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.lr_height = int(self.hr_height / self.scale_factor)  # Low resolution height
        self.lr_width = int(self.hr_width / self.scale_factor)  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.height = 256
        self.width = 256
        self.dataset_name = "Linnaeus5"
        self.model_name = "210201_NewPlan_NoB_UpBilinear_(000)e3,1,1"
        self.load_name = "best_" + self.model_name
        #self.load_name = "Generator_20"
        self.generator = load_model("./{}.h5".format(self.load_name), custom_objects={'tf': tf})
        self.interpolation = Image.BILINEAR
        self.up_interpolation = Image.BILINEAR
        # Configure data loader
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width),
                                      low_img_res=(self.lr_height, self.lr_width),
                                      interpolation=self.interpolation,
                                      up_interpolation=self.up_interpolation)

    def np2img(self, array):
        return Image.fromarray((array * 255).astype(np.uint8))

    def SSIM(self, im1, im2):
        return compare_ssim(im1, im2)

    def test(self):
        os.makedirs('./test/%s' % self.model_name, exist_ok=True)
        hr, lr = self.data_loader.load_select_data(dataset='./test/*')
        len_hr = len(hr)
        fakes_lr, fakes_hr = self.generator.predict(hr)
        hr = (0.5 * hr + 0.5)
        lr = (0.5 * lr + 0.5)
        fakes_hr = (0.5 * fakes_hr + 0.5)
        fakes_lr = (0.5 * fakes_lr + 0.5)
        ssims1 = 0.
        ssims2 = 0.

        for i in range(len_hr):
            os.makedirs('./test/%s/%d' % (self.model_name, i), exist_ok=True)
            img_hr = np.asarray(self.np2img(hr[i]).resize((self.hr_height, self.hr_height), Image.BILINEAR))
            img_lr = np.asarray(self.np2img(hr[i]).resize((self.lr_height, self.lr_height), Image.BILINEAR))
            img_lrhr = np.asarray(
                Image.fromarray(img_lr).resize((self.hr_height, self.hr_height), Image.BILINEAR))
            fake_hr = np.asarray(
                self.np2img(fakes_lr[i]).resize((self.hr_height, self.hr_height), Image.BILINEAR))
            fake_lr = np.asarray(
                self.np2img(fakes_lr[i]).resize((self.lr_height, self.lr_height), Image.BILINEAR))

            ssim1 = self.SSIM(self.np2img(hr[i]), self.np2img(fakes_hr[i]))
            ssim2 = self.SSIM(self.np2img(hr[i]), Image.fromarray(img_lrhr))
            ssims1 += ssim1
            ssims2 += ssim2
            (self.np2img(hr[i])).save('./test/%s/%d/%d_0_original.png' % (self.model_name, i, i))
            self.np2img(fakes_lr[i]).save('./test/%s/%d/%d_1_generated_%.4f.png' % (self.model_name, i, i, ssim1))
            self.np2img(lr[i]).save('./test/%s/%d/%d_2_interpolated_%.4f.png' % (self.model_name, i, i, ssim2))
            compare1 = np.hstack((fakes_lr[i], img_lr))
            compare1 = np.vstack((hr[i], compare1))
            (self.np2img(compare1)).save('./test/%s/compare_test_%d.png' % (self.model_name, i))
            compare2 = np.hstack((hr[i], fakes_hr[i]))
            compare2 = np.hstack((compare2, img_lrhr))
            (self.np2img(compare2)).save('./test/%s/compare_test2_%d.png' % (self.model_name, i))
        ssims1 /= len_hr
        ssims2 /= len_hr
        print('%.4f, %.4f' %(ssims1, ssims2))
        print("finish")
    def compare_all(self):
        os.makedirs('./compare_all/%s' % self.model_name, exist_ok=True)
        hr, lr = self.data_loader.load_select_data(dataset='./test/*')
        len_hr = len(hr)
        fakes_lr, fakes_hr = self.generator.predict(hr)
        hr = (0.5 * hr + 0.5)
        hr = (hr * 255).astype(np.uint8)
        lr = (0.5 * lr + 0.5)
        lr = (lr * 255).astype(np.uint8)
        fakes_hr = (0.5 * fakes_hr + 0.5)
        fakes_hr = (fakes_hr * 255).astype(np.uint8)
        fakes_lr = (0.5 * fakes_lr + 0.5)
        fakes_lr = (fakes_lr * 255).astype(np.uint8)
        ssims_nearest = 0.
        ssims_bilinear = 0.
        ssims_bicubic = 0.
        ssims_lanczos = 0.
        ssims_hamming = 0.
        ssims_generated = 0.
        for i in range(len_hr):
            img_nearest = np.asarray(Image.fromarray(hr[i]).resize((self.lr_height, self.lr_width), Image.NEAREST))
            img_bilinear = np.asarray(Image.fromarray(hr[i]).resize((self.lr_height, self.lr_width), Image.BILINEAR))
            img_bicubic = np.asarray(Image.fromarray(hr[i]).resize((self.lr_height, self.lr_width), Image.BICUBIC))
            img_lanczos = np.asarray(Image.fromarray(hr[i]).resize((self.lr_height, self.lr_width), Image.LANCZOS))
            img_hamming = np.asarray(Image.fromarray(hr[i]).resize((self.lr_height, self.lr_width), Image.HAMMING))
            img_generated = np.asarray(Image.fromarray(fakes_lr[i]))
            ssim_nearest = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_nearest).resize((self.hr_height, self.hr_width), Image.BILINEAR)))
            ssim_bilinear = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_bilinear).resize((self.hr_height, self.hr_width), Image.BILINEAR)))
            ssim_bicubic = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_bicubic).resize((self.hr_height, self.hr_width), Image.BILINEAR)))
            ssim_lanczos = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_lanczos).resize((self.hr_height, self.hr_width), Image.BILINEAR)))
            ssim_hamming = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_hamming).resize((self.hr_height, self.hr_width), Image.BILINEAR)))
            ssim_generated = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_generated).resize((self.hr_height, self.hr_width), Image.BILINEAR)))

            ssims_nearest += ssim_nearest
            ssims_bilinear += ssim_bilinear
            ssims_bicubic += ssim_bicubic
            ssims_lanczos += ssim_lanczos
            ssims_hamming += ssim_hamming
            ssims_generated += ssim_generated
            compare1 = np.hstack([img_nearest, img_bilinear])
            compare1 = np.hstack([compare1, img_bicubic])
            compare2 = np.hstack([img_lanczos, img_hamming])
            compare2 = np.hstack([compare2, img_generated])
            compare = np.vstack([compare1, compare2])
            compare = np.hstack([hr[i], compare])
            Image.fromarray(compare).save('./compare_all/%s/%d_stack.png' % (self.model_name, i))

            r = 2
            c = 3
            fig, axs = plt.subplots(r, c)
            cnt = 0
            axs[0, 0].imshow(np.asarray(hr[i]))
            axs[0, 0].set_title('Nearest_%.4f' % ssim_nearest)
            axs[0, 0].axis('off')
            axs[0, 1].imshow(np.asarray(hr[i]))
            axs[0, 1].set_title('Bilinear_%.4f' % ssim_bilinear)
            axs[0, 1].axis('off')
            axs[0, 2].imshow(np.asarray(hr[i]))
            axs[0, 2].set_title('Bicubic_%.4f' % ssim_bicubic)
            axs[0, 2].axis('off')
            axs[1, 0].imshow(np.asarray(hr[i]))
            axs[1, 0].set_title('Lanczos_%.4f' % ssim_lanczos)
            axs[1, 0].axis('off')
            axs[1, 1].imshow(np.asarray(hr[i]))
            axs[1, 1].set_title('Hamming_%.4f' % ssim_hamming)
            axs[1, 1].axis('off')
            axs[1, 2].imshow(np.asarray(hr[i]))
            axs[1, 2].set_title('Generated_%.4f' % ssim_generated)
            axs[1, 2].axis('off')

            fig.savefig('./compare_all/%s/%d.png' % (self.model_name, i))
            plt.close()
        ssims_nearest /= len_hr
        ssims_bilinear /= len_hr
        ssims_bicubic /= len_hr
        ssims_lanczos /= len_hr
        ssims_hamming /= len_hr
        ssims_generated /= len_hr
        print('%.4f, %.4f, %.4f' % (ssims_nearest, ssims_bilinear, ssims_bicubic))
        print('%.4f, %.4f, %.4f' % (ssims_lanczos, ssims_hamming, ssims_generated))
        print("finish")

    def test_hr(self):
        os.makedirs('./testhr/%s' % self.model_name, exist_ok=True)
        hr, lr = self.data_loader.load_select_data(dataset='./test/*')
        len_hr = len(hr)

        for idx in range(len_hr):
            img = hr[idx].copy()
            gen = img[:128, :128, :].copy()
            h, w, c = hr[idx].shape
            mini_h = h // 256
            mini_w = w // 256
            for i in range(mini_h):
                for j in range(mini_w):
                    fake = self.generator.predict(hr[idx, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])
                    gen[idx, i * 128:(i + 1) * 128, j * 128:(j + 1) * 128, :] = fake

            img = (0.5 * img + 0.5)
            img = (img * 255).astype(np.uint8)
            gen = (0.5 * gen + 0.5)
            gen = (gen * 255).astype(np.uint8)

            img_lr = np.asarray(Image.fromarray(img).resize((self.lr_height, self.lr_height), Image.BILINEAR))
            img_lrhr = np.asarray(Image.fromarray(img_lr).resize((self.hr_height, self.hr_height), Image.BILINEAR))
            gen_hr = np.asarray(Image.fromarray(gen).resize((self.hr_height, self.hr_height), Image.BILINEAR))

            ssim_nearest = self.SSIM(Image.fromarray(hr[i]), (Image.fromarray(img_nearest).resize((self.hr_height, self.hr_width), Image.BILINEAR)))





if __name__ == '__main__':
    cnn = KNN()
    #cnn.test()
    cnn.compare_all()
