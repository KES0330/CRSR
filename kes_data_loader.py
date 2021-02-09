import scipy
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256), low_img_res=(128, 128),
                 interpolation=Image.NEAREST, up_interpolation=Image.NEAREST):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.low_img_res = low_img_res
        self.interpolation = interpolation
        self.up_interpolation = up_interpolation

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        imgs_lrhr = []
        for img_path in batch_images:
            img = Image.open(img_path)
            img.convert('RGB')

            img_hr = img.resize(self.img_res, self.up_interpolation)
            img_lr = img.resize(self.low_img_res, self.interpolation)
            img_lrhr = img_lr.resize(self.img_res, self.up_interpolation)

            img_hr = np.asarray(img_hr)
            img_lr = np.asarray(img_lr)
            img_lrhr = np.asarray(img_lrhr)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)
                img_lrhr = np.fliplr(img_lrhr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)
            imgs_lrhr.append(img_lrhr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs_lrhr = np.array(imgs_lrhr) / 127.5 - 1.

        return imgs_hr, imgs_lr#imgs_lrhr

    def load_data_with_idx(self, idx=0, batch_size=1, scale_factor=2, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        imgs_lrhr = []
        for img_path in batch_images:
            img = Image.open(img_path)
            img.convert('RGB')

            img_hr = img.resize(self.img_res, self.up_interpolation)
            img_lr = img.resize(self.low_img_res, self.interpolation)
            img_lrhr = img_lr.resize(self.img_res, self.up_interpolation)

            img_hr = np.asarray(img_hr)
            img_lr = np.asarray(img_lr)
            img_lrhr = np.asarray(img_lrhr)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)
                img_lrhr = np.fliplr(img_lrhr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)
            imgs_lrhr.append(img_lrhr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs_lrhr = np.array(imgs_lrhr) / 127.5 - 1.

        return imgs_hr, imgs_lr#imgs_lrhr

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

    def get_total(self, batch_size=1, scale_factor=2, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        return len(path)

    def get_batch(self, batch_size=1, scale_factor=2, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)
        return self.n_batches

    def load_batch(self, batch_size=1, scale_factor=2, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path = np.random.choice(path, total_samples, replace=False)

        for i in range(self.n_batches): # n_batches-1
            minibatch = path[i*batch_size:(i+1)*batch_size]
            imgs_hr = []
            imgs_lr = []
            imgs_lrhr = []
            for img_path in minibatch:
                img = Image.open(img_path)
                img.convert('RGB')

                img_hr = img.resize(self.img_res, self.up_interpolation)
                img_lr = img.resize(self.low_img_res, self.interpolation)
                img_lrhr = img_lr.resize(self.img_res, self.up_interpolation)

                img_hr = np.asarray(img_hr)
                img_lr = np.asarray(img_lr)
                img_lrhr = np.asarray(img_lrhr)

                # If training => do random flip
                if not is_testing and np.random.random() < 0.5:
                    img_hr = np.fliplr(img_hr)
                    img_lr = np.fliplr(img_lr)
                    img_lrhr = np.fliplr(img_lrhr)

                imgs_hr.append(img_hr)
                imgs_lr.append(img_lr)
                imgs_lrhr.append(img_lrhr)

            imgs_hr = np.array(imgs_hr) / 127.5 - 1.
            imgs_lr = np.array(imgs_lr) / 127.5 - 1.
            imgs_lrhr = np.array(imgs_lrhr) / 127.5 - 1.

            yield imgs_hr, imgs_lr#imgs_lrhr

    def load_select_data(self, dataset=""):

        path = glob(dataset)

        imgs_hr = []
        imgs_lr = []
        imgs_lrhr = []

        for img_path in path:
            if os.path.isdir(img_path):
                continue
            img = Image.open(img_path)
            img.convert('RGB')
            filename = os.path.basename(img_path)

            img_hr = img.resize(self.img_res, self.up_interpolation)
            img_lr = img.resize(self.low_img_res, self.interpolation)
            img_lrhr = img_lr.resize(self.img_res, self.up_interpolation)

            imgs_hr.append(np.asarray(img_hr))
            imgs_lr.append(np.asarray(img_lr))
            imgs_lrhr.append(np.asarray(img_lrhr))

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs_lrhr = np.array(imgs_lrhr) / 127.5 - 1.

        return imgs_hr, imgs_lr#imgs_lrhr

    def load_hr_data(self, dataset=""):

        path = glob(dataset)

        imgs_hr = []

        for img_path in path:
            if os.path.isdir(img_path):
                continue
            img = Image.open(img_path)
            img.convert('RGB')



            img_hr = img.resize(self.img_res, self.up_interpolation)
            img_lr = img.resize(self.low_img_res, self.interpolation)
            img_lrhr = img_lr.resize(self.img_res, self.up_interpolation)

            imgs_hr.append(np.asarray(img_hr))
            imgs_lr.append(np.asarray(img_lr))
            imgs_lrhr.append(np.asarray(img_lrhr))

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs_lrhr = np.array(imgs_lrhr) / 127.5 - 1.

        return imgs_hr, imgs_lr#imgs_lrhr