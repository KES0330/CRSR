import scipy
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256), low_img_res=(128, 128), interpolation=Image.BICUBIC):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.low_img_res = low_img_res
        self.interpolation = interpolation

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        imgs_down = []

        for img_path in batch_images:
            img = Image.open(img_path)
            img.convert('RGB')
            width, height = self.img_res
            w, h = img.size
            rw = rh = 0
            if w > 256:
                rw = random.randrange(0, w - 256)
            if h > 256:
                rh = random.randrange(0, h - 256)
            img = img.crop((rw, rh, rw + width, rh + height))
            #img = img.resize(self.img_res, Image.BILINEAR)
            img_down = img.resize(self.low_img_res, self.interpolation)

            img = np.asarray(img)
            img_down = np.asarray(img_down)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)
                img_down = np.fliplr(img_down)
            if not is_testing and np.random.random() < 0.5:
                img = np.flipud(img)
                img_down = np.flipud(img_down)

            imgs.append(img)
            imgs_down.append(img_down)

        imgs = np.array(imgs) / 127.5 - 1.
        imgs_down = np.array(imgs_down) / 127.5 - 1.

        return imgs_down, imgs

    def load_data_with_idx(self, idx=0, batch_size=1, scale_factor=2, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        imgs_down = []

        for img_path in batch_images:
            img = Image.open(img_path)
            img.convert('RGB')
            width, height = self.img_res
            w, h = img.size
            rw = rh = 0
            if w > 256:
                rw = random.randrange(0, w - 256)
            if h > 256:
                rh = random.randrange(0, h - 256)
            img = img.crop((rw, rh, rw + width, rh + height))
            # img = img.resize(self.img_res, Image.BILINEAR)
            img_down = img.resize(self.low_img_res, self.interpolation)

            img = np.asarray(img)
            img_down = np.asarray(img_down)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)
                img_down = np.fliplr(img_down)
            if not is_testing and np.random.random() < 0.5:
                img = np.flipud(img)
                img_down = np.flipud(img_down)

            imgs.append(img)
            imgs_down.append(img_down)

        imgs = np.array(imgs) / 127.5 - 1.
        imgs_down = np.array(imgs_down) / 127.5 - 1.

        return imgs_down, imgs

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
            imgs = []
            imgs_down = []

            for img_path in minibatch:
                img = Image.open(img_path)
                img.convert('RGB')
                width, height = self.img_res
                w, h = img.size

                rw = rh = 0
                if w > 256:
                    rw = random.randrange(0, w - 256)
                if h > 256:
                    rh = random.randrange(0, h - 256)
                img = img.crop((rw, rh, rw + width, rh + height))
                # img = img.resize(self.img_res, Image.BILINEAR)
                img_down = img.resize(self.low_img_res, self.interpolation)

                img = np.asarray(img)
                img_down = np.asarray(img_down)

                # If training => do random flip
                if not is_testing and np.random.random() < 0.5:
                    img = np.fliplr(img)
                    img_down = np.fliplr(img_down)
                if not is_testing and np.random.random() < 0.5:
                    img = np.flipud(img)
                    img_down = np.flipud(img_down)

                imgs.append(img)
                imgs_down.append(img_down)

            imgs = np.array(imgs) / 127.5 - 1.
            imgs_down = np.array(imgs_down) / 127.5 - 1.
            yield imgs_down, imgs

    def load_select_data(self, dataset=""):

        path = glob(dataset)

        imgs = []
        imgs_down = []

        for img_path in path:
            if os.path.isdir(img_path):
                continue
            img = Image.open(img_path)
            img.convert('RGB')
            # width, height = self.img_res
            # w, h = img.size
            # rw = random.randrange(0, w - 256)
            # rh = random.randrange(0, h - 256)
            # img = img.crop((rw, rh, rw + width, rh + height))
            img = img.resize(self.img_res, self.interpolation)
            img_down = img.resize(self.low_img_res, self.interpolation)

            img = np.asarray(img)
            img_down = np.asarray(img_down)

            imgs.append(img)
            imgs_down.append(img_down)

        imgs = np.array(imgs) / 127.5 - 1.
        imgs_down = np.array(imgs_down) / 127.5 - 1.

        return imgs_down, imgs