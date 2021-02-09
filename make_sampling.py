import scipy
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import imread

class DataLoader2():
    def __init__(self, data_path, img_res=(256, 256)):
        self.data_path = data_path
        self.data_type = ""
        self.img_res = img_res

    def sampling(self, scale_factor=2):

        path = glob('./%s/*' % self.data_path)

        for img_path in path:
            filename = os.path.basename(img_path)
            print(filename)
            if os.path.isdir(img_path):
                print('Directory pass!')
                continue

            img = self.imread(img_path)
            if "gen" in filename:
                self.data_type = "generated"
            elif "nearest" in filename:
                self.data_type = "nearest"
            else:
                print('Not Target pass!')
                continue

            os.makedirs('./%s/%s' % (self.data_path, self.data_type), exist_ok=True)
            h, w = self.img_res
            low_h, low_w = int(h / scale_factor), int(w / scale_factor)

            img_lr = scipy.misc.imresize(img, (low_h, low_w), 'nearest')
            scipy.misc.imsave('./%s/%s/%s' % (self.data_path, self.data_type, filename), img_lr)
            print('clear!')
        print('Finished!')

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    loader = DataLoader2(data_path="images/210105_alphaPixele3,1/17")
    loader.sampling()
