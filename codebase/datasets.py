import numpy as np
import os
from scipy.io import loadmat
import scipy
import sys
import cPickle as pkl
import tensorbayes as tb
from itertools import izip
from codebase.utils import u2t, s2t

PATH = './data/'

def get_info(domain_id, domain):
    train, test = domain.train, domain.test
    print '{} info'.format(domain_id)
    print 'Train X/Y shapes: {}, {}'.format(train.images.shape, train.labels.shape)
    print 'Train X min/max/cast: {}, {}, {}'.format(
        train.images.min(),
        train.images.max(),
        train.cast)
    print 'Test shapes: {}, {}'.format(test.images.shape, test.labels.shape)
    print 'Test X min/max/cast: {}, {}, {}\n'.format(
        test.images.min(),
        test.images.max(),
        test.cast)

class Data(object):
    def __init__(self, images, labels=None, labeler=None, cast=False):
        """Data object constructs mini-batches to be fed during training

        images - (NHWC) data
        labels - (NK) one-hot data
        labeler - (tb.function) returns simplex value given an image
        cast - (bool) converts uint8 to [-1, 1] float
        """
        self.images = images
        self.labels = labels
        self.labeler = labeler
        self.cast = cast

    def preprocess(self, x):
        if self.cast:
            return u2t(x)
        else:
            return x

    def next_batch(self, bs):
        """Constructs a mini-batch of size bs without replacement
        """
        idx = np.random.choice(len(self.images), bs, replace=False)
        x = self.preprocess(self.images[idx])
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y

class Mnist(object):
    def __init__(self, shape=(32, 32, 3)):
        """MNIST domain train/test data

        shape - (3,) HWC info
        """
        print "Loading MNIST"
        data = np.load(os.path.join(PATH, 'mnist.npz'))

        trainx = data['x_train']
        trainy = data['y_train']
        trainy = np.eye(10)[trainy].astype('float32')

        testx = data['x_test']
        testy = data['y_test'].astype('int')
        testy = np.eye(10)[testy].astype('float32')

        trainx = self.resize_cast(trainx, shape)
        testx = self.resize_cast(testx, shape)

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

    @staticmethod
    def resize_cast(x, shape):
        H, W, C = shape
        x = x.reshape(-1, 28, 28)

        resized_x = np.empty((len(x), H, W), dtype='float32')
        for i, img in enumerate(x):
            # imresize returns uint8
            resized_x[i] = u2t(scipy.misc.imresize(img, (H, W)))

        # Retile to make RGB
        resized_x = resized_x.reshape(-1, H, W, 1)
        resized_x = np.tile(resized_x, (1, 1, 1, C))
        return resized_x


class Svhn(object):
    def __init__(self, train='train'):
        """SVHN domain train/test data

        train - (str) flag for using 'train' or 'extra' data
        """
        print "Loading SVHN"
        train = loadmat(os.path.join(PATH, '{:s}_32x32.mat'.format(train)))
        test = loadmat(os.path.join(PATH, 'test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train)
        testx, testy = self.change_format(test)

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat):
        """Convert X: (HWCN) -> (NHWC) and Y: [1,...,10] -> one-hot
        """
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        y = np.eye(10)[y]
        return x, y
