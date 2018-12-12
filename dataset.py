import gzip
import array
import struct
import numpy as np
import matplotlib.pyplot as plt
from common import transform_to_one_hot


def _read_int32(fin):
    buf = fin.read(4)
    return struct.unpack('>i', buf)[0]


def _read_byte(fin):
    buf = fin.read(1)
    return struct.unpack('B', buf)[0]


def _read_images(fin):
    magic = _read_int32(fin)
    assert magic == 0x00000803, "magic number != 0x%08x" % magic
    num_images = _read_int32(fin)
    num_rows = _read_int32(fin)
    num_cols = _read_int32(fin)
    data = array.array('B')
    data.fromfile(fin, num_images * num_rows * num_cols)
    return np.array(data).reshape((num_images, num_rows, num_cols))


def _read_labels(fin):
    magic = _read_int32(fin)
    assert magic == 0x00000801, "magic number != 0x%08x" % magic
    num_labels = _read_int32(fin)
    data = array.array('B')
    data.fromfile(fin, num_labels)
    return np.array(data)


class Dataset(object):

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def sample_size(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        raise NotImplementedError

    @property
    def target_dim(self):
        raise NotImplementedError


class MNIST(Dataset):

    def __init__(self, image_file, label_file):
        with gzip.open(image_file) as fin:
            self.images = _read_images(fin)

        with gzip.open(label_file) as fin:
            self.labels = _read_labels(fin)

        self._num_images, self._image_width, self._image_height = self.images.shape
        self._input_dim = (self._image_width, self._image_height, 1)
        self._target_dim = 10

        self._inputs = self.images.reshape((-1, *self.input_dim)) / 255.0
        self._targets = transform_to_one_hot(self.labels, self._target_dim)

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def sample_size(self):
        return self._num_images

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def target_dim(self):
        return self._target_dim

    @property
    def image_size(self):
        return (self._image_width, self._image_height)

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_height(self):
        return self._image_height

    def show_image(self, image_index, title=''):
        title = "label %d - %s" % (self.labels[image_index], title)
        image = self.images[image_index]
        image_rgb = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        plt.figure()
        plt.title(title)
        return plt.imshow(image_rgb,)
