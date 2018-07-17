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
    def outputs(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    @property
    def num_inputs(self):
        raise NotImplementedError

    @property
    def num_outputs(self):
        raise NotImplementedError


class MNIST(Dataset):

    def __init__(self, image_file, label_file):
        with gzip.open(image_file) as fin:
            self.images = _read_images(fin)

        with gzip.open(label_file) as fin:
            self.labels = _read_labels(fin)

        self._size, num_rows, num_cols = self.images.shape
        self._num_inputs = num_rows * num_cols
        self._num_outputs = 10

        self._inputs = self.images.reshape(self._size, self._num_inputs) / 255
        self._outputs = transform_to_one_hot(self.labels, self._num_outputs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def size(self):
        return self._size

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_outputs(self):
        return self._num_outputs

    def show_image(self, image_index, title=''):
        title = "label %d - %s" % (self.labels[image_index], title)
        image = self.images[image_index]
        image_rgb = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        plt.figure()
        plt.title(title)
        return plt.imshow(image_rgb,)

