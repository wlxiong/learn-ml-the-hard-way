import unittest

import common
from narray import lib as np


class TestIm2row(unittest.TestCase):

    def test_unit_filter_size(self):
        image_size=(3,3,1)
        filter_size=(1,1,1)
        stride=(1,1)
        pad=(0,0)

        indices = common.im2row_index(image_size, filter_size, stride, pad)
        expected = np.arange(np.prod(image_size)).reshape(np.prod(image_size), 1)
        np.testing.assert_array_equal(indices, expected)

    def test_filter_size_as_large_as_image(self):
        image_size=(3,3,1)
        filter_size=(3,3,1)
        stride=(1,1)
        pad=(0,0)

        indices = common.im2row_index(image_size, filter_size, stride, pad)
        expected = np.arange(np.prod(image_size))[np.newaxis, :]
        np.testing.assert_array_equal(indices, expected)


if __name__ == '__main__':
    unittest.main()
