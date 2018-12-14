import numpy as np


def split_data(x, y, batch_size, shuffle=False):
    if shuffle:
        assert len(x) == len(y)
        indices = np.random.permutation(len(x))
        x = x[indices]
        y = y[indices]
    batches = []
    for i in range(0, len(x), batch_size):
        xs = x[i:i+batch_size]
        ys = y[i:i+batch_size]
        batches.append((xs, ys))
    return batches

def transform_to_one_hot(labels, num_outputs):
    z = np.zeros((labels.size, num_outputs))
    z[np.arange(labels.size), labels] = 1
    return z

def get_padded_image_size(image_size, filter_size, stride, pad):
    width, height, depth = image_size
    width_padded = width + 2 * pad[0]
    height_padded = height + 2 * pad[1]
    image_size_padded = (width_padded, height_padded, depth)
    assert (width_padded - filter_size[0]) % stride[0] == 0
    assert (height_padded - filter_size[1]) % stride[1] == 0
    return image_size_padded

# get_1st_index() and im2row_index() are basically line-by-line translations of im2col in octave:
# https://sourceforge.net/p/octave/image/ci/default/tree/inst/im2col.m
def get_1st_index(image_size, filter_size):
    # reverse the dimensions since numpy arrays are stored in row-major order
    # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
    image_size, filter_size = tuple(reversed(image_size)), tuple(reversed(filter_size))
    # example - image_size: [17, 17, 3], filter_size: [3, 3, 3]
    stride = np.cumprod((1,) + image_size[:-1])
    # stride: [1, 17, 289]
    limit = filter_size * stride
    # limit: [3, 3, 3] * [1, 17, 289] = [3, 51, 867]
    ind = np.zeros((1,1), dtype=np.int)
    for dim in range(len(image_size)):
        ind = np.arange(0, limit[dim], stride[dim], dtype=np.int).reshape((-1, 1)) + ind.reshape(-1)
        # dim 0: ind = [0] + [0, 1, 2] = [0, 1, 2]
        # dim 1: ind = [[0], [17], [34]] + [0, 1, 2] = [[0, 1, 2], [17, 18, 19], [34, 35, 36]]
        # dim 2: ind = ...
    return ind, stride

def im2row_index(image_size, filter_size, stride, pad):
    # get padded image size
    image_size_padded = get_padded_image_size(image_size, filter_size, stride, pad)
    # get linear indices for the first block
    ind, dim_stride = get_1st_index(image_size_padded, filter_size)
    stride = np.append(stride, np.ones(len(dim_stride) - len(stride), dtype=int))
    stride = np.flip(stride, axis=0)
    stride = dim_stride * stride
    # get linear indices for all blocks
    slides = np.array(image_size_padded, dtype=np.int) - np.array(filter_size, dtype=np.int)
    slides = np.flip(slides, axis=0)
    limit = slides * dim_stride + 1
    for dim in range(len(image_size_padded)):
        ind = np.arange(0, limit[dim], stride[dim], dtype=np.int).reshape((-1, 1)) + ind.reshape(-1)
    return ind.reshape(-1, np.prod(filter_size))

def row2im(rows, row_indices, image_size, filter_size, stride, pad):
    # convert rows to image
    flatten_rows = rows.reshape((-1, np.prod(row_indices.shape)))
    image_padded = np.apply_along_axis(lambda g: np.bincount(row_indices.reshape(-1), g), 1, flatten_rows)
    image_size_padded = get_padded_image_size(image_size, filter_size, stride, pad)
    image_padded = image_padded.reshape((-1, *image_size_padded))
    # cut padding areas
    width_padded, height_padded, _ = image_size_padded
    width_indices, height_indices = np.ix_(range(pad[0], width_padded - pad[0]), range(pad[1], height_padded - pad[1]))
    return image_padded[:, width_indices, height_indices, ...]
