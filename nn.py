import numpy as np
import random
import common
import neurons


class NeuralNetwork(object):

    def __init__(self, data_layer, compute_layers, loss_layer):
        self.data_layer = data_layer
        self.compute_layers = compute_layers
        self.loss_layer = loss_layer

    def evaluate(self, inputs):
        outputs = inputs
        for layer in self.compute_layers:
            if not layer.skip_eval:
                outputs = layer.forward(inputs)
                inputs = outputs
        return outputs

    def train(self, learning_rate):
        cumulative_loss = 0.0
        for inputs, targets in self.data_layer.forward():
            loss = self.forward(inputs, targets)
            self.backward()
            self.update_parameters(learning_rate)
            cumulative_loss += loss
        return cumulative_loss

    def forward(self, inputs, targets):
        outputs = inputs
        for layer in self.compute_layers:
            outputs = layer.forward(inputs)
            inputs = outputs
        loss = self.loss_layer.forward(outputs, targets)
        return loss

    def backward(self):
        grad_out = self.loss_layer.backward()
        grad_in = grad_out
        for layer in reversed(self.compute_layers):
            grad_in = layer.backward(grad_out)
            grad_out = grad_in

    def update_parameters(self, learning_rate):
        for layer in self.compute_layers:
            params = layer.parameters()
            grad = layer.gradient()
            for p, g in zip(params, grad):
                p[:] = p - learning_rate * g


class ComputeLayer(object):

    def __init__(self, num_inputs, num_outputs, neuron_type):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neuron = neuron_type() if neuron_type is not None else None

    def __str__(self):
        return "%s(%d, %d, %s)" % (type(self).__name__, self.num_inputs, self.num_outputs, self.neuron)

    __repr__ = __str__

    @property
    def skip_eval(self):
        return False

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError


class DataLayer(object):

    def __init__(self, inputs, targets):
        self.inputs, self.targets = inputs, targets

    def forward(self):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


class BatchDataLayer(DataLayer):

    def __init__(self, inputs, targets, batch_size, shuffle=False):
        super(BatchDataLayer, self).__init__(inputs, targets)
        self.batch_size = batch_size
        self.dataset = list(zip(inputs, targets))
        if shuffle:
            random.shuffle(self.dataset)

    def forward(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i+self.batch_size]
            inputs, targets = zip(*batch)
            yield np.array(inputs), np.array(targets)

    def backward(self, grad_out):
        pass


class LossLayer(object):

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

    def forward(self, x, targets):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropyLayer(LossLayer):

    def __init__(self, num_inputs):
        super(CrossEntropyLayer, self).__init__(num_inputs)

    def forward(self, yhat, y):
        self.yhat, self.y = yhat, y
        loss = - np.sum(y * np.log(yhat + 1e-6))
        return loss

    def backward(self):
        grad = - self.y / (self.yhat + 1e-6)
        return grad


class SoftmaxLayer(ComputeLayer):

    def __init__(self):
        super(SoftmaxLayer, self).__init__(None, None, neurons.Softmax)

    def parameters(self):
        return []

    def gradient(self):
        return []

    def forward(self, x):
        return self.neuron.forward(x)

    def backward(self, grad_out):
        return self.neuron.backward(grad_out)


class FullyConnectedLayer(ComputeLayer):

    def __init__(self, num_inputs, num_outputs, scale=1.0, neuron_type=neurons.Identity):
        super(FullyConnectedLayer, self).__init__(num_inputs, num_outputs, neuron_type)
        self.W = np.random.randn(num_inputs, num_outputs) * scale
        self.b = np.random.randn(num_outputs) * scale

    def parameters(self):
        return [self.W, self.b]

    def gradient(self):
        return [self.grad_W, self.grad_b]

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        outputs = self.neuron.forward(self.z)
        return outputs

    def backward(self, grad_out):
        grad_z = self.neuron.backward(grad_out)
        # This is basically a cross join between the last axes of x and z.
        # grad_W_i = np.einsum("ij,ik->ijk", self.x, grad_z)
        # grad_b_i = grad_z
        # self.grad_W = np.sum(grad_W_i, axis=0)
        # self.grad_b = np.sum(grad_b_i, axis=0)

        # x shape: (num_examples, num_inputs)
        # z shape: (num_examples, num_outputs)
        self.grad_W = np.einsum("ij,ik->jk", self.x, grad_z)
        self.grad_b = np.sum(grad_z, axis=0)
        # np.einsum("ik,jk->ij", grad_z, self.W)
        grad_in = np.dot(grad_z, np.transpose(self.W))
        return grad_in


class ConvolutionalLayer(ComputeLayer):

    def __init__(self, image_size, filter_size, stride=(1,1), pad=(0,0), num_filters=1,
            scale=1.0, neuron_type=neurons.Identity):
        self.image_size = image_size
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.num_filters = num_filters
        # check input and filter dims
        width_in, height_in, depth_in = image_size
        width_filter, height_filter, depth_filter = filter_size
        assert width_filter <= width_in
        assert height_filter <= height_in
        assert depth_in == depth_filter
        # get padded image size
        width_padded, height_padded, _ = common.get_padded_image_size(image_size, filter_size, stride, pad)
        # get output dims
        width_out = (width_padded - width_filter) // stride[0] + 1
        height_out = (height_padded - height_filter) // stride[1] + 1
        # call parent init
        num_inputs = width_in * height_in * depth_in
        num_outputs = width_out * height_out * num_filters
        super(ConvolutionalLayer, self).__init__(num_inputs, num_outputs, neuron_type)
        # create image to row indices
        self.row_indices = common.im2row_index(image_size, filter_size, stride, pad)
        self.num_rows, self.num_filter_inputs = self.row_indices.shape
        assert self.num_rows == width_out * height_out
        assert self.num_filter_inputs == np.prod(filter_size)
        # initialize weight and bias
        self.W = np.random.randn(self.num_filter_inputs, num_filters) * scale
        self.b = np.random.randn(num_filters) * scale

    def parameters(self):
        return [self.W, self.b]

    def gradient(self):
        return [self.grad_W, self.grad_b]

    def forward(self, x):
        num_samples, _ = x.shape
        # convert row vectors as images
        images = x.reshape((-1, *self.image_size))
        # np.pad() copies the original array multiple times:
        # https://stackoverflow.com/questions/40076280/how-is-numpy-pad-implemented-for-constant-value
        # (TODO) allocate the padded array first and create a view of the central part of the array to assign values
        pad_width = (0, *self.pad, 0)
        images_padded = np.pad(images, tuple(zip(pad_width, pad_width)), 'constant')
        # convert images back to row vectors
        x_padded = images_padded.reshape((num_samples, -1))
        # extract input rows for filters
        self.input_rows = x_padded[:, self.row_indices.reshape(-1)].reshape((-1, *self.row_indices.shape))
        # input_rows shape: (num_examples, num_rows, num_filter_inputs)
        # output_rows shape: (num_examples, num_rows, num_filters)
        output_rows = np.dot(self.input_rows, self.W) + self.b
        _, num_rows, num_filters = output_rows.shape
        assert num_rows * num_filters == self.num_outputs
        # convert rows to row vectors
        z = output_rows.reshape((-1, self.num_outputs))
        outputs = self.neuron.forward(z)
        return outputs

    def backward(self, grad_out):
        grad_out = grad_out.reshape(-1, self.num_rows, self.num_filters)
        grad_z = self.neuron.backward(grad_out)
        # input_rows shape: (num_examples, num_rows, num_filter_inputs)
        # grad_z shape: (num_examples, num_rows, num_filters)
        self.grad_W = np.einsum("ijk,ijl->kl", self.input_rows, grad_z)
        self.grad_b = np.sum(grad_z, axis=(0, 1))
        # grad_z shape: (num_examples, num_rows, num_filters)
        # W shape: num_filter_inputs, num_filters
        # grad_rows shape: (num_examples, num_rows, num_filter_inputs)
        grad_rows = np.dot(grad_z, np.transpose(self.W))
        # np.einsum("ijl,kl->ijk", grad_z, self.W)
        grad_image = common.row2im(grad_rows, self.row_indices, self.image_size, self.filter_size, self.stride, self.pad)
        return grad_image.reshape((-1, self.num_inputs))


class PoolingOperator(object):
    pass

class PoolingMax(PoolingOperator):

    def forward(self, input_rows):
        self.input_rows = input_rows
        self.max_elem_indices = np.argmax(input_rows, axis=2)
        axis0_size, axis1_size = input_rows.shape[:2]
        self.axis0_indices, self.axis1_indices = np.ix_(range(axis0_size), range(axis1_size))
        return input_rows[self.axis0_indices, self.axis1_indices, self.max_elem_indices]

    def backward(self, grad_out):
        grad_rows = np.zeros_like(self.input_rows)
        grad_rows[self.axis0_indices, self.axis1_indices, self.max_elem_indices] = grad_out
        return grad_rows


class PoolingAvg(PoolingOperator):

    def forward(self, input_rows):
        self.input_rows = input_rows
        return np.average(input_rows, axis=2)

    def backward(self, grad_out):
        grad_rows = np.zeros_like(self.input_rows)
        _, _, num_pooling_inputs = self.input_rows.shape
        grad_rows = grad_out[..., np.newaxis] / num_pooling_inputs
        return grad_rows


class PoolingLayer(ComputeLayer):

    def __init__(self, image_size, filter_size, stride=(1,1), pad=(0,0), pooling_op=PoolingMax):
        self.image_size = image_size
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.pooling_op = pooling_op()
        # check input and filter dims
        width_in, height_in, depth_in = image_size
        width_filter, height_filter, depth_filter = filter_size
        assert width_filter <= width_in
        assert height_filter <= height_in
        assert depth_in == depth_filter
        # get padded image size
        width_padded, height_padded, _ = common.get_padded_image_size(image_size, filter_size, stride, pad)
        # get output dims
        width_out = (width_padded - width_filter) // stride[0] + 1
        height_out = (height_padded - height_filter) // stride[1] + 1
        # call parent init
        num_inputs = width_in * height_in * depth_in
        num_outputs = width_out * height_out
        super(PoolingLayer, self).__init__(num_inputs, num_outputs, None)
        # create image to row indices
        self.row_indices = common.im2row_index(image_size, filter_size, stride, pad)
        self.num_rows, self.num_pooling_inputs = self.row_indices.shape
        assert self.num_rows == width_out * height_out
        assert self.num_pooling_inputs == np.prod(filter_size)

    def parameters(self):
        return []

    def gradient(self):
        return []

    def forward(self, x):
        num_samples, _ = x.shape
        # convert row vectors as images
        images = x.reshape((-1, *self.image_size))
        pad_width = (0, *self.pad, 0)
        images_padded = np.pad(images, tuple(zip(pad_width, pad_width)), 'constant')
        # convert images back to row vectors
        x_padded = images_padded.reshape((num_samples, -1))
        # extract input rows for filters
        self.input_rows = x_padded[:, self.row_indices.reshape(-1)].reshape((-1, *self.row_indices.shape))
        # input_rows shape: (num_examples, num_rows, num_pooling_inputs)
        # output_rows shape: (num_examples, num_rows)
        outputs = self.pooling_op.forward(self.input_rows)
        _, num_rows = outputs.shape
        assert num_rows == self.num_outputs
        return outputs

    def backward(self, grad_out):
        # grad_out shape: (num_examples, num_rows)
        grad_rows = self.pooling_op.backward(grad_out)
        grad_image = common.row2im(grad_rows, self.row_indices, self.image_size, self.filter_size, self.stride, self.pad)
        return grad_image.reshape((-1, self.num_inputs))


class IdentityLayer(ComputeLayer):

    def __init__(self, num_inputs, neuron_type=neurons.Identity):
        super(IdentityLayer, self).__init__(num_inputs, num_inputs, neuron_type)

    def forward(self, x):
        outputs = self.neuron.forward(x)
        return outputs

    def backward(self, grad_out):
        grad_in = self.neuron.backward(grad_out)
        return grad_in

    def parameters(self):
        return []

    def gradient(self):
        return []


class DropoutLayer(ComputeLayer):

    def __init__(self, num_inputs, drop_prob=.5):
        super(DropoutLayer, self).__init__(num_inputs, num_inputs, None)
        self.drop_prob = drop_prob

    @property
    def skip_eval(self):
        return True

    def forward(self, x):
        keep_prob = 1.0 - self.drop_prob
        self.mask = np.random.rand(*x.shape) < keep_prob
        outputs = x * self.mask / keep_prob
        return outputs

    def backward(self, grad_out):
        grad_in = grad_out * self.mask
        return grad_in

    def parameters(self):
        return []

    def gradient(self):
        return []
