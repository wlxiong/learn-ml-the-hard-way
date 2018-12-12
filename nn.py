import numpy as np
import random
import common
import neurons


class NeuralNetwork(object):

    def __init__(self, data_layer, compute_layers, loss_layer):
        self.data_layer = data_layer
        self.compute_layers = compute_layers
        self.loss_layer = loss_layer
        self.initialize()

    def initialize(self):
        input_dim = self.data_layer.output_dim
        for i, layer in enumerate(self.compute_layers):
            output_dim = layer.initialize(input_dim)
            input_dim = output_dim

    def evaluate(self, inputs):
        outputs = inputs
        for layer in self.compute_layers:
            if not layer.skip_eval:
                outputs = layer.forward(inputs)
                if layer.neuron is not None:
                    outputs = layer.neuron.forward(outputs)
                inputs = outputs
        return outputs

    def train(self, learning_rate, num_batches=None, verbose=False):
        cumulative_loss = 0.0
        for i, (inputs, targets) in enumerate(self.data_layer.forward()):
            if num_batches is not None and i >= num_batches: break
            if verbose: print("Batch %d" % i)
            loss = self.forward(inputs, targets, verbose)
            self.backward(verbose)
            self.update_parameters(learning_rate)
            cumulative_loss += loss
        return cumulative_loss

    def forward(self, inputs, targets, verbose=False):
        if verbose: print("Forward pass")
        outputs = inputs
        for i, layer in enumerate(self.compute_layers):
            if verbose: print("Layer %d - %s" % (i, layer))
            try:
                layer._inputs = inputs
                outputs = layer.forward(inputs)
                if layer.neuron is not None:
                    outputs = layer.neuron.forward(outputs)
                layer._outputs = outputs
                inputs = outputs
            except:
                print("Exception raised in forward pass of compute layer %d - %s" % (i, layer))
                raise
        loss = self.loss_layer.forward(outputs, targets)
        return loss

    def backward(self, verbose=False):
        if verbose: print("Backward pass")
        grad_out = self.loss_layer.backward()
        grad_in = grad_out
        for i, layer in enumerate(reversed(self.compute_layers)):
            if verbose: print("Layer %d - %s" % (i, layer))
            try:
                if layer.neuron is not None:
                    grad_out = layer.neuron.backward(grad_out)
                grad_in = layer.backward(grad_out)
                grad_out = grad_in
            except:
                print("Exception raised in backward pass of compute layer %d - %s" % (i, layer))
                raise

    def update_parameters(self, learning_rate):
        for layer in self.compute_layers:
            params = layer.parameters()
            grad = layer.gradient()
            for p, g in zip(params, grad):
                p[:] = p - learning_rate * g


class ComputeLayer(object):

    def __init__(self, neuron_type):
        self._input_dim = None
        self._output_dim = None
        self._inputs = None
        self._outputs = None
        self._neuron = neuron_type() if neuron_type is not None else None

    def __str__(self):
        return "%s(%s, %s, %s)" % (type(self).__name__, self.input_dim, self.output_dim, self._neuron)

    __repr__ = __str__

    @property
    def skip_eval(self):
        return False

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def neuron(self):
        return self._neuron

    def initialize(self, input_dim):
        output_dim = self.init_parameters(input_dim)
        self._input_dim = input_dim
        self._output_dim = output_dim
        return output_dim

    def init_parameters(self, input_dim):
        raise NotImplementedError

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
        self.inputs = inputs
        self.targets = targets
        self._output_dim = self.inputs[0].shape

    @property
    def output_dim(self):
        return self._output_dim

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

    def __init__(self):
        pass

    def forward(self, x, targets):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropyLayer(LossLayer):

    def __init__(self):
        super(CrossEntropyLayer, self).__init__()

    def forward(self, yhat, y):
        self.yhat, self.y = yhat, y
        loss = - np.sum(y * np.log(yhat + 1e-6))
        return loss

    def backward(self):
        grad = - self.y / (self.yhat + 1e-6)
        return grad


class IdentityLayer(ComputeLayer):

    def __init__(self, neuron_type=neurons.Identity):
        super(IdentityLayer, self).__init__(neuron_type)

    def init_parameters(self, input_dim):
        return input_dim

    def forward(self, x):
        return x

    def backward(self, grad_out):
        return grad_out

    def parameters(self):
        return []

    def gradient(self):
        return []


class SoftmaxLayer(IdentityLayer):

    def __init__(self):
        super(SoftmaxLayer, self).__init__(neurons.Softmax)


class DropoutLayer(ComputeLayer):

    def __init__(self, drop_prob=.5):
        super(DropoutLayer, self).__init__(None)
        self.drop_prob = drop_prob

    @property
    def skip_eval(self):
        return True

    def init_parameters(self, input_dim):
        return input_dim

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


class FullyConnectedLayer(ComputeLayer):

    def __init__(self, num_neurons, scale=1.0, neuron_type=neurons.Identity):
        super(FullyConnectedLayer, self).__init__(neuron_type)
        self.num_neurons = num_neurons
        self.scale = scale
        self.W = None
        self.b = None

    def init_parameters(self, input_dim):
        num_inputs = np.prod(input_dim)
        self.W = np.random.randn(num_inputs, self.num_neurons) * self.scale
        self.b = np.random.randn(self.num_neurons) * self.scale
        return (self.num_neurons,)

    def parameters(self):
        return [self.W, self.b]

    def gradient(self):
        return [self.grad_W, self.grad_b]

    def forward(self, x):
        num_samples = x.shape[0]
        self.x = x.reshape((num_samples, -1))
        self.z = np.dot(self.x, self.W) + self.b
        return self.z

    def backward(self, grad_out):
        num_samples = grad_out.shape[0]
        grad_out = grad_out.reshape((num_samples, -1))
        # This is basically a cross join between the last axes of x and z.
        # grad_W_i = np.einsum("ij,ik->ijk", self.x, grad_z)
        # grad_b_i = grad_z
        # self.grad_W = np.sum(grad_W_i, axis=0)
        # self.grad_b = np.sum(grad_b_i, axis=0)

        # x shape: (num_samples, num_inputs)
        # z shape: (num_samples, num_outputs)
        self.grad_W = np.einsum("ij,ik->jk", self.x, grad_out)
        self.grad_b = np.sum(grad_out, axis=0)
        # np.einsum("ik,jk->ij", grad_out, self.W)
        grad_in = np.dot(grad_out, np.transpose(self.W))
        return grad_in.reshape((-1, *self.input_dim))


class ConvolutionalLayer(ComputeLayer):

    def __init__(self, filter_size, stride=(1,1), pad=(0,0), num_filters=1,
            scale=1.0, neuron_type=neurons.Identity):
        super(ConvolutionalLayer, self).__init__(neuron_type)
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.num_filters = num_filters
        self.scale = scale

    def init_parameters(self, input_dim):
        # check input and filter dims
        width_in, height_in, depth_in = input_dim
        width_filter, height_filter = self.filter_size
        self.filter_size = (width_filter, height_filter, depth_in)
        assert width_filter <= width_in
        assert height_filter <= height_in
        # get padded image size
        width_padded, height_padded, _ = common.get_padded_image_size(input_dim, self.filter_size, self.stride, self.pad)
        # get output dims
        width_out = (width_padded - width_filter) // self.stride[0] + 1
        height_out = (height_padded - height_filter) // self.stride[1] + 1
        depth_out = self.num_filters
        # create image to row indices
        self.row_indices = common.im2row_index(input_dim, self.filter_size, self.stride, self.pad)
        self.num_rows, self.num_filter_inputs = self.row_indices.shape
        assert self.num_rows == width_out * height_out
        assert self.num_filter_inputs == np.prod(self.filter_size)
        # initialize weight and bias
        self.W = np.random.randn(self.num_filter_inputs, self.num_filters) * self.scale
        self.b = np.random.randn(self.num_filters) * self.scale
        return (width_out, height_out, depth_out)

    def parameters(self):
        return [self.W, self.b]

    def gradient(self):
        return [self.grad_W, self.grad_b]

    def forward(self, x):
        num_samples = x.shape[0]
        # convert row vectors as images
        images = x.reshape((-1, *self.input_dim))
        # np.pad() copies the original array multiple times:
        # https://stackoverflow.com/questions/40076280/how-is-numpy-pad-implemented-for-constant-value
        # (TODO) allocate the padded array first and create a view of the central part of the array to assign values
        pad_width = (0, *self.pad, 0)
        images_padded = np.pad(images, tuple(zip(pad_width, pad_width)), 'constant')
        # convert images back to row vectors
        x_padded = images_padded.reshape((num_samples, -1))
        # extract input rows for filters
        self.input_rows = x_padded[:, self.row_indices.reshape(-1)].reshape((-1, *self.row_indices.shape))
        # input_rows shape: (num_samples, num_rows, num_filter_inputs)
        # output_rows shape: (num_samples, num_rows, num_filters)
        output_rows = np.dot(self.input_rows, self.W) + self.b
        # convert rows to row vectors
        outputs = output_rows.reshape((-1, *self.output_dim))
        return outputs

    def backward(self, grad_out):
        grad_out = grad_out.reshape(-1, self.num_rows, self.num_filters)
        # input_rows shape: (num_samples, num_rows, num_filter_inputs)
        # grad_z shape: (num_samples, num_rows, num_filters)
        self.grad_W = np.einsum("ijk,ijl->kl", self.input_rows, grad_out)
        self.grad_b = np.sum(grad_out, axis=(0, 1))
        # grad_z shape: (num_samples, num_rows, num_filters)
        # W shape: num_filter_inputs, num_filters
        # grad_rows shape: (num_samples, num_rows, num_filter_inputs)
        grad_rows = np.dot(grad_out, np.transpose(self.W))
        # np.einsum("ijl,kl->ijk", grad_z, self.W)
        grad_in = common.row2im(grad_rows, self.row_indices, self.input_dim, self.filter_size, self.stride, self.pad)
        assert grad_in.shape[1:] == self.input_dim
        return grad_in


class PoolingOperator(object):
    pass


class PoolingMax(PoolingOperator):

    def forward(self, input_rows):
        self.input_rows = input_rows
        self.max_elem_indices = np.argmax(input_rows, axis=2)
        num_samples, num_rows = input_rows.shape[:2]
        self.example_indices, self.row_indices = np.ix_(range(num_samples), range(num_rows))
        return input_rows[self.example_indices, self.row_indices, self.max_elem_indices]

    def backward(self, grad_out):
        grad_rows = np.zeros_like(self.input_rows)
        grad_rows[self.example_indices, self.row_indices, self.max_elem_indices] = grad_out
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

    def __init__(self, filter_size, stride=(1,1), pad=(0,0), pooling_op=PoolingMax):
        super(PoolingLayer, self).__init__(None)
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.pooling_op = pooling_op()

    def init_parameters(self, input_dim):
        # check input and filter dims
        width_in, height_in, depth_in = input_dim
        width_filter, height_filter = self.filter_size
        self.filter_size = (width_filter, height_filter, depth_in)
        assert width_filter <= width_in
        assert height_filter <= height_in
        # get padded image size
        self.filter_size = (width_filter, height_filter, 1)
        width_padded, height_padded, _ = common.get_padded_image_size(input_dim, self.filter_size, self.stride, self.pad)
        # get output dims
        width_out = (width_padded - width_filter) // self.stride[0] + 1
        height_out = (height_padded - height_filter) // self.stride[1] + 1
        depth_out = depth_in
        # create image to row indices
        self.row_indices = common.im2row_index(input_dim, self.filter_size, self.stride, self.pad)
        self.num_rows, self.num_pooling_inputs = self.row_indices.shape
        assert self.num_rows == width_out * height_out * depth_in
        assert self.num_pooling_inputs == np.prod(self.filter_size)
        return (width_out, height_out, depth_out)

    def parameters(self):
        return []

    def gradient(self):
        return []

    def forward(self, x):
        num_samples = x.shape[0]
        # convert row vectors as images
        images = x.reshape((-1, *self.input_dim))
        pad_width = (0, *self.pad, 0)
        images_padded = np.pad(images, tuple(zip(pad_width, pad_width)), 'constant')
        # convert images back to row vectors
        x_padded = images_padded.reshape((num_samples, -1))
        # extract input rows for filters
        self.input_rows = x_padded[:, self.row_indices.reshape(-1)].reshape((-1, *self.row_indices.shape))
        # input_rows shape: (num_samples, num_rows, num_pooling_inputs)
        # output_rows shape: (num_samples, num_rows)
        output_rows = self.pooling_op.forward(self.input_rows)
        outputs = output_rows.reshape((-1, *self.output_dim))
        return outputs

    def backward(self, grad_out):
        grad_out = grad_out.reshape((-1, self.num_rows))
        # grad_out shape: (num_samples, num_rows)
        grad_rows = self.pooling_op.backward(grad_out)
        grad_in = common.row2im(grad_rows, self.row_indices, self.input_dim, self.filter_size, self.stride, self.pad)
        assert grad_in.shape[1:] == self.input_dim
        return grad_in
