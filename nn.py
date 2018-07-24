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

    def __init__(self, num_inputs, num_outputs, neuron):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neuron = neuron

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
        loss = - np.sum(y * np.log(yhat+1e-6))
        return loss

    def backward(self):
        grad = - self.y / (self.yhat + 1e-6)
        return grad


class FullyConnectedLayer(ComputeLayer):

    def __init__(self, num_inputs, num_outputs, scale=1.0, neuron_type=neurons.Identity):
        super(FullyConnectedLayer, self).__init__(num_inputs, num_outputs, neuron_type)
        self.W = np.random.randn(num_inputs, num_outputs) * scale
        self.b = np.random.randn(num_outputs) * scale
        self.neuron = neuron_type()

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
        grad_W_i = np.einsum("ij,ik->ijk", self.x, grad_z)
        grad_b_i = grad_z
        self.grad_W = np.sum(grad_W_i, axis=0)
        self.grad_b = np.sum(grad_b_i, axis=0)
        grad_in = np.dot(grad_z, np.transpose(self.W))
        return grad_in


class IdentityLayer(ComputeLayer):

    def __init__(self, num_inputs, neuron_type=neurons.Identity):
        super(IdentityLayer, self).__init__(num_inputs, num_inputs, neuron_type)
        self.neuron = neuron_type()

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
