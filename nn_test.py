import unittest
import numpy as np

import nn
import neurons
import dataset


class TestFullyConnectedLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.train_data = dataset.MNIST('data/3/train-images-idx3-ubyte.gz', 'data/3/train-labels-idx1-ubyte.gz')
        self.test_data = dataset.MNIST('data/3/t10k-images-idx3-ubyte.gz', 'data/3/t10k-labels-idx1-ubyte.gz')

    def test_forward_pass(self):
        num_inputs = self.train_data.num_inputs
        num_outputs = self.train_data.num_outputs

        data_layer = nn.BatchDataLayer(self.train_data.inputs, self.train_data.outputs, batch_size=64, shuffle=False)
        inputs, targets = next(data_layer.forward())

        fc_layer = nn.FullyConnectedLayer(num_inputs, num_outputs)
        outputs = fc_layer.forward(inputs)


class TestConvolutionalLayer(unittest.TestCase):

    def setUp(self):
        self.train_data = dataset.MNIST('data/3/train-images-idx3-ubyte.gz', 'data/3/train-labels-idx1-ubyte.gz')
        self.test_data = dataset.MNIST('data/3/t10k-images-idx3-ubyte.gz', 'data/3/t10k-labels-idx1-ubyte.gz')
        self.image_size = (self.train_data.image_width, self.train_data.image_height, 1)

    def test_preserve_image_size(self):
        filter_size = (3, 3, 1)
        stride = (1, 1)
        pad = (1, 1)

        data_layer = nn.BatchDataLayer(self.train_data.inputs, self.train_data.outputs, batch_size=64, shuffle=False)
        inputs, targets = next(data_layer.forward())

        conv_layer = nn.ConvolutionalLayer(self.image_size, filter_size, stride, pad)
        outputs = conv_layer.forward(inputs)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_fully_connected_as_special_case(self):
        filter_size = self.image_size
        stride = (1, 1)
        pad = (0, 0)
        num_filters = self.train_data.num_outputs

        # get a data batch
        data_layer = nn.BatchDataLayer(self.train_data.inputs, self.train_data.outputs, batch_size=64, shuffle=False)
        inputs, targets = next(data_layer.forward())

        # forward pass

        # get forward output of fully connected layer
        np.random.seed(0)
        fc_layer = nn.FullyConnectedLayer(self.train_data.num_inputs, self.train_data.num_outputs, scale=0.01)
        fc_outputs = fc_layer.forward(inputs)

        # get forward output of convolutional layer
        np.random.seed(0)
        conv_layer = nn.ConvolutionalLayer(self.image_size, filter_size, stride, pad, num_filters, scale=0.01)
        conv_outputs = conv_layer.forward(inputs)

        # compare forward outputs
        np.testing.assert_array_equal(fc_layer.W, conv_layer.W)
        np.testing.assert_array_equal(fc_layer.b, conv_layer.b)
        np.testing.assert_array_equal(inputs, conv_layer.input_rows.reshape(inputs.shape))
        # np.testing.assert_array_equal(fc_outputs, conv_outputs)
        np.testing.assert_array_almost_equal(fc_outputs, conv_outputs, decimal=10)

        # dot_prod1 = np.dot(conv_layer.input_rows.reshape(inputs.shape), conv_layer.W)
        # dot_prod2 = np.dot(conv_layer.input_rows, conv_layer.W)
        # out1 = dot_prod1 + conv_layer.b
        # out2 = (dot_prod2 + conv_layer.b).reshape(out1.shape)
        # print(dot_prod1.shape)
        # print(dot_prod2.shape)
        # print(dot_prod1)
        # print(dot_prod2)
        # print(np.sum(np.abs(dot_prod1 - dot_prod2)))
        # np.testing.assert_array_equal(dot_prod1, dot_prod2.reshape(dot_prod1.shape))
        # np.testing.assert_array_equal(fc_outputs, out1)
        # np.testing.assert_array_equal(fc_outputs, out2)

        # backward pass

        # get grad from loss layer
        loss_layer = nn.CrossEntropyLayer(self.train_data.num_outputs)
        loss_layer.forward(conv_outputs + 1.0, targets)
        grad_out = loss_layer.backward()

        # backward pass of fully connected layer
        fc_grad = fc_layer.backward(grad_out)
        # backward pass of convolutional layer
        conv_grad = conv_layer.backward(grad_out)
        # compare backward outputs
        np.testing.assert_array_equal(fc_grad, conv_grad)


class TestPoolingLayer(unittest.TestCase):

    def setUp(self):
        self.train_data = dataset.MNIST('data/3/train-images-idx3-ubyte.gz', 'data/3/train-labels-idx1-ubyte.gz')
        self.test_data = dataset.MNIST('data/3/t10k-images-idx3-ubyte.gz', 'data/3/t10k-labels-idx1-ubyte.gz')
        self.image_size = (self.train_data.image_width, self.train_data.image_height, 1)

    def test_identity_layer_as_special_case(self):
        filter_size = (1, 1, 1)
        stride = (1, 1)
        pad = (0, 0)

        data_layer = nn.BatchDataLayer(self.train_data.inputs, self.train_data.outputs, batch_size=64, shuffle=False)
        inputs, targets = next(data_layer.forward())

        for pooling_op in (nn.PoolingAvg, nn.PoolingMax):
            pool_layer = nn.PoolingLayer(self.image_size, filter_size, stride, pad, pooling_op)
            outputs = pool_layer.forward(inputs)
            np.testing.assert_array_equal(inputs, outputs)

            # get grad from loss layer
            loss_layer = nn.CrossEntropyLayer(pool_layer.num_outputs)
            loss_layer.forward(outputs + 1, outputs)
            grad_out = loss_layer.backward()

            grad_in = pool_layer.backward(grad_out)
            np.testing.assert_array_equal(grad_in, grad_out)


if __name__ == '__main__':
    unittest.main()