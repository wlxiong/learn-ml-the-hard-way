from narray import lib as xp


class Neuron(object):

    def __init__(self):
        pass

    def __str__(self):
        return "<%s>" % (type(self).__name__,)

    def forward(self, z):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


class Identity(Neuron):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, z):
        return z

    def backward(self, grad_out):
        return grad_out


class Softmax(Neuron):

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, z):
        self.z = z
        z_min = xp.min(z, axis=1)
        z = z - z_min[:, xp.newaxis]
        exp = xp.exp(z)
        sum_exp = xp.sum(exp, axis=1)[:, xp.newaxis]
        outputs = exp / sum_exp
        return outputs

    def backward(self, grad_out):
        # diag
        # D(softmax) / dz_i
        # = D(exp(z_i) * sum(exp(z))^-1) / dz_i
        # = D(exp(z_i)) / dz * sum(exp(z))^-1 + exp(z_i) * D(sum(exp(z))^-1) / dz_i
        # = exp(z_i) * sum(exp(z))^-1 + exp(z_i) * -1 * sum(exp(z))^-2 * exp(z_i)
        # = exp(z_i) / sum(exp(z)) - exp(z_i)^2 / sum(exp(z))^2

        # off diag
        # D(softmax) / dz_j
        # = D(exp(z_i) * sum(exp(z))^-1) / dz_j
        # = exp(z_i) * D(sum(exp(z))^-1) / dz_j)
        # = exp(z_i) * -1 * sum(exp(z))^-2 * exp(z_j)
        # = - exp(z_i) * exp(z_j) / sum(exp(z))^2

        z_min = xp.min(self.z, axis=1)
        z = self.z - z_min[:, xp.newaxis]
        exp = xp.exp(z)
        sum_exp = xp.sum(exp, axis=1)
        outer_mat = - xp.einsum("ij,ik->ijk", exp, exp)
        outer_mat /= sum_exp[:, xp.newaxis, xp.newaxis]**2
        diag = exp / sum_exp[:, xp.newaxis]
        diag_idx = xp.arange(diag.shape[1])
        outer_mat[:, diag_idx, diag_idx] += diag
        grad_in = xp.einsum("ik,ikj->ij", grad_out, outer_mat)
        # Directly use matmul() instead of einsum
        # grad_in = xp.squeeze(xp.matmul(grad_out[:, xp.newaxis, :], outer_mat))
        return grad_in


class ReLU(Neuron):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, z):
        self.outputs = xp.maximum(z, 0)
        return self.outputs

    def backward(self, grad_out):
        sign = xp.sign(self.outputs)
        grad_in = sign * grad_out
        return grad_in
