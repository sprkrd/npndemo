# Author: Alejandro Suarez Hernandez
# The following code is an implementation of the Natural Parameter Networks
# proposed by H. Wang et al [1]. As for now, it only supports Gaussian
# distributions and sigmoid activation functions.

import numpy as np

alpha = 4 - 2*np.sqrt(2)
beta = -np.log(np.sqrt(2) + 1)
zeta_sq = np.pi/8.0


def kappa(x):
    return 1.0 / np.sqrt(1.0 + zeta_sq*x)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def hadamard(*args):
    product = np.multiply(args[0], args[1])
    for mat in args[2:]:
        np.multiply(product, mat, product)
    return product


def kl_loss(m1, s1, m2, s2):
    assert m1.shape == s1.shape == m2.shape == s2.shape
    return 0.5*np.sum((hadamard(m2 - m1, m2 - m1) + s1)/s2 - 1.0 + np.log(s2) - np.log(s1))


def dkl_loss(m1, s1, m2, s2):
    assert m1.shape == s1.shape == m2.shape == s2.shape
    A = (m2 - m1)/s2
    B = s1/hadamard(s2, s2)
    C = 1.0/s2
    dkl_dm2 = A
    dkl_ds2 = -0.5*(hadamard(A, A) + B - C)
    return np.sum(dkl_dm2, 1), np.sum(dkl_ds2, 1)


class Layer:

    def __init__(self, size, distribution='gaussian', transform='sigmoid'):
        assert  distribution == 'gaussian',\
                "Non supported distribution: {}".format(distribution)
        assert  transform in ('linear', 'sigmoid'),\
                "Non supported transform: {}".format(transform)
        self.size = size
        self.distribution = distribution
        self.transform = transform
        self.a_m = None
        self.a_s = None
        self.previous = None
        self.next_ = None

    def set_previous_layer(self, layer):
        self.previous = layer

    def set_next_layer(self, layer):
        self.next_ = layer

    def feedforward(self, a_m_in, a_s_in):
        assert False, "Not implemented"

    def backpropagate(self, error_m_in, error_s_in):
        assert False, "Not implemented"


class ComputationLayer(Layer):

    def __init__(self, size, distribution='gaussian', transform='sigmoid',
                 std0=0.0):
        super(ComputationLayer, self).__init__(size, distribution, transform)
        self.std0 = std0
        self.W_m = None
        self.W_s = None
        self.b_m = np.matrix(np.random.normal(0, std0, (size, 1)))
        self.b_s = np.matrix(np.random.exponential(std0, (size, 1)))
        self.o_m = None
        self.o_s = None
        self.delta_m = None
        self.delta_s = None
        self.previous = None
        self.next_ = None

    def set_previous_layer(self, layer):
        self.previous = layer
        shape = (self.size, layer.size)
        self.W_m = np.matrix(np.random.normal(0, self.std0, shape))
        self.W_s = np.matrix(np.random.exponential(self.std0, shape))

    def feedforward(self, a_m_in, a_s_in):
        assert a_m_in.shape == (self.W_m.shape[1], 1)
        assert a_s_in.shape == (self.W_s.shape[1], 1)
        # Linear transform
        self.o_m = self.W_m*a_m_in + self.b_m
        self.o_s = self.b_s + self.W_s*a_s_in + \
                   hadamard(self.W_m, self.W_m)*a_s_in + \
                   self.W_s*hadamard(a_m_in, a_m_in)
        # Non-linear transform (as for now, only sigmoid)
        if self.transform == 'sigmoid':
            kappa_1 = kappa(self.o_s)
            kappa_2 = kappa(alpha*alpha*self.o_s)
            self.a_m = sigmoid(hadamard(self.o_m, kappa_1))
            self.a_s = sigmoid(hadamard(alpha*(self.o_m + beta), kappa_2)) - \
                       hadamard(self.a_m, self.a_m)
        else: # self.transform == 'linear':
            self.a_m = self.o_m
            self.a_s = self.o_s
        # self.a_s = np.maximum(0, self.a_s)
        # Propagate to next layer, if applicable
        if self.next_ is not None:
            self.next_.feedforward(self.a_m, self.a_s)

    def backpropagate(self, error_m_in, error_s_in):
        # sanity check
        assert error_m_in.shape[0] == self.size
        assert error_s_in.shape[0] == self.size

        if self.transform == 'sigmoid':
            # pre-calculate some recurrent vectors:
            kappa_1 = kappa(self.o_s)
            kappa_2 = kappa(alpha*alpha*self.o_s)
            dsigmoid_am = hadamard(self.a_m, 1.0 - self.a_m)
            sigmoid_as = sigmoid(alpha*hadamard(self.o_m + beta, kappa_2))
            dsigmoid_as = hadamard(sigmoid_as, 1.0 - sigmoid_as)

            # Calculating derivatives of a_m, a_s w.r.t. o_m, o_s
            dam_dom = hadamard(dsigmoid_am, kappa_1)
            das_dom = alpha*hadamard(dsigmoid_as, kappa_2) - \
                      2*hadamard(self.a_m, dam_dom)
            dam_dos = -0.5*zeta_sq*hadamard(dsigmoid_am, self.o_m,
                                            np.power(kappa_1, 3))
            das_dos = -0.5*zeta_sq*alpha**3*hadamard(dsigmoid_as,
                                                     self.o_m + beta,
                                                     np.power(kappa_2, 3)) - \
                      2*hadamard(self.a_m, dam_dos)
        else: # self.transform == 'linear'
            # identity function, dam_dom = das_dos = 1; das_dom = dam_dos = 0
            dam_dom = np.matrix(np.ones((self.size, 1)))
            das_dom = np.matrix(np.zeros((self.size, 1)))
            dam_dos = np.matrix(np.zeros((self.size, 1)))
            das_dos = np.matrix(np.ones((self.size, 1)))

        # das_dom = hadamard(das_dom, self.a_s > 0)
        # das_dos = hadamard(das_dos, self.a_s > 0)

        # Calculate deltas (derivatives of the error w.r.t o_m and o_s
        self.delta_m = hadamard(error_m_in, dam_dom) + \
                       hadamard(error_s_in, das_dom)
        self.delta_s = hadamard(error_m_in, dam_dos) + \
                       hadamard(error_s_in, das_dos)

        if not isinstance(self.previous, InputLayer):
            # (Back)propagate only if the previous layer is not the input
            error_m_out = self.W_m.T*self.delta_m + 2*hadamard(
                    self.W_s.T*self.delta_s, self.previous.a_m)
            error_s_out = self.W_s.T*self.delta_s + \
                          hadamard(self.W_m.T, self.W_m.T)*self.delta_s

            self.previous.backpropagate(error_m_out, error_s_out)

    def gradients(self, lambda_s=0.0, lambda_d=0.0, h=0.0):
        gradient_W_m = self.delta_m*self.previous.a_m.T + \
                       2*hadamard(self.W_m, self.delta_s*self.previous.a_s.T)
        gradient_W_m += lambda_s*self.W_m
        gradient_W_s = self.delta_s*self.previous.a_s.T + \
                       self.delta_s*hadamard(self.previous.a_m.T,
                                             self.previous.a_m.T)
        gradient_W_s += lambda_s - lambda_d*(self.W_s - h)
        gradient_b_m = self.delta_m
        gradient_b_s = self.delta_s
        return gradient_W_m, gradient_W_s, gradient_b_m, gradient_b_s
 

class InputLayer(Layer):

    def __init__(self, size, distribution='gaussian'):
        super(InputLayer, self).__init__(size, distribution, 'linear')

    def set_previous_layer(self, layer):
        assert False, "Input layers cannot have previous layers"

    def feedforward(self, a_m_in, a_s_in=None):
        assert a_m_in.shape == (self.size, 1)
        if a_s_in is None:
            a_s_in = np.zeros((self.size, 1))
        else:
            assert a_s_in.shape == (self.size, 1)
        self.a_m = a_m_in
        self.a_s = a_s_in
        self.next_.feedforward(self.a_m, self.a_s)

    def backpropagate(self, *args):
        assert False, "Input layer: not able to propagate"


class NPNet:

    def __init__(self, architecture, transforms, distribution, std0=0.0):
        self.layers = [InputLayer(architecture[0], distribution)]
        self.n_parameters = 0.0
        for size, transform in zip(architecture[1:], transforms):
            layer = ComputationLayer(size, distribution, transform, std0)
            self.layers.append(layer)
            self.layers[-1].set_previous_layer(self.layers[-2])
            self.layers[-2].set_next_layer(self.layers[-1])
            self.n_parameters += 2*layer.W_m.size + 2*layer.b_m.size

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x, sigma=0):
        assert x.shape[1] == self.layers[0].size
        y_m = np.matrix(np.zeros((x.shape[0], self.layers[-1].size)))
        y_s = y_m.copy()
        a_s_in = np.repeat(np.matrix(sigma), x.shape[1], 0)
        for idx, row in enumerate(x):
            self.layers[0].feedforward(row.T, a_s_in)
            y_m[idx, :] = self.layers[-1].a_m.T
            y_s[idx, :] = self.layers[-1].a_s.T
        return y_m, y_s

    def train(self, x, y, sigma=0.0, epsilon=1e-9, eta=1e-1, eta_stop=1e-6,
              max_iterations=1000, lambda_s=0.0, lambda_d=0.0, h=0.0,
              verbose=False):
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.layers[0].size
        assert y.shape[1] == self.layers[-1].size

        x_s = sigma*np.matrix(np.ones(x.shape))
        y_s = epsilon*np.matrix(np.ones(y.shape))

        guess_m, guess_s = self.predict(x)
        error = kl_loss(y, y_s, guess_m, guess_s)
        error_avg = np.sum(error)/x.shape[0]

        if verbose: print("Initial error avg: {}".format(error_avg))

        for itx in range(max_iterations):
            assert error_avg == error_avg # check non-nan
            gW_m = [np.zeros(layer.W_m.shape) for layer in self.layers[1:]]
            gW_s = [np.zeros(layer.W_s.shape) for layer in self.layers[1:]]
            gb_m = [np.zeros(layer.b_m.shape) for layer in self.layers[1:]]
            gb_s = [np.zeros(layer.b_s.shape) for layer in self.layers[1:]]
            error_total = 0.0
            for idx in range(x.shape[0]):
                target_m = y[idx, :]
                target_s = y_s[idx, :]
                self.layers[0].feedforward(x[idx, :].T, x_s[idx, :].T)
                guess_m = self.layers[-1].a_m
                guess_s = self.layers[-1].a_s
                error_total += kl_loss(target_m.T, target_s.T, guess_m, guess_s)
                dkl_dm2, dkl_ds2 = dkl_loss(target_m.T, target_s.T, guess_m, guess_s)
                self.layers[-1].backpropagate(dkl_dm2, dkl_ds2)
                for n_layer, layer in enumerate(self.layers[1:]):
                    gW_m_, gW_s_, gb_m_, gb_s_ = layer.gradients(lambda_s,
                                                                 lambda_d, h)
                    gW_m[n_layer] += gW_m_
                    gW_s[n_layer] += gW_s_
                    gb_m[n_layer] += gb_m_
                    gb_s[n_layer] += gb_s_
            for n_layer, layer in enumerate(self.layers[1:]):
                layer.W_m -= eta*gW_m[n_layer]
                layer.W_s = np.maximum(layer.W_s - eta*gW_s[n_layer], 0)
                layer.b_m -= eta*gb_m[n_layer]
                layer.b_s = np.maximum(layer.b_s - eta*gb_s[n_layer], 0)
            error_avg_ = error_total / x.shape[0]
            if (error_avg_ - error_avg) > -1e-4: eta *= 0.95
            error_avg = error_avg_
            if verbose:
                print("Iteration: {}, error_avg: {}, eta: {}".format(
                    itx, error_avg, eta))
            if eta < eta_stop: break
        if verbose: print("error_avg: {}, iterations: {}".format(error_avg, itx))

# [1] Wang, H., Shi, X., & Yeung, D.-Y. (2016). Natural-Parameter Networks:
#     A Class of Probabilistic Neural Networks, (1), 1–9. 
#     Retrieved from http://arxiv.org/abs/1611.00448 

