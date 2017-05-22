#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from npnet import *
from utils import *

if __name__ == '__main__':

    # Checking that feedforward works with random parameters:

    # npn = NPNet([1, 100, 1], ['sigmoid', 'linear'], 'gaussian', 1.0)
    # np.random.seed(73)
    # f = lambda x: np.power(x, 3)
    # x_, y_ = function_truth(-5, 5, 100, f)
    # x, y = sample_function(-5, 5, 30, f, 9)
    # plot_truth_and_guess(x_, y_, x, y, npn)

    # Checking that we are capable of calculating the gradients

    # np.random.seed(42)
    # npn = NPNet([1, 100, 100, 2], ['sigmoid', 'sigmoid', 'linear'], 'gaussian', 1.0)
    # y1, s1 = npn.predict(np.matrix(5))
    # E1 = kl_loss(np.matrix([1.0, 2.0]), np.matrix([5e-2, 5e-2]), y1, s1)
    # dkl_dy1, dkl_ds1 = dkl_loss(np.matrix([1.0, 2.0]).T, np.matrix([5e-2, 5e-2]).T, y1.T, s1.T)
    # npn.layers[-1].backpropagate(dkl_dy1, dkl_ds1)
    # gW_m, gW_s, gb_m, gb_s = npn.layers[1].gradients()
    # dydp1 = gW_s[20, 0]
    # npn.layers[1].W_s[20, 0] += 1e-6
    # y2, s2 = npn.predict(np.matrix(5))
    # E2 = kl_loss(np.matrix([1.0, 2.0]), np.matrix([5e-2, 5e-2]), y2, s2)
    # dydp2 = (E2 - E1)/1e-6
    # print("{} ~ {}? {}".format(dydp1, dydp2, abs(dydp1 - dydp2) < 1e-3))

    # Checking training (I)

    np.random.seed(42)
    npn = NPNet([1, 100, 1], ['sigmoid', 'linear'], 'gaussian', 1.0)
    np.random.seed(72)
    f = lambda x: np.power(x, 2)
    g = lambda x: np.power(x+5, 1.25)
    x, y = sample_function(-5, 5, 30,  f, g)
    npn.train(x, y, eta=1e-1, max_iterations=1000, sigma=0.0, lambda_s=1e-3,
              lambda_d=0.0, h=0.0, epsilon=1e-9, verbose=True)
    plt.show(plot_truth_and_guess(f, g, x, y, npn))



