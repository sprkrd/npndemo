import numpy as np
import matplotlib.pyplot as plt

from npnet import *


def function_truth(a, b, size, f):
    x = np.matrix(np.linspace(a, b, size)).T
    y = f(x)
    return x, y


def sample_function(a, b, size, f, noise_std):
    x = np.matrix(np.random.uniform(a, b, (size, 1)))
    y = f(x) + np.random.normal(0, noise_std(x), (size, 1))
    return x, y


def plot_truth_and_guess(f, g, x, y, npn):
    x_, y_ = function_truth(-5, 5, 100, f)
    y_dev_ = g(x_)

    y_h_ = y_ + 3*y_dev_
    y_l_ = y_ - 3*y_dev_
    
    guess_y_m, guess_y_s = npn.predict(x_)

    guess_std = np.sqrt(guess_y_s)

    guess_low = guess_y_m - 3*guess_std
    guess_high = guess_y_m + 3*guess_std

    fig, ax = plt.subplots(1)
    ax.fill_between(x_.flat, guess_high.flat, guess_low.flat, facecolor='red', alpha=0.25)
    ax.fill_between(x_.flat, y_h_.flat, y_l_.flat, facecolor='blue', alpha=0.25)
    ax.plot(x_.flat, y_.flat, label='Ground truth', color='blue')
    ax.plot(x_.flat, guess_y_m.flat, label='Guessed function', color='red')
    ax.scatter(x.flat, y.flat, label='Noisy point', color='green')
    ax.set_title('Ground truth and guess with regions of uncertainty')
    ax.legend(loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()

    return fig

