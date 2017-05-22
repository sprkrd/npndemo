# npndemo

Python implementation of the Natural Parameter Networks proposed by Wang and
others [1]. As for now this implementation is just for demonstration purposes
and only the Gaussian distribution and the sigmoid non-linear activation
function have been implemented. The Jupyter notebook is a slideshow intended
for a presentation in a Probabilistic Graphical Models course.

This implementation trains the networks with gradient descent using the
KL loss. It is functional but not very efficient. It accepts regularization,
imposing a N(0, lambda_s^-1) prior to the weights and including the KL
difference in the overal cost.

## Contents:

- npnet.py: contains all the classes and methods that implement NPN.
- npnet_test.py: contains a few non-exhaustive tests to assess if the
feedforward, backpropagation and training algorithms work
- Presentation.{ipynb,html}: presentation slides. It is recommended to
visualize the ipynb using `jupyter-nbconverter Presentation.ipynb --to slides --post serve`
- img/\*: slide images

## References

[1] Wang, H., Shi, X., & Yeung, D.-Y. (2016). Natural-Parameter Networks: A
Class of Probabilistic Neural Networks, (1), 1–9. Retrieved from http://arxiv.org/abs/1611.00448


