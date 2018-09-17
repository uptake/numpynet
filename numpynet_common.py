"""
numpynet_common
Contains common code shared across different modules
@author: Chronocook (chronocook@gmail.com)
"""
import numpy as np
from loggit import log


class Activation:
    """
    A class to hold all of the activation functions, ensures all have derivatives
    """
    function = None
    available = ["tanh", "tanhp1", "bugged_sigmoid", "sigmoid", "relu", "softmax", "stablesoftmax"]

    def __init__(self, choice="sigmoid"):
        """
        :param choice: Which activation function you want, must be in self.available
        """
        if choice not in self.available:
            log.out.error("Choice of activation (" + choice + ") not available!")
            raise ValueError
        elif choice == "tanh":
            self.function = self._tanh
        elif choice == "tanhp1":
            self.function = self._tanhp1
        elif choice == "bugged_sigmoid":
            self.function = self._bugged_sigmoid
        elif choice == "sigmoid":
            self.function = self._sigmoid
        elif choice == "relu":
            self.function = self._relu
        elif choice == "softmax":
            self.function = self._softmax
        elif choice == "stablesoftmax":
            self.function = self._stablesoftmax

    @staticmethod
    def _tanh(x, deriv=False):
        """
        Hyperbolic tangent activation
        """
        if deriv:
            return 1.0 - np.power(np.tanh(x), 2)
        return np.tanh(x)

    @staticmethod
    def _tanhp1(x, deriv=False):
        """
        Hyperbolic tangent plus 1 activation
        """
        if deriv:
            return 1.0 - np.power(np.tanh(x), 2) / 2.0
        return (np.tanh(x) + 1.0) / 2.0

    @staticmethod
    def _bugged_sigmoid(x, deriv=False):
        """
        A function meant to be sigmoid and a bug i made for derivative
        TODO Why does this work? Going to leave it for now, because it's interesting
        """
        if deriv:
            return x * (1.0 - x)  # This is wrong.
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _sigmoid(x, deriv=False):
        """
        The sigmoid function and its derivative
        """
        y = 1.0 / (1.0 + np.exp(-x))
        if deriv:
            return y * (1.0 - y)
        return y

    @staticmethod
    def _relu(x, deriv=False):
        """
        Rectified linear unit activation function
        """
        if deriv:
            return 1.0 * (x > 0)
        return np.maximum(x, 0, x)

    @staticmethod
    # TODO: this should be right but doesn't seem to work :(  Check it again
    def _softmax(x, deriv=False):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        if deriv:
            return (x - np.max(x)) * e_x / e_x.sum(axis=0)
        return e_x / e_x.sum(axis=0)

    @staticmethod
    # TODO: this needs to be completed
    def _stablesoftmax(x, deriv=False):
        """
        Compute the softmax of vector x in a numerically stable way.
        """
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)


def predict_2d_space(net, delta=0.05):
    """
    Iterate predictions over a 2d space
    :param net: (object) A NumpyNet model object
    :param delta: space between predictions
    :return: prediction_matrix: the actual predictions
             axis_x and axis_y: the axes (useful for plotting)
    """
    axis_x = np.arange(net.predict_space[0], net.predict_space[1] + delta, delta)
    axis_y = np.arange(net.predict_space[2], net.predict_space[3] + delta, delta)
    prediction_matrix = np.empty((len(axis_x), len(axis_y)))
    for i, x in enumerate(axis_x):
        for j, y in enumerate(axis_y):
            test_prediction = np.array([x, y])
            test_prediction = net.predict(test_prediction)
            prediction_matrix[i, j] = test_prediction
    return prediction_matrix, axis_x, axis_y


def activation_test():
    """
    This is mostly for dev, it just plots the activations functions
    """
    import numpynet_visualize as nnviz
    x_vals = np.linspace(-10.0, 10.0, 100)
    x = x_vals

    activation_test = 1.0 / (1.0 + np.exp(-x_vals))
    dactivation_test = activation_test * (1.0 - activation_test)
    nnviz.plot_func(x_vals, activation_test, title="sigmoid")
    nnviz.plot_func(x_vals, dactivation_test, title="derivative sigmoid")

    activation_test = 1.0 / (1.0 + np.exp(-x_vals))
    dactivation_test = x_vals * (1.0 - x_vals)
    nnviz.plot_func(x_vals, activation_test, title="magic sigmoid")
    nnviz.plot_func(x_vals, dactivation_test, title="derivative magic sigmoid")

    activation_test = np.tanh(x_vals)
    dactivation_test = 1.0 - np.power(np.tanh(x_vals), 2)
    nnviz.plot_func(x_vals, activation_test, title="tanh")
    nnviz.plot_func(x_vals, dactivation_test, title="derivative tanh")

    e_x = np.exp(x - np.max(x))
    activation_test = e_x / e_x.sum(axis=0)
    dactivation_test = -(x - np.max(x)) * e_x / e_x.sum(axis=0)
    nnviz.plot_func(x_vals, activation_test, title="softmax")
    nnviz.plot_func(x_vals, dactivation_test, title="derivative softmax")

    # TODO Why does this break the plots?
    activation_test = np.maximum(x_vals, 0, x_vals)
    dactivation_test = 1.0 * (x_vals > 0)
    nnviz.plot_func(x_vals, activation_test, title="relU")
    nnviz.plot_func(x_vals, dactivation_test, title="derivative relU")



