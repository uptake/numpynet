"""
numpynet_common
Contains common code shared across different modules
@author: Chronocook (chronocook@gmail.com)
"""
import numpy as np
from numpynet.loggit import log


class Activation:
    """
    A class to hold all of the activation functions, ensures all have derivatives
    """
    function = None
    available = ["tanh", "_tanhpos", "sigmoid", "relu"]

    def __init__(self, choice="sigmoid"):
        """
        :param choice: Which activation function you want, must be in self.available
        """
        if choice not in self.available:
            msg = "Choice of activation (" + choice + ") not available!"
            log.out.error(msg)
            raise ValueError(msg)
        elif choice == "tanh":
            self.function = self._tanh
        elif choice == "_tanhpos":
            self.function = self._tanhpos
        elif choice == "sigmoid":
            self.function = self._sigmoid
        elif choice == "relu":
            self.function = self._relu

    @staticmethod
    def _tanh(x, deriv=False):
        """
        Hyperbolic tangent activation
        """
        if deriv:
            return 1.0 - np.power(np.tanh(x), 2)
        return np.tanh(x)

    @staticmethod
    def _tanhpos(x, deriv=False):
        """
        Positive hyperbolic tangent activation
        """
        if deriv:
            return (1.0 - np.power(np.tanh(x), 2)) / 2.0
        return (np.tanh(x) + 1.0) / 2.0

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
        return np.maximum(x, 0)


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
