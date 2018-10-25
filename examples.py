"""
numpynet_examples
Contains examples and demos for numpynet
@author: Chronocook (chronocook@gmail.com)
"""

import time
import numpy as np
from numpynet.model import NumpyNet
import numpynet.common as common
from numpynet.visualize import NumpynetVizClient
from numpynet.loggit import log
from visdom import Visdom

"""
# TODO
Add epochs (randomly subsample instead of all stuff)
weight decay is STRANGE
Activation functions are GARBAGE!!
dropout DOESN'T EXIST
Lol remember this is for fun
"""


def make_checkerboard_training_set(num_points=0, noise=0.0, randomize=True,
                                   x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    """
    Makes a binary array like a checkerboard (to work on an xor like problem)
    :param num_points: (int) The number of points you want in your training set
    :param noise: (float) percent to bit-flip in the training data, allows it to be imperfect
    :param randomize: (bool) True if you want the locations to be random, False if you want an ordered grid
    :param x_min: (float) minimum x of the 2D domain
    :param x_max: (float) maximum x of the 2D domain
    :param y_min: (float) minimum y of the 2D domain
    :param y_max: (float) maximum y of the 2D domain
    :return:
    """
    log.out.info("Generating target data.")
    # Select coordinates to do an XOR like operation on
    coords = []
    bools = []
    if randomize:
        for i in range(num_points):
            # Add num_points randomly
            coord_point = np.random.random(2)
            coord_point[0] = coord_point[0] * (x_max - x_min) + x_min
            coord_point[1] = coord_point[1] * (y_max - y_min) + y_min
            coords.append(coord_point)
    else:
        x_points = np.linspace(x_min, x_max, int(np.sqrt(num_points)))
        y_points = np.linspace(y_min, y_max, int(np.sqrt(num_points)))
        for i in range(int(np.sqrt(num_points))):
            for j in range(int(np.sqrt(num_points))):
                # Add num_points randomly
                coord_point = [x_points[i], y_points[j]]
                coords.append(coord_point)
    # Assign an xor boolean value to the coordinates
    for coord_point in coords:
        bool_point = np.array([np.round(coord_point[0]) % 2, np.round(coord_point[1]) % 2]).astype(bool)
        bools.append(np.logical_xor(bool_point[0], bool_point[1]))
    # If noisy then bit flip
    if noise > 0.0:
        for i in enumerate(bools):
            if np.random.random() < noise:
                bools[i] = np.logical_not(bools[i])
    # Build training vectors
    train_in = None
    train_out = None
    for i, coord in enumerate(coords):
        # Need to initialize the arrays
        if i == 0:
            train_in = np.array([coord])
            train_out = np.array([[bools[i]]])
        else:
            train_in = np.append(train_in, np.array([coord]), axis=0)
            train_out = np.append(train_out,  np.array([[bools[i]]]), axis=1)

    train_out = train_out.T
    return train_in, train_out


def make_smiley_training_set(num_points=0, delta=0.05):
    """
    Makes a binary array that looks like a smiley face (for fun, challenging problem)
    :param num_points: (int) The number of points you want in your training set
    :param delta:
    :return:
    """
    log.out.info("Generating happy data.")
    # Select coordinates to do an XOR like operation on
    coords = []
    bools = []
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0
    for i in range(num_points):
        # Add num_points randomly
        coord_point = np.random.random(2)
        coord_point[0] = coord_point[0] * (x_max - x_min) + x_min
        coord_point[1] = coord_point[1] * (y_max - y_min) + y_min
        coords.append(coord_point)

    # Assign an xor boolean value to the coordinates
    for coord_point in coords:
        x = coord_point[0]
        y = coord_point[1]
        if (abs(x - 0.65) < delta) & (abs(y - 0.65) < (0.05+delta)):
            bools.append(True)
        elif (abs(x - 0.35) < delta) & (abs(y - 0.65) < (0.05+delta)):
            bools.append(True)
        elif ((x > 0.2) & (x < 0.8) &
              (abs(y - ((1.5 * (x - 0.5))**2 + 0.25)) < delta)):
            bools.append(True)
        else:
            bools.append(False)

    # Build training vectors
    train_in = None
    train_out = None
    for i, coord in enumerate(coords):
        # Need to initialize the arrays
        if i == 0:
            train_in = np.array([coord])
            train_out = np.array([[bools[i]]])
        else:
            train_in = np.append(train_in, np.array([coord]), axis=0)
            train_out = np.append(train_out,  np.array([[bools[i]]]), axis=1)

    train_out = train_out.T
    return train_in, train_out


def complete_a_picture(viz_client):
    """
    Here we give the net some random points (1 or 0) from a function and it tries to fill the
    rest of the space with what it thinks it should be

    :param viz_client: An instance of NumpynetVizClient
    """
    # Get a training set for a set of x-y coordinates, this one is part of a checkerboard pattern
    x_min = 0.0; x_max = 2.0; y_min = 0; y_max = 1.0
    train_in, train_out = make_checkerboard_training_set(num_points=1000, noise=0.00, randomize=True,
                                                         x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # Plot he training set
    viz_client.plot_2d_classes(train_in, train_out, title="Training data",
                               x_min=x_min, x_max=x_max,
                               y_min=y_min, y_max=y_max, delta=0.01)

    training_size = train_in.shape[0]
    batch_size = round(training_size / 3.0)
    num_features = train_in.shape[1]

    # Initialize a numpynet object
    numpy_net = NumpyNet(num_features, batch_size,
                         num_hidden=5, hidden_sizes=[4, 8, 16, 8, 4],
                         activation=["tanh", "tanh", "tanh", "tanh", "tanh", "tanh"],
                         learning_rate=0.0001,
                         dropout_rate=None, weight_decay=None,
                         random_seed=1337)
    # Hook the object up to the viz client
    numpy_net.set_viz_client(viz_client)

    # A basic report of the net to the logs
    numpy_net.report_model()

    # Train the model!
    numpy_net.train(train_in, train_out, epochs=10000,
                    visualize=True, visualize_percent=1, save_best="./numpynet_best_model.pickle",
                    debug_visualize=True)

    # A silly viz of the network architecture (if the net isn't too huge to make it muddled)
    if max(numpy_net.layer_sizes) <= 16:
        numpy_net.viz.network_svg(numpy_net)


def plot_activations():
    for activation in common.Activation.available:
        x = np.linspace(-10.0, 10.0, 100)
        y = common.Activation(activation).function(x, deriv=False)
        dy = common.Activation(activation).function(x, deriv=True)
        viz_client.plot_func(x, y, title=activation)
        viz_client.plot_func(x, dy, title="d_"+activation)


# TODO write this!
def paint_a_picture():
    """
    Here we give the net many examples of how to paint something like a square based on
    coordinates and try to get it to learn how to make that shape given input coords
    """
    # Make a training set (many random i,j coord and an x by y box around that coord to start with)
    # Throw it into the net
    # Test how it does for some random coordinate inputs
    pass


def load_a_model(filename, viz_client):
    my_net = NumpyNet.load(filename)
    prediction_matrix, axis_x, axis_y = common.predict_2d_space(my_net, delta=0.002)
    viz_client.plot_2d_prediction(prediction_matrix, axis_x, axis_y, title="Best Prediction")


if __name__ == '__main__':
    """
    Main driver.
    """
    # Set the logging level to normal and start run
    start_time = time.time()

    # Set up visdom configuration and kick up a connection
    viz = Visdom()
    viz.close()

    # Set up NumpynetVizClient
    viz_client = NumpynetVizClient(viz=viz)

    complete_a_picture(viz_client)

    # load_a_model("./numpynet_best_model.pickle", viz_client)

    # Shut down and clean up
    total_time = round((time.time() - start_time), 0)
    log.out.info("Execution time: " + str(total_time) + " sec")
    log.out.info("All Done!")
    log.stopLog()
