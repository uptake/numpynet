"""
NumpyNet!

@author: Brad Beechler (brad.e.beechler@gmail.com)
"""

import time
import numpy as np
from numpynet_model import NumpyNet
import numpynet_visualize as nnviz
from loggit import log

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
    log.out.info("Generating happy data.")
    # Select coordinates to do an XOR like operation on
    coords = []
    bools = []
    # eye: x=0.6 {0.6<y<0.7}
    # eye: x=0.4 {0.6<y<0.7}
    # smile: (1.5*(x-0.5))**2 + 0.25 {0.25<x<0.75}
    x_min = 0.0; x_max = 1.0; y_min = 0.0; y_max = 1.0
    for i in range(num_points):
        # Add num_points randomly
        coord_point = np.random.random(2)
        coord_point[0] = coord_point[0] * (x_max - x_min) + x_min
        coord_point[1] = coord_point[1] * (y_max - y_min) + y_min
        coords.append(coord_point)

    # Assign an xor boolean value to the coordinates

    for coord_point in coords:
        x = coord_point[1]
        y = coord_point[0]
        if (abs(x - 0.65) < delta) & (abs(y - 0.65) < (0.05+delta)):
            bools.append(True)
        elif (abs(x - 0.35) < delta) & (abs(y - 0.65) < (0.05+delta)):
            bools.append(True)
        elif ((x > 0.2) & (x < 0.8) &
              (abs(y - ((1.5*(x-0.5))**2 + 0.25)) < delta)):
            bools.append(True)
        else:
            bools.append(False)

    # Build training vectors
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


def complete_a_picture():
    """
    Here we give the net some random points (1 or 0) from a function and it tries to fill the
    rest of the space with what it thinks it should be
    """
    # x_min = -1.5; x_max = 1.5; y_min = -1.5; y_max = 1.5
    x_min = 0; x_max = 1.5; y_min = 0; y_max = 1.5
    train_in, train_out = make_checkerboard_training_set(num_points=1000, noise=0.00, randomize=True,
                                                         x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # x_min = 0.0; x_max = 1.0; y_min = 0.0; y_max = 1.0
    # train_in, train_out = make_smiley_training_set(num_points=1000)

    nnviz.plot_2d_classes(train_in, train_out, title="Training data",
                          x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, delta=0.01)

    training_size = train_in.shape[0]
    batch_size = round(training_size / 10.0)
    num_features = train_in.shape[1]

    numpy_net = NumpyNet(num_features, batch_size,
                         num_hidden=4, hidden_sizes=[64, 32, 32, 32],
                         activation="sigmoid", learning_rate=0.0002,
                         dropout_rate=None, weight_decay=None,
                         random_seed=None)
    numpy_net.report_model()

    numpy_net.train(train_in, train_out, epochs=200000,
                    visualize=True, visualize_percent=1, debug_visualize=True)


def paint_a_picture():
    """
    Here we give the net many examples of how to paint something like a square based on
    coordinates and try to get it to learn how to make that shape given input coords
    """
    print("foo")
    # Make a training set (many random i,j coord and an x by y box around that coord to start with)

    # Throw it into the net

    # Test how it does for some random coordinate inputs


if __name__ == '__main__':
    """
    Main driver.
    """
    # Set the logging level to normal and start run
    start_time = time.time()

    complete_a_picture()

    # paint_a_picture()

    # Shut down and clean up
    total_time = round((time.time() - start_time), 0)
    log.out.info("Execution time: " + str(total_time) + " sec")
    log.out.info("All Done!")
    log.stopLog()
