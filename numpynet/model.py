"""
numpynet_model
Supports numpynet visualizations in Visdom
@author: Chronocook (chronocook@gmail.com)
"""
import math
import pickle
import numpy as np
import numpynet.common as common
from numpynet.loggit import log
import atexit
import copy

best_self = None

class NumpyNet:
    """
    The main object that defines a neural network architecture and it's train, predict, and forward operators
    """
    layer = []   # Also called neurons
    weight = []  # Also called synapses
    num_layers = 0
    viz = None

    def __init__(self, num_features, batch_size, num_hidden=0, hidden_sizes=None,
                 activation="sigmoid", learning_rate=0.01,
                 learning_decay=None, weight_decay=None, dropout_rate=None,
                 init_weight_spread=1.0, random_seed=None):
        """
        Initialize a blank numpy net object
        This object will have input/output layers and weights (neurons and synapses)
        Both will be lists of numpy arrays having varying sizes
        Synapses are initialized with random weights with mean 0
        :param num_features: (int) Shape of the input layer
        :param batch_size: (int) Size of the batches you will be running through the net while training
        :param num_hidden: (int) Number of hidden layers
        :param hidden_sizes: (list[int]) The sizes you want your hidden layers to be
        :param activation: (str) or (list[str]) The activation function you want to use,
                           if given a list will specify an activation function for each layer explicitly
        :param learning_rate:
        :param learning_decay:
        :param weight_decay:
        :param dropout_rate:
        :param init_weight_spread:
        :param random_seed: (int) Can specify the random seed if you want to reproduce the same runs
        """
        self.default_layer_size = 16
        # Initialize arrays used for neurons and synapses
        self.batch_size = batch_size
        self.num_layers = 2 + num_hidden  # Input, output, and hidden layers
        self.num_hidden = num_hidden
        self.layer = [np.empty(0)] * self.num_layers
        self.weight = [np.empty(0)] * (self.num_layers - 1)

        # Set all of the activation functions, you need one for each layer of weights, or one
        # less than the number of layers. This can be a list of different functions or a string for all the same type
        if isinstance(activation, str):
            self.activation_function = [common.Activation(activation).function] * (self.num_layers - 1)
            self.activation_names = [activation] * self.num_layers
        elif isinstance(activation, list):
            if len(activation) == self.num_layers - 1:
                self.activation_function = list()
                for i in range(self.num_layers - 1):
                    print(i, activation[i])
                    self.activation_function.append(common.Activation(activation[i]).function)
                self.activation_names = activation
            else:
                msg = ("activation_function must be one less than the number of layers in your network "
                       "(num_layers-1 = " + str(self.num_layers - 1) + ")")
                log.out.error(msg)
                raise ValueError(msg)
        else:
            msg = "activation_function must be a string or a list of strings"
            log.out.error(msg)
            raise ValueError(msg)

        # Set network hyperparameters
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # For diagnostics
        self.loss_history = list()  # This will be appended to a lot so a list is a bit better
        self.predict_space = None  # If left undefined will define by training input bounds
        self.input_shape = [batch_size, num_features]
        if hidden_sizes is None:
            self.hidden_sizes = [self.default_layer_size] * self.num_hidden
        else:
            self.hidden_sizes = hidden_sizes
        self.output_shape = [batch_size, 1]

        # If requested seed random numbers to make calculation (makes repeatable)
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            current_seed = int(np.random.random(1) * 4.0E9)  # 4 billion is close to limit of 32 bit unsigned int
            np.random.seed(current_seed)
            log.out.info("No random seed selected, using: " + str(current_seed))

        # Initialize weights with random noise centered around zero, spread set by init_weight_spread
        self.weight[0] = (init_weight_spread * 2) * np.random.random([self.input_shape[1], self.hidden_sizes[0]]) - init_weight_spread
        for i in range(self.num_hidden-1):
            self.weight[i+1] = (init_weight_spread * 2) * np.random.random([self.weight[i].shape[1], self.hidden_sizes[i+1]]) - init_weight_spread
        self.weight[self.num_hidden] = (init_weight_spread * 2) * np.random.random([self.weight[self.num_hidden-1].shape[1], self.output_shape[1]]) - init_weight_spread

        # Initialize layers with zeros
        self.forward(np.zeros(self.input_shape))

        # Add this list of all the layer sizes for easy access later
        self.layer_sizes = list()
        for layer in self.layer:
            self.layer_sizes.append(layer.shape[1])

    def set_viz_client(self, viz_client):
        """
        Optionally instrument NumpyNet with a Visdom client
        to make cool plots during training.
        """
        self.viz = viz_client

    def forward(self, input_features):
        """
        The Forward operator, this method updates the net's current layers
        :param input_features: (np.array) An n-dimensional array of the input feature vectors
        """
        # Feed forward through layers
        self.layer[0] = input_features
        for i in range(self.num_layers - 1):
            self.layer[i + 1] = self.activation_function[i](np.dot(self.layer[i], self.weight[i]))

    def predict(self, input_features):
        """
        A prediction method, this method does not update the object but returns a prediction
        :param input_features: (np.array) An n-dimensional array of the input feature vectors
        :return: The predicted vector from the current model
        """
        # Feed forward through layers not saving result in network and return the result
        prediction = input_features
        for i in range(self.num_layers - 1):
            prediction = self.activation_function[i](np.dot(prediction, self.weight[i]))
        return prediction

    def train(self, train_in, train_out, epochs=100, save_best=None,
              visualize=True, visualize_percent=5.0, debug_visualize=True):
        """
        This method trains the network by feeding in input features (train_in) and testing the model's
        prediction against a known set of target vectors (train_out)
        :param train_in: (np.array) Input training data (features)
        :param train_out: (np.array) Output training data (targets)
        :param epochs: How many times you want to iterate over the whole training dataset
        :param save_best: (str) If specified
        :param visualize: (bool) Whether or not to visualize the training process
        :param visualize_percent: (float) If above is true how often you want to visualize (percent of epochs)
        :param debug_visualize: (bool) Turns on debug visualizations
        """
        self.save_best = save_best
        atexit.register(self.execute_at_exit)
        set_size = train_in.shape[0]
        log.out.info("Given " + str(set_size) + " training points.")
        iterations = math.ceil(set_size / self.batch_size)
        log.out.info("Will train in " + str(iterations) + " iterations per epoch for " + str(epochs) +
                     " epochs. (In batches of " + str(self.batch_size) + ")")
        runfracround = round(epochs * (0.01 * visualize_percent))
        log.out.info("Will output every " + str(runfracround) + " epochs.")
        # Set prediction space (for diagnostics)
        if self.predict_space is None:
            self.predict_space = [np.min(train_in[:, 0]), np.max(train_in[:, 0]),
                                  np.min(train_in[:, 1]), np.max(train_in[:, 1])]
        # Set error matrix
        error = [None] * len(self.layer)
        delta = [None] * len(self.layer)

        # Epoch training loop (each epoch goes over entire data set once)
        for e, epoch in enumerate(range(epochs)):
            # Reset the available data indices
            available_indexes = np.arange(set_size)
            # Loop over the batches of data for this epoch
            batch_loss = list()
            for t in range(iterations):
                # Get random data for this batch
                if available_indexes.size < self.batch_size:
                    add_randoms = np.random.randint(set_size, size=self.batch_size-available_indexes.size)
                    available_indexes = np.concatenate((available_indexes, add_randoms))
                # TODO: try grid sampling instead of random
                batch_indexes = np.random.choice(available_indexes, self.batch_size, replace=False)
                batch_in = train_in[batch_indexes, :]
                batch_out = train_out[batch_indexes, :]
                # Remove these indices from the available pool
                available_indexes = available_indexes[~np.in1d(available_indexes,
                                                               batch_indexes).reshape(available_indexes.shape)]

                # Run the network forward with the current weights
                self.forward(batch_in)

                # Propagate backwards through the layers and calculate error
                # Start with the output layer
                layer_index = self.num_layers - 1
                error[layer_index] = batch_out - self.layer[layer_index]
                # Find the direction of the target value and move towards it depending on confidence
                delta[layer_index] = (self.learning_rate * error[layer_index] *
                                      self.activation_function[layer_index-1](self.layer[layer_index], deriv=True))

                # Work backwards through the hidden layers
                for layer_index in range(len(self.layer) - 2, 0, -1):
                    error[layer_index] = delta[layer_index + 1].dot(self.weight[layer_index].T)
                    # Find the direction of the target value and move towards it depending on confidence
                    delta[layer_index] = error[layer_index] * \
                                         self.activation_function[layer_index-1](self.layer[layer_index], deriv=True)

                # Update the weights using the deltas we just found
                for layer_index in range(len(self.layer) - 1, 0, -1):
                    self.weight[layer_index - 1] += self.layer[layer_index - 1].T.dot(delta[layer_index])

                batch_loss.append(np.sum(np.abs(error[-1])))

            self.loss_history.append(sum(batch_loss) / (iterations * len(train_in)))

            # If saving the best models check if this is the best so far
            if save_best is not None:
                if self.loss_history[-1] == min(self.loss_history):
                    log.out.info("Current model is the best so far, copying.")
                    best_self = copy.copy(self)

            # Report error every x% and output visualization
            if (e % runfracround) == 0:
                log.out.info("Epoch: " + str(e) + " Current loss: " + str(self.loss_history[-1]))
                if visualize and self.viz is not None:
                    self.viz.plot_loss(self.loss_history, rolling_size=runfracround)
                    prediction_matrix, axis_x, axis_y = common.predict_2d_space(self, delta=0.02)
                    self.viz.plot_2d_prediction(prediction_matrix, axis_x, axis_y)
                    if debug_visualize:
                        self.viz.plot_network(self)

            if self.weight_decay is not None:
                for layer_index in range(len(self.layer) - 1, 0, -1):
                    self.weight[layer_index - 1] -= self.weight[layer_index - 1] * self.learning_rate * self.weight_decay
            #TODO visualize weight growth, should this be by epoch or every?
            # if self.dropout_rate is not None:

        log.out.info("Final Error: " + str(np.mean(np.abs(error[-1]))))
        prediction_matrix, axis_x, axis_y = common.predict_2d_space(self, delta=0.002)
        if visualize and self.viz is not None:
            self.viz.plot_2d_prediction(prediction_matrix, axis_x, axis_y, title="Final Prediction")

    def report_model(self):
        """
        Simple method to report the model details (could override print but this is fine)
        """
        log.out.info("Model topology: ")
        log.out.info("Number of layers: " + str(self.num_layers) + " (" + str(self.num_layers - 2) + " hidden)")
        for l in range(self.num_layers-1):
            log.out.info("Layer " + str(l+1) + ": " + str(self.layer[l].shape))
            log.out.info("Weight " + str(l+1) + ": " + str(self.weight[l].shape))
            log.out.info("Activation " + str(l+1) + ": " + str(self.activation_names[l]))
        log.out.info("Layer " + str(self.num_layers) + ": " + str(self.layer[-1].shape))

    def execute_at_exit(self):
        if self.save_best is not None:
            log.out.info("Saving numpynet object to a pickle")
            with open(self.save_best, 'wb') as output_handle:
                pickle.dump(best_self, output_handle, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """
        Loads a NumpyNet object from a pickle
        :param filename: (str) The name of the file you wan to load
        :return: (object) The net object
        """
        log.out.info("Loading numpynet object from a pickle")
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        return net
