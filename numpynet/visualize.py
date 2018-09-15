"""
numpynet_visualize
Supports numpynet visualizations in Visdom
@author: Chronocook (chronocook@gmail.com)
"""
import numpy as np
from visdom import Visdom

#viz = Visdom()
#viz.close()

# Initialize these global windows for things like refreshing
#loss_window = None
#loss_window_rolling = None
#prediction_2d = None
#layer_window_01 = None
#layer_window_02 = None
#weight_window_01 = None
#weight_window_02 = None


def plot_loss(loss_history, rolling_size=50):
    """
    Send loss plots to visdom client.  The first is the entire history of the loss and the
    second is a rolling window so that you can see details of the loss behavior
    :param loss_history: (list[float]) The history of losses through training
    :param rolling_size: Size of the rolling window
    :return:
    """
    global loss_window
    global loss_window_rolling
    y_vals = np.array(loss_history)
    x_vals = np.arange(1, len(loss_history)+1)
    rolling_size = max(1, rolling_size)
    rolling_size_safe = min(len(loss_history), rolling_size)
    y_vals_rolling = np.array(loss_history[len(loss_history)-rolling_size_safe:len(loss_history)])
    x_vals_rolling = np.arange(len(loss_history)-rolling_size_safe+1,
                               len(loss_history)-rolling_size_safe + len(y_vals_rolling)+1)
    if loss_window is None:
        loss_window = viz.line(Y=y_vals,
                               X=x_vals,
                               opts=dict(
                                   title="Loss History",
                                   showlegend=False)
                               )
        loss_window_rolling = viz.line(Y=y_vals_rolling,
                                       X=x_vals_rolling,
                                       opts=dict(
                                           title="Loss History Rolling",
                                           showlegend=False)
                                       )
    else:
        loss_window = viz.line(Y=y_vals,
                               X=x_vals,
                               win=loss_window,
                               update='replace')
        loss_window_rolling = viz.line(Y=y_vals_rolling,
                                       X=x_vals_rolling,
                                       win=loss_window_rolling,
                                       update='replace')


def plot_network(net):
    """
    Plots the guts of the network on the visdom client
    :param net: (object) a numpynet model object
    """
    # TODO can add functionality to this, the plots aren't super meaningful right now
    global layer_window_01
    global layer_window_02
    global weight_window_01
    global weight_window_02

    num_layer = len(net.layer)
    if layer_window_01 is None:
        layer_window_01 = viz.heatmap(X=net.layer[1],
                                      opts=dict(
                                          title="First Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
        layer_window_02 = viz.heatmap(X=net.layer[num_layer - 2],
                                      opts=dict(
                                          title="Last Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
    else:
        layer_window_01 = viz.heatmap(X=net.layer[1],
                                      win=layer_window_01,
                                      opts=dict(
                                          title="First Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
        layer_window_02 = viz.heatmap(X=net.layer[num_layer - 2],
                                      win=layer_window_02,
                                      opts=dict(
                                          title="Last Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
    if weight_window_01 is None:
        weight_window_01 = viz.heatmap(X=net.weight[0],
                                       opts=dict(
                                           title="First Weights",
                                           colormap='Electric',
                                           )
                                       )
        weight_window_02 = viz.heatmap(X=net.weight[num_layer - 2],
                                       opts=dict(
                                           title="Last Weights",
                                           colormap='Electric',
                                           )
                                       )
    else:
        weight_window_01 = viz.heatmap(X=net.weight[0],
                                       win=weight_window_01,
                                       opts=dict(
                                           title="First Weights",
                                           colormap='Electric',
                                           )
                                       )
        weight_window_02 = viz.heatmap(X=net.weight[num_layer - 2],
                                       win=weight_window_02,
                                       opts=dict(
                                           title="Last Weights",
                                           colormap='Electric',
                                           )
                                       )


def plot_2d_prediction(prediction_matrix, axis_x, axis_y, title="Current Prediction"):
    """
    Sends a plot of a prediction over 2D to visdom
    :param prediction_matrix: (np.array) array of predicted values
    :param axis_x: (np.array) values of x axis
    :param axis_y: (np.array) values of y axis
    :param title: (str) whatever you want to title the plot
    """
    global prediction_2d
    if prediction_2d is None:
        prediction_2d = viz.heatmap(X=prediction_matrix,
                                    opts=dict(
                                        title=title,
                                        columnnames=list(axis_x.astype(str)),
                                        rownames=list(axis_y.astype(str)),
                                        colormap='Electric',
                                        )
                                    )
    else:
        prediction_2d = viz.heatmap(X=prediction_matrix,
                                    win=prediction_2d,
                                    opts=dict(
                                        title=title,
                                        columnnames=list(axis_x.astype(str)),
                                        rownames=list(axis_y.astype(str)),
                                        colormap='Electric',
                                    ))


def plot_2d_classes(train_in, train_out,
                    x_min=0.0, x_max=1.0,
                    y_min=0.0, y_max=1.0, delta=0.02,
                    title="2d Classifications"):
    """
    A lot like plot_2d_prediction above but supports plotting the sparse grid
    :param train_in: (np.array) array of predicted values
    :param train_out: (np.array) array of predicted values
    :param x_min: (float) minimum x value of the domain
    :param x_max: (float) maximum x value of the domain
    :param y_min: (float) minimum y value of the domain
    :param y_max: (float) maximum y value of the domain
    :param delta: (float) The resolution of the grid
    :param title: (str) A title for the plot
    """
    axis_x = np.arange(x_min, x_max + delta, delta)
    axis_y = np.arange(y_min, y_max + delta, delta)
    vis_matrix = np.zeros((len(axis_x), len(axis_y))) + 0.5
    for c, coord in enumerate(train_in):
        idx = (np.abs(axis_x - coord[0])).argmin()
        idy = (np.abs(axis_y - coord[1])).argmin()
        vis_matrix[idx, idy] = train_out[c]

    # Make a heatmap
    viz = Visdom()
    viz.heatmap(
        X=vis_matrix,
        opts=dict(
            title=title,
            columnnames=list(axis_x.astype(str)),
            rownames=list(axis_y.astype(str)),
            colormap='Electric',
        )
    )


def plot_func(x_vals, y_vals, title="Foo"):
    """
    Plot any function you want
    :param x_vals: (np.array) x values
    :param y_vals: (np.array) y values
    :param title: (str) better title than "Foo"
    """
    viz.line(Y=y_vals,
             X=x_vals,
             opts=dict(
                title=title,
                showlegend=False)
             )
