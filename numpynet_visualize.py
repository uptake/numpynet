"""
Supports numpynet visualizations in Visdom
"""
import numpy as np
from visdom import Visdom

viz = Visdom()
viz.close()
# Initialize these global windows for things like refreshing
loss_window = None
loss_window_rolling = None
prediction_2d = None
layer_window_01 = None
layer_window_02 = None
weight_window_01 = None
weight_window_02 = None


def plot_loss(loss_history, rolling_size=50):
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


def plot_network(layer, weight):
    global layer_window_01
    global layer_window_02
    global weight_window_01
    global weight_window_02

    num_layer = len(layer)
    size_layer = len(layer[0])
    if layer_window_01 is None:
        layer_window_01 = viz.heatmap(X=layer[1],
                                      opts=dict(
                                          title="First Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
        layer_window_02 = viz.heatmap(X=layer[num_layer-2],
                                      opts=dict(
                                          title="Last Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
    else:
        layer_window_01 = viz.heatmap(X=layer[1],
                                      win=layer_window_01,
                                      opts=dict(
                                          title="First Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
        layer_window_02 = viz.heatmap(X=layer[num_layer-2],
                                      win=layer_window_02,
                                      opts=dict(
                                          title="Last Hidden Layer",
                                          colormap='Electric',
                                          )
                                      )
    if weight_window_01 is None:
        weight_window_01 = viz.heatmap(X=weight[0],
                                       opts=dict(
                                           title="First Weights",
                                           colormap='Electric',
                                           )
                                       )
        weight_window_02 = viz.heatmap(X=weight[num_layer-2],
                                       opts=dict(
                                           title="Last Weights",
                                           colormap='Electric',
                                           )
                                       )
    else:
        weight_window_01 = viz.heatmap(X=weight[0],
                                       win=weight_window_01,
                                       opts=dict(
                                           title="First Weights",
                                           colormap='Electric',
                                           )
                                       )
        weight_window_02 = viz.heatmap(X=weight[num_layer-2],
                                       win=weight_window_02,
                                       opts=dict(
                                           title="Last Weights",
                                           colormap='Electric',
                                           )
                                       )

def plot_2d_prediction(prediction_matrix, axis_x, axis_y, title="Current Prediction"):
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
                    title="2d Classification"):
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
    viz.line(Y=y_vals,
             X=x_vals,
             opts=dict(
                title=title,
                showlegend=False)
             )
