"""
numpynet_visualize
Supports numpynet visualizations in Visdom
@author: Chronocook (chronocook@gmail.com)
"""
import numpy as np
from visdom import Visdom


class NumpynetVizClient:

    def __init__(self,
                 viz,
                 loss_window=None,
                 loss_window_rolling=None,
                 prediction_2d=None,
                 layer_window_01=None,
                 layer_window_02=None,
                 weight_window_01=None,
                 weight_window_02=None
                 ):

        self.viz = viz
        self.loss_window = loss_window
        self.loss_window_rolling = loss_window_rolling
        self.prediction_2d = prediction_2d
        self.layer_window_01 = layer_window_01
        self.layer_window_02 = layer_window_02
        self.weight_window_01 = weight_window_01
        self.weight_window_02 = weight_window_02

    def plot_loss(self, loss_history, rolling_size=50):
        """
        Send loss plots to visdom client.  The first is the entire history of the loss and the
        second is a rolling window so that you can see details of the loss behavior
        :param loss_history: (list[float]) The history of losses through training
        :param rolling_size: Size of the rolling window
        :return:
        """

        y_vals = np.array(loss_history)
        x_vals = np.arange(1, len(loss_history)+1)
        rolling_size = max(1, rolling_size)
        rolling_size_safe = min(len(loss_history), rolling_size)
        y_vals_rolling = np.array(loss_history[len(loss_history)-rolling_size_safe:len(loss_history)])
        x_vals_rolling = np.arange(len(loss_history)-rolling_size_safe+1,
                                   len(loss_history)-rolling_size_safe + len(y_vals_rolling)+1)
        if self.loss_window is None:
            self.loss_window = self.viz.line(Y=y_vals,
                                        X=x_vals,
                                        opts=dict(
                                           title="Loss History",
                                           showlegend=False)
                                        )
            self.loss_window_rolling = self.viz.line(Y=y_vals_rolling,
                                                X=x_vals_rolling,
                                                opts=dict(
                                                    title="Loss History Rolling",
                                                    showlegend=False
                                                )
                                                )
        else:
            self.loss_window = self.viz.line(Y=y_vals,
                                        X=x_vals,
                                        win=self.loss_window,
                                        update='replace')
            self.loss_window_rolling = self.viz.line(Y=y_vals_rolling,
                                                X=x_vals_rolling,
                                                win=self.loss_window_rolling,
                                                update='replace')

    def plot_network(self, net):
        """
        Plots the guts of the network on the visdom client
        :param net: (object) a numpynet model object
        """

        num_layer = len(net.layer)
        if self.layer_window_01 is None:
            self.layer_window_01 = self.viz.heatmap(X=net.layer[1],
                                          opts=dict(
                                              title="First Hidden Layer",
                                              colormap='Electric',
                                              )
                                          )
            self.layer_window_02 = self.viz.heatmap(X=net.layer[num_layer - 2],
                                          opts=dict(
                                              title="Last Hidden Layer",
                                              colormap='Electric',
                                              )
                                          )
        else:
            self.layer_window_01 = self.viz.heatmap(X=net.layer[1],
                                          win=self.layer_window_01,
                                          opts=dict(
                                              title="First Hidden Layer",
                                              colormap='Electric',
                                              )
                                          )
            self.layer_window_02 = self.viz.heatmap(X=net.layer[num_layer - 2],
                                          win=self.layer_window_02,
                                          opts=dict(
                                              title="Last Hidden Layer",
                                              colormap='Electric',
                                              )
                                          )
        if self.weight_window_01 is None:
            self.weight_window_01 = self.viz.heatmap(X=net.weight[0],
                                           opts=dict(
                                               title="First Weights",
                                               colormap='Electric',
                                               )
                                           )
            self.weight_window_02 = self.viz.heatmap(X=net.weight[num_layer - 2],
                                           opts=dict(
                                               title="Last Weights",
                                               colormap='Electric',
                                               )
                                           )
        else:
            self.weight_window_01 = self.viz.heatmap(X=net.weight[0],
                                           win=self.weight_window_01,
                                           opts=dict(
                                               title="First Weights",
                                               colormap='Electric',
                                               )
                                           )
            self.weight_window_02 = self.viz.heatmap(X=net.weight[num_layer - 2],
                                           win=self.weight_window_02,
                                           opts=dict(
                                               title="Last Weights",
                                               colormap='Electric',
                                               )
                                           )

    def plot_2d_prediction(self, prediction_matrix, axis_x, axis_y, title="Current Prediction"):
        """
        Sends a plot of a prediction over 2D to visdom
        :param prediction_matrix: (np.array) array of predicted values
        :param axis_x: (np.array) values of x axis
        :param axis_y: (np.array) values of y axis
        :param title: (str) whatever you want to title the plot
        """
        if self.prediction_2d is None:
            self.prediction_2d = self.viz.heatmap(X=prediction_matrix,
                                        opts=dict(
                                            title=title,
                                            columnnames=list(axis_x.astype(str)),
                                            rownames=list(axis_y.astype(str)),
                                            colormap='Electric',
                                            )
                                        )
        else:
            self.prediction_2d = self.viz.heatmap(X=prediction_matrix,
                                        win=self.prediction_2d,
                                        opts=dict(
                                            title=title,
                                            columnnames=list(axis_x.astype(str)),
                                            rownames=list(axis_y.astype(str)),
                                            colormap='Electric',
                                        ))

    def plot_2d_classes(self, train_in, train_out,
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
        self.viz.heatmap(
            X=vis_matrix,
            opts=dict(
                title=title,
                columnnames=list(axis_x.astype(str)),
                rownames=list(axis_y.astype(str)),
                colormap='Electric',
            )
        )

    def plot_func(self, x_vals, y_vals, title="Foo"):
        """
        Plot any function you want
        :param x_vals: (np.array) x values
        :param y_vals: (np.array) y values
        :param title: (str) better title than "Foo"
        """
        self.viz.line(Y=y_vals,
                 X=x_vals,
                 opts=dict(
                    title=title,
                    showlegend=False)
                 )
