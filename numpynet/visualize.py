"""
numpynet_visualize
Supports numpynet visualizations in Visdom
@author: Chronocook (chronocook@gmail.com)
"""
import numpy as np
from visdom import Visdom
from numpynet.loggit import log


# TODO clean up indents

class NumpynetVizClient:

    def __init__(self,
                 viz=None,
                 loss_window=None,
                 loss_window_rolling=None,
                 prediction_2d=None,
                 layer_window_01=None,
                 layer_window_02=None,
                 weight_window_01=None,
                 weight_window_02=None
                 ):
        if viz is None:
            self.viz = Visdom()
        else:
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
            self.prediction_2d = self.viz.heatmap(X=prediction_matrix.T,
                                        opts=dict(
                                            title=title,
                                            columnnames=list(axis_x.astype(str)),
                                            rownames=list(axis_y.astype(str)),
                                            colormap='Electric',
                                            )
                                        )
        else:
            self.prediction_2d = self.viz.heatmap(X=prediction_matrix.T,
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
        vis_matrix = np.zeros((len(axis_y), len(axis_x))) + 0.5
        for c, coord in enumerate(train_in):
            idx = (np.abs(axis_x - coord[0])).argmin()
            idy = (np.abs(axis_y - coord[1])).argmin()
            vis_matrix[idy, idx] = train_out[c]

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

    def test_svg(self, net):

        svg_str = """
        <svg width="900px" height="500px" viewBox="0 0 900 500" preserveAspectRatio="xMidYMid meet" >
        <rect id="numpynet_architecture" x="0" y="0" width="900" height="500" style="fill: none; stroke: none;"/>
        
        <circle id="c_000_000" cx="100" cy="100" style="fill:khaki;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_000_001" cx="100" cy="150" style="fill:khaki;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_000_002" cx="100" cy="200" style="fill:khaki;stroke:black;stroke-width:1px;" r="20"/>
        
        <circle id="c_001_000" cx="200" cy="50" style="fill:cornflowerblue;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_001_001" cx="200" cy="100" style="fill:cornflowerblue;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_001_002" cx="200" cy="150" style="fill:cornflowerblue;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_001_003" cx="200" cy="200" style="fill:cornflowerblue;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_001_004" cx="200" cy="250" style="fill:cornflowerblue;stroke:black;stroke-width:1px;" r="20"/>
        
        <circle id="c_002_000" cx="300" cy="100" style="fill:darkorange;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_002_001" cx="300" cy="150" style="fill:darkorange;stroke:black;stroke-width:1px;" r="20"/>
        <circle id="c_002_002" cx="300" cy="200" style="fill:darkorange;stroke:black;stroke-width:1px;" r="20"/>
        
        <line id="l_000000" x1="100" y1="100" x2="200" y2="40" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000001" x1="100" y1="100" x2="200" y2="90" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000002" x1="100" y1="100" x2="200" y2="140" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000003" x1="100" y1="150" x2="200" y2="40" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000004" x1="100" y1="150" x2="200" y2="90" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000005" x1="100" y1="150" x2="200" y2="140" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000006" x1="100" y1="200" x2="200" y2="40" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000007" x1="100" y1="200" x2="200" y2="90" style="stroke:black;fill:none;stroke-width:1px;"/>
        <line id="l_000008" x1="100" y1="200" x2="200" y2="140" style="stroke:black;fill:none;stroke-width:1px;"/>
        
        <text style="fill:black;font-family:Arial;font-size:20px;" x="100" y="400" id="e16_texte" >Layer 1</text>
        
        </svg>
        """

        width = 800
        height = 600
        margin = 0.05
        w = int(width * (1.0 - margin))
        h = int(height * (1.0 - margin))
        # Figure out size of circles and how they'll sit on the drawing
        max_layer_size = np.max(net.layer_sizes)
        radius = int(np.floor(h / (2.5 * max_layer_size)))
        if radius < 10.0:
            log.out.warning("High layer sizes, this image is going to be pretty busy.")

        # Construct the svg
        svg_str = """
        <svg width="%spx" height="%spx" viewBox="0 0 %s %s" preserveAspectRatio="xMidYMid meet" >
        <rect id="numpynet_architecture" x="0" y="0" width="%s" height="%s" style="fill: none; stroke: none;"/>
        """
        populate_tuple = (width, height, width, height, width, height)
        svg_str = svg_str % populate_tuple

        # Add a layer of circles
        layer_sizes = net.layer_sizes
        layer_colors = ["khaki"]
        layer_colors += ["cornflowerblue"] * (net.num_layers - 2)
        layer_colors += ["darkorange"]
        cid = 0
        lid = 0

        x_positions = list()
        y_positions = list()
        x_space = int(np.floor(w / (len(layer_sizes)+0.5)))
        for i in range(len(layer_sizes)):
            x_positions.append(x_space + (x_space*i))
        for i in range(len(layer_sizes)):
            y_positions_last = list(y_positions)
            y_positions = list()
            if layer_sizes[i] % 2 == 0:
                for j in range(int(np.floor(layer_sizes[i]/2))):
                    y_positions.append(int(h/2.0) + ((j+1)*(radius*2.5)) - (radius*1.25))
                    y_positions.append(int(h/2.0) - ((j+1)*(radius*2.5)) + (radius*1.25))
            else:
                y_positions.append(int(h/2.0))
                for j in range(int(np.floor(layer_sizes[i]/2))):
                    y_positions.append(int(h/2.0) + ((j+1)*(radius*2.5)))
                    y_positions.append(int(h/2.0) - ((j+1)*(radius*2.5)))
            y_positions = sorted(y_positions)

            for j in range(layer_sizes[i]):
                svg_str += """
                    \n <circle id="c_%s" cx="%s" cy="%s" style="fill:%s;stroke:black;stroke-width:1px;" r="%s"/>
                    """
                populate_tuple = (cid, x_positions[i], y_positions[j], layer_colors[i], radius)
                svg_str = svg_str % populate_tuple
                cid += 1
                # Draw lines
                if i > 0:
                    for last_j in range(layer_sizes[i-1]):
                        svg_str += """
                            \n <line id="%s" x1="%s" y1="%s" x2="%s" y2="%s" 
                            style="stroke:black;fill:none;stroke-width:1px;"/>
                            """
                        populate_tuple = (lid, x_positions[i], y_positions[j], x_positions[i-1], y_positions_last[last_j])
                        svg_str = svg_str % populate_tuple
                        lid += 1
        svg_str += "\n </svg>"

        self.viz.svg(svg_str, opts=dict(title="Numpynet Architecture",
                                   width=width,
                                   height=height,
                                   showlegend=False))
