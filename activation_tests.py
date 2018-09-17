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
