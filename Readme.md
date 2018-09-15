# NumpyNet


## What is NumpyNet?
----
NumpyNet is a very simple python framework for neural networks.  It meant to be a teaching tool so that people can really get under the hood and learn the basics about how neural network are built and how they work.  
It includes nice visualizations of the process so that the user can watch what is going on as the models learn and make predictions.  It's only dependencies are numpy, which does the math, and [visdom](https://github.com/facebookresearch/visdom), which does the visualizations.

## State of the Repo
----
Currently this project is in it's infancy. The basic functionality is there but there's still a lot to do. So get in there and [add some issues](https://github.com/UptakeOpenSource/numpynet/issues) you'd like to see or better yet contribute some code!

## Quick Start
----
Grab NumpyNet:

    git clone https://github.com/UptakeOpenSource/numpynet.git
    cd numpynet

Install NumpyNet (will install `visdom` as well):

    python setup.py install

Start visdom server locally:

    visdom

[Open up http://localhost:8097 in a browser](http://localhost:8097)

Run a demo and have some fun:

    python examples.py
