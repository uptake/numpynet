# NumpyNet
[![Travis Build Status](https://img.shields.io/travis/uptake/numpynet?logo=travis)](https://travis-ci.org/uptake/numpynet)
[![Open issues](https://img.shields.io/github/issues-raw/uptake/numpynet)](https://github.com/uptake/numpynet/issues)
[![license](https://img.shields.io/github/license/uptake/numpynet)](https://github.com/uptake/numpynet/blob/master/LICENSE)



## What is NumpyNet?
----
NumpyNet is a very simple python framework for neural networks.  It meant to be a teaching tool so that people can really get under the hood and learn the basics about how neural network are built and how they work.  
It includes nice visualizations of the process so that the user can watch what is going on as the models learn and make predictions.  It's only dependencies are numpy, which does the math, and [visdom](https://github.com/facebookresearch/visdom), which does the visualizations.
![](https://raw.githubusercontent.com/uptake/numpynet/master/readme_figures/demo.gif)



## Quick Start
----
Grab NumpyNet:

    git clone https://github.com/uptake/numpynet.git
    cd numpynet

Install NumpyNet (will install `visdom` as well):

    python setup.py install

Start visdom server locally:

    visdom

[Open up http://localhost:8097 in a browser](http://localhost:8097)

Run a demo and have some fun:

    python examples.py

## State of the Repo
----
Currently this project is in it's infancy. The basic functionality is there but there's still a lot to do. So get in there and [add some issues](https://github.com/UptakeOpenSource/numpynet/issues) you'd like to see or better yet contribute some code!

## Testing the code
----
Take a look at our [travis.yml](.travis.yml) for integration testing using [Travis CI](https://travis-ci.org). For local testing use `./integration.sh`.

## Other Educational Resources
----
Check out these resouces in concert with `NumpyNet` for a full appreciation of how a neural network works: 

### [KDnuggets - Nothing but NumPy: Understanding & Creating Neural Networks with Computational Graphs from Scratch](https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html)
> Understanding new concepts can be hard, especially these days when there is an avalanche of resources with only cursory explanations for complex concepts. This blog is the result of a dearth of detailed walkthroughs on how to create neural networks in the form of computational graphs.  In this blog posts, I consolidate all that I have learned as a way to give back to the community and help new entrants. I will be creating common forms of neural networks all with the help of nothing but NumPy.

### [Andrew Ng's Machine Learning](https://www.coursera.org/learn/machine-learning)
> Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI.

