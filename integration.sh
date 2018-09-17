#!/bin/bash

# failure is a natural part of life
set -e

# build the package
python setup.py install

# run tests
python -m pytest
