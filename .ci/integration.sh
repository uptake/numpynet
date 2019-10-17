#!/bin/bash

# failure is a natural part of life
set -e

# build the package
python setup.py install

echo "Checking code for style problems..."

    pycodestyle \
        --show-pep8 \
        --show-source \
        --verbose \
        $(pwd)

echo "Done checking code for style problems."

echo "Running integration tests ..."

python -m pytest

echo "Done runnning integration tests."
