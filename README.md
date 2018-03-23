# example

[![Build
Status](https://travis-ci.org/mepittma/bmi203-final.svg?branch=master)](https://travis-ci.org/mepittma/bmi206-final)

Example python project with testing.

## usage

To use the package, first make a new conda environment and activate it

```
conda create -n nn_env python=3
source activate nn_env
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function can be run as follows

```
python train_nets.py
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
