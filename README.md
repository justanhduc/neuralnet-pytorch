# Introduction
![Python - Version](https://img.shields.io/pypi/pyversions/neuralnet-pytorch.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/neuralnet-pytorch.svg)](https://pypi.org/project/neuralnet-pytorch/)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/neuralnet-pytorch.svg)](https://pypi.org/project/neuralnet-pytorch/)
[![Github - Tag](https://img.shields.io/github/tag/justanhduc/neuralnet-pytorch.svg)](https://github.com/justanhduc/neuralnet-pytorch/releases/tag/rel-0.0.4)
[![License](https://img.shields.io/github/license/justanhduc/neuralnet-pytorch.svg)](https://github.com/justanhduc/neuralnet-pytorch/blob/master/LICENSE.txt)
[![Build Status](https://travis-ci.org/justanhduc/neuralnet-pytorch.svg?branch=master)](https://travis-ci.org/justanhduc/neuralnet-pytorch)
[![Documentation Status](https://readthedocs.org/projects/neuralnet-pytorch/badge/?version=latest)](https://neuralnet-pytorch.readthedocs.io/en/latest/?badge=latest)

__A high level framework for general purpose neural networks in Pytorch.__

Personally, going from Theano to Pytorch is pretty much like
time traveling from 90s to the modern day.
However, we feel like despite having a lot of bells and whistles,
Pytorch is still missing many elements
that are confirmed to never be added to the library.
Therefore, this library is written to supplement more features
to the current magical Pytorch.
All the modules in the package directly subclass
the corresponding modules from Pytorch,
so everything should still be familiar.
For example, the following snippet in Pytorch

```
from torch import nn
model = nn.Sequential(
    nn.Conv2d(1, 20, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5, padding=2),
    nn.ReLU()
)
```

can be rewritten in Neuralnet-pytorch as 
```
import neuralnet_pytorch as nnt
model = nnt.Sequential(
    nnt.Conv2d(1, 20, 5, padding='half', activation='relu'),
    nnt.Conv2d(20, 64, 5, padding='half', activation='relu')
)
```

which is the same as the native Pytorch, or 

```
import neuralnet_pytorch as nnt
model = nnt.Sequential(input_shape=1)
model.add_module('conv1', nnt.Conv2d(model.output_shape, 20, 5, padding='half', activation='relu'))
model.add_module('conv2', nnt.Conv2d(model.output_shape, 64, 5, padding='half', activation='relu'))
```
which frees you from a lot of memorizations and manual calculations when adding one layer on top of another. 
Theano folks will also find some reminiscence as many functions are highly inspired by Theano.  

# Requirements

[Pytorch](https://pytorch.org/) >= 1.0.0

[Matplotlib](https://matplotlib.org/)

[Visdom](https://github.com/facebookresearch/visdom)

[Tensorboard](https://www.tensorflow.org/tensorboard)

[Gin-config](https://github.com/google/gin-config) (optional)

[Pykeops](https://github.com/getkeops/keops) 
(optional, required for the calculation of Sinkhorn Wasserstein loss.)

[Geomloss](https://github.com/jeanfeydy/geomloss) 
(optional, required for the calculation of Sinkhorn Wasserstein loss.)

# Documentation

https://neuralnet-pytorch.readthedocs.io

# Installation

Stable version
```
pip install --upgrade neuralnet-pytorch
```

Bleeding-edge version (recommended)

```
pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@master
```

To install the package with optional dependencies, try

```
pip install "neuralnet-pytorch[option] @ git+git://github.com/justanhduc/neuralnet-pytorch.git@master"
```
in which `option` can be `gin`/`geom`/`visdom`/`slack`.

To install the version with some collected Cuda/C++ ops, use

```
pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@fancy
```

# Usages

The manual reference is still under development and is available at https://neuralnet-pytorch.readthedocs.io.

# TODO

- [x] Adding introduction and installation 
- [x] Writing documentations
- [ ] Adding examples

# Disclaimer

This package is a product from my little free time during my PhD, 
so most but not all the written modules are properly checked. 
No replacements or refunds for buggy performance. 
All PRs are welcome. 

# Acknowledgements

The CUDA Chamfer distance is taken from the [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet) repo.

The AdaBound optimizer is taken from its [official repo](https://github.com/Luolc/AdaBound).

The adapted Gin for Pytorch code is taken from [Gin-config](https://github.com/google/gin-config).

The monitor scheme is inspired from [WGAN](https://github.com/igul222/improved_wgan_training).

The EMD CUDA implementation is adapted form [here](https://github.com/daerduoCarey/PyTorchEMD).
