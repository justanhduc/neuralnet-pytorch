# Introduction
![Python - Version](https://img.shields.io/pypi/pyversions/neuralnet-pytorch.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/neuralnet-pytorch.svg)](https://pypi.org/project/neuralnet-pytorch/)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/neuralnet-pytorch.svg)](https://pypi.org/project/neuralnet-pytorch/)
[![Github - Tag](https://img.shields.io/github/tag/justanhduc/neuralnet-pytorch.svg)](https://github.com/justanhduc/neuralnet-pytorch/releases/tag/rel-0.0.4)
[![License](https://img.shields.io/github/license/justanhduc/neuralnet-pytorch.svg)](https://github.com/justanhduc/neuralnet-pytorch/blob/master/LICENSE.txt)

__A high level framework for general purpose neural networks in Pytorch.__

Personally, going from Theano to Pytorch is pretty much like time traveling from 90s to the modern day. 
However, despite a lot of bells and whistles, I still feel there are some missing elements from Pytorch 
which are confirmed to be never added to the library. 
Therefore, this library is written to add more features to the current magical Pytorch. All the modules here
directly subclass the corresponding modules from Pytorch, so everything should still be familiar. For example, the 
following snippet in Pytorch

```
from torch import nn
model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(1, 20, 5, padding=2))
model.add_module('relu1', nn.ReLU())
model.add_module('conv2', nn.Conv2d(20, 64, 5, padding=2))
model.add_module('relu2', nn.ReLU())
```

can be rewritten in Neuralnet-pytorch as 

```
import neuralnet_pytorch as nnt
model = nnt.Sequential((None, 1, None, None))
model.add_module('conv1', nnt.Conv2d(model.output_shape, 20, 5, padding='half', activation='relu'))
model.add_module('conv2', nnt.Conv2d(model.output_shape, 64, 5, padding='half', activation='relu'))
```
which frees you from doing a lot of manual calculations when adding one layer on top of another. Theano folks will also
find some reminiscence as many functions are highly inspired by Theano.  

# Requirements

[Pytorch](http://deeplearning.net/software/theano/)

[Scipy](https://www.scipy.org/install.html) 

[Numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) 

[Matplotlib](https://matplotlib.org/)

[Visdom](https://github.com/facebookresearch/visdom)

[TensorboardX](https://github.com/lanpa/tensorboardX)

# Installation

Stable version
```
pip install --upgrade neuralnet-pytorch
```

Bleeding-edge version

```
pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@master
```

To install the version without some collected Cuda/C++ ops, use

```
pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@plain
```

# Usages

The manual reference is still under development and is available at https://neuralnet-pytorch.readthedocs.io.

# TODO

- [x] Adding introduction and installation 
- [ ] Writing documentations
- [ ] Adding examples

# Disclaimer

This package is a product from my little free time during my PhD, 
so most but not all the written modules are properly checked. 
No replacements or refunds for buggy performance. 
All PRs are welcome. 

# Acknowledgements

The CUDA Chamfer distance is taken from the [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet) repo.

The AdaBound optimizer is taken from its [official repo](https://github.com/Luolc/AdaBound). 
