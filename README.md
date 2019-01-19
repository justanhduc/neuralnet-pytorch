# Introduction
A high level framework for general purpose neural networks in Pytorch.

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
model.add_module('conv2', nn.Conv2d(20, 64, 5, padding=2))
```

can be rewritten in Neuralnet-pytorch as 

```
import neuralnet_pytorch as nnt
model = nnt.Sequential((None, 1, None, None))
model.add_module('conv1', nnt.Conv2d(model.output_shape, 20, 5, padding='half', activation='relu'))
model.add_module('conv2', nnt.Conv2d(model.output_shape, 64, 5, padding='half', activation='relu'))
```
which frees you from doing a lot of manual calculations when adding one layer on top of others. Theano folks will also
find some reminiscence as many function interfaces are highly inspired by Theano. 

## Requirements

[Pytorch](http://deeplearning.net/software/theano/)

[Scipy](https://www.scipy.org/install.html) 

[Numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) 

[Matplotlib](https://matplotlib.org/)

[Visdom](https://github.com/facebookresearch/visdom)

## Installation

Stable version
```
pip install --upgrade neuralnet-pytorch
```

Bleeding-edge version

```
pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@master
```

## Usages

...
## TODO

- [x] Adding introduction and installation 
- [ ] Writing documentations
- [ ] Adding examples

## Disclaimer

Most but not all the written modules are properly checked. No replacements or refunds for buggy performance. 
All PRs are welcome. 
