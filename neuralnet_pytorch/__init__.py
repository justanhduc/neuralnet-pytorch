from __future__ import print_function

minimum_required = '1.0.0'
try:
    import torch
except ImportError:
    # print out a more helpful error, then reraise
    print('Please install Pytorch (> %s) first at https://pytorch.org/' % minimum_required)
    raise

del torch
del minimum_required

from .version import author as __author__
from . import utils
from .utils import DataLoader, cuda_available, function
from .layers import *
from .metrics import *
from .resizing import *
from .normalization import *
from .optimizer import *
from .monitor import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
