from __future__ import print_function

minimum_required = '1.0.0'


# Ensure Pytorch is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import Pytorch, too.
def _ensure_pt_install():  # pylint: disable=g-statement-before-imports
    """Attempt to import Pytorch, and ensure its version is sufficient.
    Raises:
    ImportError: if either Pytorch is not importable or its version is
    inadequate.
    """
    try:
        import torch
    except ImportError:
        # Print more informative error message, then reraise.
        print('\n\nFailed to import Pytorch. '
              'To use neuralnet-pytorch, please install '
              'Pytorch (> %s) by following instructions at '
              'https://pytorch.org/get-started/locally/.\n\n' % minimum_required)
        raise

    del torch


_ensure_pt_install()

# Cleanup symbols to avoid polluting namespace.
del minimum_required
import sys as _sys

for symbol in ['_ensure_pt_install', '_sys']:
    delattr(_sys.modules[__name__], symbol)

try:
    import neuralnet_pytorch.ext as ext
    cuda_ext_available = True
    del ext
except ModuleNotFoundError:
    cuda_ext_available = False

from . import utils
from .utils import DataLoader, DataPrefetcher, cuda_available, function
from .layers import *
from .metrics import *
from .monitor import *
from . import optim
from . import zoo

from .version import author as __author__
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
