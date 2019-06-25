"""Init file for Pytorch-specific Gin-Config package. Adapted from Gin-config"""


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
        import torch as T
    except ImportError:
        # Print more informative error message, then reraise.
        print('\n\nFailed to import Pytorch.'
              'To use Gin-Config, please install the most recent version'
              'of Pytorch, by following instructions at'
              'https://pytorch.org/get-started/locally/.\n\n')
        raise


_ensure_pt_install()

try:
    from gin import *
    from neuralnet_pytorch.gin_nnt import external_configurables
except ImportError:
    print('Please install Git-config first via \'pip install git-config\'')
    raise

# Cleanup symbols to avoid polluting namespace.
import sys as _sys
for symbol in ["_ensure_pt_install", "_sys"]:
  delattr(_sys.modules[__name__], symbol)
