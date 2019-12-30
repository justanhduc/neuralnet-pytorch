import logging

log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

from .tensor_utils import *
from .layer_utils import *
from .numpy_utils import *
from .activation_utils import *
from .cv_utils import *
from .misc_utils import *
from .data_utils import *
from .layer_utils import _make_input_shape

import torch

cuda_available = torch.cuda.is_available()
_image_shape = _make_input_shape(1, 2)
_matrix_shape = _make_input_shape(1, 0)
_pointset_shape = _make_input_shape(2, 0)

del torch
del _make_input_shape
