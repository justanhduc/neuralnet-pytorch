import torch as T

from .. import utils
from .abstract import MultiMultiInputModule, MultiSingleInputModule, SingleMultiInputModule

__all__ = ['AdaIN', 'MultiModuleAdaIN', 'MultiInputAdaIN']


class _AdaIN:

    def normalize(self, input1, input2):
        mean1, std1 = T.mean(input1, self.dim1, keepdim=True), T.sqrt(T.var(input1, self.dim1, keepdim=True) + 1e-8)
        mean2, std2 = T.mean(input2, self.dim2, keepdim=True), T.sqrt(T.var(input2, self.dim2, keepdim=True) + 1e-8)
        return std2 * (input1 - mean1) / std1 + mean2


class AdaIN(SingleMultiInputModule, _AdaIN):
    """
    The original Adaptive Instance Normalization from https://arxiv.org/abs/1703.06868.

    :math:`Y_1 = \\text{module}(X)`

    :math:`Y_2 = \\text{module}(X)`

    :math:`Y = \\sigma_{Y_2} * (Y_1 - \\mu_{Y_1}) / \\sigma_{Y_1} + \\mu_{Y_2}`

    Parameters
    ----------
    module
        a :mod:`torch` module which generates target feature maps.
    dim
        dimension to reduce in the target feature maps.
        Default: ``(2, 3)``.
    """

    def __init__(self, module, dim=(2, 3)):
        super().__init__(module)
        self.dim1 = dim
        self.dim2 = dim

    def forward(self, *input, **kwargs):
        out1, out2 = super().forward(*input, **kwargs)
        return super().normalize(out1, out2)

    def extra_repr(self):
        s = 'dim={dim1}'.format(**self.__dict__)
        return s


class MultiModuleAdaIN(MultiSingleInputModule, _AdaIN):
    """
    A modified Adaptive Instance Normalization from https://arxiv.org/abs/1703.06868.

    :math:`Y_1 = \\text{module1}(X)`

    :math:`Y_2 = \\text{module2}(X)`

    :math:`Y = \\sigma_{Y_2} * (Y_1 - \\mu_{Y_1}) / \\sigma_{Y_1} + \\mu_{Y_2}`

    Parameters
    ----------
    module1
        a :mod:`torch` module which generates target feature maps.
    module2
        a :mod:`torch` module which generates style feature maps.
    dim1
        dimension to reduce in the target feature maps.
        Default: ``(2, 3)``.
    dim2
        dimension to reduce in the style feature maps.
        Default: ``(2, 3)``.
    """

    def __init__(self, module1, module2, dim1=(2, 3), dim2=(2, 3)):
        super().__init__(module1, module2)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input, *args, **kwargs):
        out1, out2 = super().forward(input, *args, **kwargs)
        return super().normalize(out1, out2)

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0]

    def extra_repr(self):
        s = 'dim1={dim1}, dim2={dim2}'.format(**self.__dict__)
        return s


class MultiInputAdaIN(MultiMultiInputModule, _AdaIN):
    """
    A modified Adaptive Instance Normalization from https://arxiv.org/abs/1703.06868.

    :math:`Y_1 = \\text{module1}(X_1)`

    :math:`Y_2 = \\text{module2}(X_2)`

    :math:`Y = \\sigma_{Y_2} * (Y_1 - \\mu_{Y_1}) / \\sigma_{Y_1} + \\mu_{Y_2}`

    Parameters
    ----------
    module1
        a :mod:`torch` module which generates target feature maps.
    module2
        a :mod:`torch` module which generates style feature maps.
    dim1
        dimension to reduce in the target feature maps.
        Default: ``(2, 3)``.
    dim2
        dimension to reduce in the style feature maps.
        Default: ``(2, 3)``.
    """

    def __init__(self, module1, module2, dim1=(2, 3), dim2=(2, 3)):
        super().__init__(module1, module2)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, *input, **kwargs):
        out1, out2 = super().forward(*input, **kwargs)
        return super().normalize(out1, out2)

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0]

    def extra_repr(self):
        s = 'dim1={dim1}, dim2={dim2}'.format(**self.__dict__)
        return s
