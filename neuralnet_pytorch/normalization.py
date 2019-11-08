import torch as T
import torch.nn as nn

import neuralnet_pytorch as nnt
from neuralnet_pytorch import utils
from neuralnet_pytorch.utils import _image_shape, _matrix_shape, _pointset_shape
from neuralnet_pytorch.layers import _LayerMethod, MultiMultiInputModule, MultiSingleInputModule
from neuralnet_pytorch.utils import cuda_available

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'InstanceNorm2d', 'AdaIN', 'MultiInputAdaIN',
           'FeatureNorm1d', 'InstanceNorm1d', 'GroupNorm']


@utils.add_simple_repr
@utils.no_dim_change_op
class BatchNorm1d(nn.BatchNorm1d, _LayerMethod):
    """
    Performs batch normalization on 1D signals.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        input_shape = _matrix_shape(input_shape)
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = nnt.function(activation, **kwargs)
        self.no_scale = no_scale

        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        output = self.activation(super().forward(input))
        return output

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)


@utils.add_simple_repr
@utils.no_dim_change_op
class BatchNorm2d(nn.BatchNorm2d, _LayerMethod):
    """
    Performs batch normalization on 2D signals.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        input_shape = _image_shape(input_shape)
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = nnt.function(activation, **kwargs)
        self.no_scale = no_scale

        super().__init__(self.input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        output = self.activation(super().forward(input))
        return output

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)


@utils.add_simple_repr
@utils.no_dim_change_op
class LayerNorm(nn.LayerNorm, _LayerMethod):
    """
    Performs layer normalization on input tensor.

    Parameters
    ----------
    input_shape
        input shape from an expected input of size

        .. math::
            [\\text{input_shape}[0] \\times \\text{input_shape}[1]
                \\times \\ldots \\times \\text{input_shape}[-1]]

        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.
    eps
        a value added to the denominator for numerical stability. Default: 1e-5.
    elementwise_affine
        a boolean value that when set to ``True``, this module
        has learnable per-element affine parameters initialized to ones (for weights)
        and zeros (for biases). Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, eps=1e-5, elementwise_affine=True, activation=None, **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)
        assert None not in input_shape[1:], 'All dims in input_shape must be specified except the first dim'
        self.input_shape = _matrix_shape(input_shape)
        self.activation = nnt.function(activation, **kwargs)
        super().__init__(input_shape[1:], eps, elementwise_affine)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


@utils.add_simple_repr
@utils.no_dim_change_op
class InstanceNorm1d(nn.InstanceNorm1d, _LayerMethod):
    """
    Performs instance normalization on 1D signals.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """
    def __init__(self, input_shape, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, activation=None,
                 **kwargs):
        input_shape = _matrix_shape(input_shape)
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = nnt.function(activation, **kwargs)
        super().__init__(input_shape[-1], eps, momentum, affine, track_running_stats)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


@utils.add_simple_repr
@utils.no_dim_change_op
class InstanceNorm2d(nn.InstanceNorm2d, _LayerMethod):
    """
    Performs instance normalization on 2D signals.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """
    def __init__(self, input_shape, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, activation=None,
                 **kwargs):
        input_shape = _image_shape(input_shape)
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = nnt.function(activation, **kwargs)
        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


class GroupNorm(nn.GroupNorm, _LayerMethod):
    """
        Performs instance normalization on 2D signals.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    num_groups : int
        number of channels expected in input
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """
    def __init__(self, input_shape, num_groups, eps=1e-5, affine=True, activation=None, **kwargs):
        input_shape = _image_shape(input_shape)
        assert input_shape[1] is not None, 'Dimension at index 1 (index starts at 0) must be specified'

        self.input_shape = input_shape
        self.activation = nnt.function(activation, **kwargs)
        super().__init__(num_groups, input_shape[1], eps, affine)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


class AdaIN(MultiSingleInputModule):
    """
    Adaptive Instance Normalization from https://arxiv.org/abs/1703.06868.

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
        dimension to reduce in the target feature maps
    dim2
        dimension to reduce in the style feature maps
    """

    def __init__(self, module1, module2, dim1=1, dim2=1):
        super().__init__(module1, module2)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input, *args, **kwargs):
        out1, out2 = super().forward(input)
        mean1, std1 = T.mean(out1, self.dim1, keepdim=True), T.sqrt(T.var(out1, self.dim1, keepdim=True) + 1e-8)
        mean2, std2 = T.mean(out2, self.dim2, keepdim=True), T.sqrt(T.var(out2, self.dim2, keepdim=True) + 1e-8)
        return std2 * (out1 - mean1) / std1 + mean2

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0]

    def extra_repr(self):
        s = 'dim1={dim1}, dim2={dim2}'.format(**self.__dict__)
        return s


class MultiInputAdaIN(MultiMultiInputModule):
    """
    Adaptive Instance Normalization from https://arxiv.org/abs/1703.06868.

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
        dimension to reduce in the target feature maps
    dim2
        dimension to reduce in the style feature maps
    """

    def __init__(self, module1, module2, dim1=1, dim2=1):
        super().__init__(module1, module2)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, *input):
        out1, out2 = super().forward(*input)
        mean1, std1 = T.mean(out1, self.dim1, keepdim=True), T.sqrt(T.var(out1, self.dim1, keepdim=True) + 1e-8)
        mean2, std2 = T.mean(out2, self.dim2, keepdim=True), T.sqrt(T.var(out2, self.dim2, keepdim=True) + 1e-8)
        return std2 * (out1 - mean1) / std1 + mean2

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0]

    def extra_repr(self):
        s = 'dim1={dim1}, dim2={dim2}'.format(**self.__dict__)
        return s


@utils.add_simple_repr
@utils.no_dim_change_op
class FeatureNorm1d(nn.BatchNorm1d, _LayerMethod):
    """
    Performs batch normalization over the last dimension of the input.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :const:`~neuralnet_pytorch.utils.function`.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        input_shape = _pointset_shape(input_shape)
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = nnt.function(activation, **kwargs)
        self.no_scale = no_scale

        super().__init__(input_shape[-1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = self.activation(super().forward(input))
        output = output.view(*shape)
        return output

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
