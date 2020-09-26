import torch.nn as nn
from torch._six import container_abcs

from .. import utils
from ..utils import _image_shape, _matrix_shape, _pointset_shape
from .abstract import _LayerMethod

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'InstanceNorm2d', 'FeatureNorm1d', 'InstanceNorm1d', 'GroupNorm']


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
        assert isinstance(input_shape, container_abcs.Iterable), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = utils.function(activation, **kwargs)
        self.no_scale = no_scale

        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)

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
        assert isinstance(input_shape, container_abcs.Iterable), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = utils.function(activation, **kwargs)
        self.no_scale = no_scale

        super().__init__(self.input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)

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
        assert isinstance(input_shape, container_abcs.Iterable), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)
        assert None not in input_shape[1:], 'All dims in input_shape must be specified except the first dim'
        self.input_shape = _matrix_shape(input_shape)
        self.activation = utils.function(activation, **kwargs)
        super().__init__(input_shape[1:], eps, elementwise_affine)

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
        assert isinstance(input_shape, container_abcs.Iterable), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = utils.function(activation, **kwargs)
        super().__init__(input_shape[-1], eps, momentum, affine, track_running_stats)

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
        assert isinstance(input_shape, container_abcs.Iterable), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = utils.function(activation, **kwargs)
        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)

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
        self.activation = utils.function(activation, **kwargs)
        super().__init__(num_groups, input_shape[1], eps, affine)

    def forward(self, input):
        output = super().forward(input)
        return self.activation(output)


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
        assert isinstance(input_shape, container_abcs.Iterable), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)

        self.input_shape = input_shape
        self.activation = utils.function(activation, **kwargs)
        self.no_scale = no_scale

        super().__init__(input_shape[-1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)

    def forward(self, input, *args, **kwargs):
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = self.activation(super().forward(input))
        output = output.view(*shape)
        return output

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
