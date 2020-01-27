import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .abstract import _LayerMethod, Module, Sequential
from .. import utils
from ..utils import _image_shape, _matrix_shape

__all__ = ['Conv2d', 'ConvTranspose2d', 'FC', 'Softmax', 'DepthwiseSepConv2D']


@utils.add_simple_repr
class Conv2d(nn.Conv2d, _LayerMethod):
    """
    Extends :class:`torch.nn.Conv2d` with :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', dilation=1, groups=1,
                 bias=True, activation=None, weights_init=None, bias_init=None, **kwargs):
        input_shape = _image_shape(input_shape)
        assert input_shape[1] is not None, 'Shape at dimension 1 (zero-based index) must be known'

        self.input_shape = input_shape
        kernel_size = _pair(kernel_size)
        self.no_bias = bias
        self.activation = utils.function(activation, **kwargs)
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.border_mode = padding
        dilation = _pair(dilation)
        groups = groups

        self.ks = [fs + (fs - 1) * (d - 1) for fs, d in zip(kernel_size, dilation)]
        if isinstance(padding, str):
            if padding == 'half':
                padding = [k >> 1 for k in self.ks]
            elif padding in ('valid', 'ref', 'rep'):
                padding = (0,) * len(self.ks)
            elif padding == 'full':
                padding = [k - 1 for k in self.ks]
            else:
                raise NotImplementedError
        elif isinstance(padding, int):
            padding = _pair(padding)
        else:
            raise ValueError('padding must be a str/tuple/int, got %s' % type(padding))

        super().__init__(int(input_shape[1]), out_channels, kernel_size, stride, tuple(padding), dilation, bias=bias,
                         groups=groups)

    def forward(self, input, *args, **kwargs):
        if self.border_mode in ('ref', 'rep'):
            padding = (self.ks[1] // 2, self.ks[1] // 2, self.ks[0] // 2, self.ks[0] // 2)
            pad = nn.ReflectionPad2d(padding) if self.border_mode == 'ref' else nn.ReplicationPad2d(padding)
        else:
            pad = lambda x: x  # noqa: E731

        input = pad(input)
        input = self.activation(super().forward(input))
        return input

    @property
    @utils.validate
    def output_shape(self):
        shape = [np.nan if s is None else s for s in self.input_shape]
        padding = (self.ks[0] >> 1, self.ks[1] >> 1) if self.border_mode in ('ref', 'rep') else self.padding
        shape[2:] = [(s - self.ks[idx] + 2 * padding[idx]) // self.stride[idx] + 1 for idx, s in enumerate(shape[2:])]
        shape[1] = self.out_channels
        return tuple(shape)

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


@utils.add_simple_repr
class ConvTranspose2d(nn.ConvTranspose2d, _LayerMethod):
    """
    Extends :class:`torch.nn.ConvTranspose2d` with :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        ``dilation * (kernel_size - 1) - padding`` zero-padding
        will be added to both sides of each dimension in the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'`` which are common padding schemes.
        Default: ``'half'``.
    output_padding
        additional size added to one side of each dimension in the output shape. Default: 0
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    dilation
        spacing between kernel elements. Default: 1.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    output_size
        size of the output tensor. If ``None``, the shape is automatically determined.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', output_padding=0, groups=1,
                 bias=True, dilation=1, activation='linear', weights_init=None, bias_init=None, output_size=None,
                 **kwargs):
        input_shape = _image_shape(input_shape)
        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.activation = utils.function(activation, **kwargs)
        self.output_size = _pair(output_size) if output_size is not None else None

        kernel_size = _pair(kernel_size)
        if isinstance(padding, str):
            if padding == 'half':
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            elif padding == 'valid':
                padding = (0, 0)
            elif padding == 'full':
                padding = (kernel_size[0] - 1, kernel_size[1] - 1)
            else:
                raise NotImplementedError
        elif isinstance(padding, int):
            padding = (padding, padding)

        super().__init__(int(input_shape[1]), out_channels, kernel_size, stride, padding, output_padding, groups, bias,
                         dilation)

    def forward(self, input, output_size=None, *args, **kwargs):
        output = self.activation(super().forward(
            input, output_size=self.output_size if output_size is None else output_size))
        return output

    @property
    @utils.validate
    def output_shape(self):
        if self.output_size is not None:
            assert len(self.output_size) == 2, \
                'output_size should have exactly 2 elements, got %d' % len(self.output_size)
            return (self.input_shape[0], self.out_channels) + tuple(self.output_size)

        shape = [np.nan if s is None else s for s in self.input_shape]
        _, _, h_in, w_in = shape
        h_out = (h_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * \
                (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        w_out = (w_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * \
                (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        return self.input_shape[0], self.out_channels, h_out, w_out

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = 'activation={}'.format(self.activation.__name__)
        return s


@utils.add_simple_repr
class FC(nn.Linear, _LayerMethod):
    """
    AKA fully connected layer in deep learning literature.
    This class extends :class:`torch.nn.Linear` by :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    bias : bool
        if set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    flatten : bool
        whether to flatten input tensor into 2D matrix. Default: ``False``.
    keepdim : bool
        whether to keep the output dimension when `out_features` is 1.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 flatten=False, keepdim=True, **kwargs):
        input_shape = _matrix_shape(input_shape)
        assert input_shape[-1] is not None, 'Shape at the last position (zero-based index) must be known'

        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function(activation, **kwargs)
        super().__init__(int(np.prod(input_shape[1:])) if flatten else input_shape[-1], out_features, bias)

    def forward(self, input, *args, **kwargs):
        if self.flatten:
            input = T.flatten(input, 1)

        output = self.activation(super().forward(input))
        return output.flatten(-2) if self.out_features == 1 and not self.keepdim else output

    @property
    @utils.validate
    def output_shape(self):
        if self.flatten:
            return self.input_shape[0], self.out_features
        else:
            return self.input_shape[:-1] + (self.out_features,)

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


class Softmax(FC):
    """
    A special case of :class:`~neuralnet_pytorch.layers.FC` with softmax activation function.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    dim : int
        dimension to apply softmax. Default: 1.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, dim=1, weights_init=None, bias_init=None, **kwargs):
        self.dim = dim
        kwargs['dim'] = dim
        super().__init__(input_shape, out_features, activation='softmax', weights_init=weights_init,
                         bias_init=bias_init, **kwargs)


@utils.add_custom_repr
class DepthwiseSepConv2D(Sequential):
    """
    Performs depthwise separable convolution in image processing.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size : int
        size of the convolving kernel.
    depth_mul
        depth multiplier for intermediate result of depthwise convolution
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, depth_mul=1, stride=1, padding='half', dilation=1,
                 bias=True, activation=None, **kwargs):
        input_shape = _image_shape(input_shape)

        super().__init__(input_shape=input_shape)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depth_mul = depth_mul
        self.padding = padding
        self.dilation = dilation
        self.activation = utils.function(activation, **kwargs)
        self.depthwise = Conv2d(self.output_shape, input_shape[1] * depth_mul, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=input_shape[1], bias=bias)
        self.pointwise = Conv2d(self.output_shape, out_channels, 1, activation=activation, padding=padding,
                                dilation=dilation, bias=False, **kwargs)

    def extra_repr(self):
        s = ('{input_shape}, {out_channels}, kernel_size={kernel_size}'
             ', depth_mul={depth_mul}')
        if self.padding != 'half':
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s
