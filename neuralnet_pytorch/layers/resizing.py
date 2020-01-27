import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _ntuple

from .. import utils
from ..utils import _image_shape
from .abstract import _LayerMethod, Module, MultiSingleInputModule, MultiMultiInputModule

__all__ = ['Interpolate', 'AvgPool2d', 'MaxPool2d', 'Cat', 'Reshape', 'Flatten', 'DimShuffle', 'GlobalAvgPool2D',
           'ConcurrentCat', 'SequentialCat']


@utils.add_custom_repr
class Interpolate(Module):
    """
    Down/Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    Parameters
    ----------
    size
        output spatial sizes. Mutually exclusive with :attr:`scale_factor`.
    scale_factor
        float or tuple of floats.
        Multiplier for spatial size. Has to match input size if it is a tuple.
        Mutually exclusive with :attr:`size`.
    mode
        talgorithm used for upsampling:
        ``'nearest'``, ``'linear'``, ``'bilinear'``,
        ``'bicubic'``, ``'trilinear'``, and ``'area'``.
        Default: ``'nearest'``.
    align_corners
        if ``True``, the corner pixels of the input
        and output tensors are aligned, and thus preserving the values at
        those pixels.
        If ``False``, the input and output tensors are aligned by the corner
        points of their corner pixels, and the interpolation uses edge value padding
        for out-of-boundary values, making this operation *independent* of input size
        when :attr:`scale_factor` is kept the same.
        This only has effect when `mode` is
        ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=None, input_shape=None):
        assert size is not None or scale_factor is not None, 'size and scale_factor cannot be not None'
        super().__init__(input_shape)

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input: T.Tensor, *args, **kwargs):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        scale_factor = _ntuple(len(self.input_shape) - 2)(self.scale_factor)
        shape = [np.nan if s is None else s for s in self.input_shape]
        return (tuple(shape[:2]) + tuple(self.size)) if self.size \
            else shape[:2] + [shape[i + 2] * scale_factor[i] for i in range(len(scale_factor))]

    def extra_repr(self):
        s = 'size={size}, scale_factor={scale_factor}, mode={mode}, align_corners={align_corners}, ' \
            'input_shape={input_shape}'
        s = s.format(**self.__dict__)
        return s


class _Pool2d(_LayerMethod):

    @property
    @utils.validate
    def output_shape(self):
        if all(self.input_shape) is None:
            return tuple(self.input_shape)

        integize = np.ceil if self.ceil_mode else np.floor
        shape = [np.nan if x is None else x for x in self.input_shape]
        ks = [1 + (fs - 1) * d for fs, d in zip(self.kernel_size, self.dilation)]
        shape[2] = integize((shape[2] + 2 * self.padding[0] - ks[0]) / self.stride[0] + 1)
        shape[3] = integize((shape[3] + 2 * self.padding[1] - ks[1]) / self.stride[1] + 1)
        return shape


@utils.add_simple_repr
class AvgPool2d(nn.AvgPool2d, _Pool2d):
    """
    Applies a 2D average pooling over an input signal composed of several input
    planes.

    Parameters
    ----------
        kernel_size
            the size of the window.
        stride
            the stride of the window. Default value is `kernel_size`.
        padding
            implicit zero padding to be added on both sides.
        ceil_mode
            when True, will use `ceil` instead of `floor` to compute the output shape.
        count_include_pad
            when True, will include the zero-padding in the averaging calculation.
        input_shape
            shape of the input image. Optional.
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=False,
                 input_shape=None):
        input_shape = _image_shape(input_shape)
        self.input_shape = input_shape
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else tuple(stride)
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (1, 1)
        super().__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                         count_include_pad=count_include_pad)

    def forward(self, input: T.Tensor, *args, **kwargs):
        return super().forward(input)


@utils.add_simple_repr
class MaxPool2d(nn.MaxPool2d, _Pool2d):
    """
    Applies a 2D max pooling over an input signal composed of several input
    planes.

    Parameters
    ----------
        kernel_size
            the size of the window.
        stride
            the stride of the window. Default value is `kernel_size`.
        padding
            implicit zero padding to be added on both sides.
        dilation
            a parameter that controls the stride of elements in the window.
        return_indices
            if ``True``, will return the max indices along with the outputs.
            Useful for :class:`torch.nn.MaxUnpool2d` later.
        ceil_mode
            when True, will use `ceil` instead of `floor` to compute the output shape.
        input_shape
            shape of the input image. Optional.
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,
                 input_shape=None):
        input_shape = _image_shape(input_shape)
        self.input_shape = input_shape
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else tuple(stride)
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        super().__init__(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices,
                         ceil_mode=ceil_mode)

    def forward(self, input: T.Tensor, *args, **kwargs):
        return super().forward(input)


@utils.add_custom_repr
class GlobalAvgPool2D(Module):
    """
    Applies a 2D global average pooling over an input signal composed of several input
    planes.

    Parameters
    ----------
    keepdim : bool
        whether to keep the collapsed dim as (1, 1). Default: ``False``.
    input_shape
        shape of the input image. Optional.
    """

    def __init__(self, keepdim=False, input_shape=None):
        input_shape = _image_shape(input_shape)
        super().__init__(input_shape)
        self.keepdim = keepdim

    def forward(self, input: T.Tensor, *args, **kwargs):
        out = F.avg_pool2d(input, input.shape[2:], count_include_pad=True)
        return out if self.keepdim else T.flatten(out, 1)

    @property
    @utils.validate
    def output_shape(self):
        if all(self.input_shape) is None:
            return tuple(self.input_shape)

        output_shape = tuple(self.input_shape[:2])
        return output_shape if not self.keepdim else output_shape + (1, 1)

    def extra_repr(self):
        s = 'keepdim={keepdim}'.format(**self.__dict__)
        return s


class Cat(MultiSingleInputModule):
    """
    Concatenates the outputs of multiple modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, dim=1, *modules_or_tensors):
        super().__init__(*modules_or_tensors)
        self.dim = dim

    def forward(self, input, *args, **kwargs):
        outputs = super().forward(input)
        return T.cat(outputs, dim=self.dim)

    @property
    @utils.validate
    def output_shape(self):
        if None in self.input_shape:
            return None

        shape_nan = [[np.nan if x is None else x for x in self.input_shape[i]] for i in range(len(self.input_shape))]
        depth = sum([shape_nan[i][self.dim] for i in range(len(self.input_shape))])

        shapes_transposed = [item for item in zip(*self.input_shape)]
        shape = list(map(utils.get_non_none, shapes_transposed))
        shape[self.dim] = depth
        return tuple(shape)

    def extra_repr(self):
        s = 'dim={}'.format(self.dim)
        return s


class SequentialCat(Cat):
    """
    Concatenates the intermediate outputs of multiple sequential modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.resizing.Cat`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

    def __init__(self, dim=1, *modules_or_tensors):
        super().__init__(dim, *modules_or_tensors)

    def forward(self, input, *args, **kwargs):
        outputs = []
        output = input
        for name, module in self.named_children():
            if name.startswith('tensor'):
                outputs.append(module())
            else:
                output = module(output)
                outputs.append(output)

        return T.cat(outputs, dim=self.dim)


class ConcurrentCat(MultiMultiInputModule):
    """
    Concatenates the outputs of multiple modules given input tensors.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    """

    def __init__(self, dim=1, *modules_or_tensors):
        super().__init__(*modules_or_tensors)
        self.dim = dim

    def forward(self, *input, **kwargs):
        outputs = super().forward(*input)
        return T.cat(outputs, dim=self.dim)

    @property
    @utils.validate
    def output_shape(self):
        if None in self.input_shape:
            return None

        shape_nan = [[np.nan if x is None else x for x in self.input_shape[i]] for i in range(len(self.input_shape))]
        depth = sum([shape_nan[i][self.dim] for i in range(len(self.input_shape))])

        shapes_transposed = [item for item in zip(*self.input_shape)]
        shape = list(map(utils.get_non_none, shapes_transposed))
        shape[self.dim] = depth
        return tuple(shape)

    def extra_repr(self):
        s = 'dim={}'.format(self.dim)
        return s


class Reshape(Module):
    """
    Reshapes the input tensor to the specified shape.

    Parameters
    ----------
    shape
        new shape of the tensor. One dim can be set to -1
        to let :mod:`torch` automatically calculate the suitable
        value.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, shape, input_shape=None):
        super().__init__(input_shape)
        self.new_shape = shape

    def forward(self, input: T.Tensor, *args, **kwargs):
        return T.reshape(input, self.new_shape)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        if -1 in self.new_shape:
            if self.new_shape[0] == -1 and len(self.new_shape) == len(self.input_shape):
                output = list(self.new_shape)
                output[0] = None
                return tuple(output)
            else:
                shape = [np.nan if x is None else x for x in self.input_shape]
                prod_shape = np.prod(shape)
                prod_new_shape = np.prod(self.new_shape) * -1
                output_shape = [x if x != -1 else prod_shape / prod_new_shape for x in self.new_shape]
                return tuple(output_shape)
        else:
            return tuple(self.new_shape)

    def extra_repr(self):
        s = 'new_shape={new_shape}'.format(**self.__dict__)
        return s


class Flatten(Module):
    """
    Collapses some adjacent dims.

    Parameters
    ----------
    start_dim
        dim where flattening starts.
    end_dim
        dim where flattening ends.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, start_dim=0, end_dim=-1, input_shape=None):
        super().__init__(input_shape)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: T.Tensor, *args, **kwargs):
        return T.flatten(input, self.start_dim, self.end_dim)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        shape = [np.nan if x is None else x for x in self.input_shape]
        start_dim = self.start_dim if self.start_dim != -1 else len(self.input_shape) - 1
        end_dim = self.end_dim if self.end_dim != -1 else len(self.input_shape) - 1
        shape = shape[:start_dim] + [np.prod(shape[start_dim:end_dim + 1])] + shape[end_dim + 1:]
        return shape

    def extra_repr(self):
        s = 'start_dim={start_dim}, end_dim={end_dim}'.format(**self.__dict__)
        return s


class DimShuffle(Module):
    """
    Reorder the dimensions of this variable, optionally inserting
    broadcasted dimensions.
    Inspired by `Theano's dimshuffle`_.

    .. _Theano's dimshuffle:
    https://github.com/Theano/Theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/var.py#L323-L356

    Parameters
    ----------
    pattern
        List/tuple of int mixed with 'x' for broadcastable dimensions.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, pattern, input_shape=None):
        super().__init__(input_shape)
        self.pattern = pattern

    def forward(self, input: T.Tensor, *args, **kwargs):
        return utils.dimshuffle(input, self.pattern)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        return tuple([self.input_shape[i] if i != 'x' else 1 for i in self.pattern])

    def extra_repr(self):
        s = 'pattern={pattern}'.format(**self.__dict__)
        return s
