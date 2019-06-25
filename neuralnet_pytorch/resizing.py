import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from neuralnet_pytorch import utils
from neuralnet_pytorch.utils import _image_shape
from neuralnet_pytorch.layers import _LayerMethod, Module, MultiSingleInputModule, MultiMultiInputModule

__all__ = ['Upsample', 'AvgPool2d', 'MaxPool2d', 'Cat', 'Reshape', 'Flatten', 'DimShuffle', 'GlobalAvgPool2D',
           'ConcurrentCat', 'SequentialCat']


@utils.add_simple_repr
class Upsample(nn.Upsample, _LayerMethod):
    """
    Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    Parameters
    ----------
    size
        output spatial sizes. Optional.
    scale_factor
        multiplier for spatial size. Has to match input size if it is a tuple. Optional.
    mode
        the upsampling algorithm: one of ``'nearest'``,
        ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
        Default: ``'nearest'``.
    align_corners
        if ``True``, the corner pixels of the input
        and output tensors are aligned, and thus preserving the values at
        those pixels. This only has effect when `mode` is
        ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``.
    input_shape
        shape of the input tensor. Optional.
    """

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False, input_shape=None):
        assert isinstance(scale_factor, (int, list, tuple)), 'scale_factor must be an int, a list or a tuple. ' \
                                                             'Received %s.' % type(scale_factor)

        self.input_shape = input_shape
        scale_factor = _pair(scale_factor)
        super().__init__(size, scale_factor, mode, align_corners)

    def forward(self, input: T.Tensor, *args, **kwargs):
        return super().forward(input)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        shape = [np.nan if s is None else s for s in self.input_shape]
        return (shape[:2] + self.new_shape) if self.size \
            else shape[:2] + [shape[i + 2] * self.scale_factor[i] for i in range(len(self.scale_factor))]


@utils.add_simple_repr
class AvgPool2d(nn.AvgPool2d, _LayerMethod):
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

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=False, input_shape=None):
        input_shape = _image_shape(input_shape)
        self.input_shape = input_shape
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else tuple(stride)
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        super().__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                         count_include_pad=count_include_pad)

    def forward(self, input: T.Tensor, *args, **kwargs):
        return super().forward(input)

    @property
    @utils.validate
    def output_shape(self):
        if all(self.input_shape) is None:
            return tuple(self.input_shape)

        shape = [np.nan if x is None else x for x in self.input_shape]
        shape[2] = (shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        shape[3] = (shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        rem_h = np.mod(shape[2], self.stride[0])
        if rem_h and not np.isnan(rem_h):
            if self.ceil_mode:
                shape[2] += np.mod(shape[2], self.stride[0])

        rem_w = np.mod(shape[3], self.stride[1])
        if rem_w and not np.isnan(rem_w):
            if self.ceil_mode:
                shape[3] += np.mod(shape[3], self.stride[1])
        return tuple(shape)


@utils.add_simple_repr
class MaxPool2d(nn.MaxPool2d, _LayerMethod):
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

    @property
    @utils.validate
    def output_shape(self):
        if all(self.input_shape) is None:
            return tuple(self.input_shape)

        shape = [np.nan if x is None else x for x in self.input_shape]
        ks = [fs + (fs - 1) * (d - 1) for fs, d in zip(self.kernel_size, self.dilation)]
        shape[2] = (shape[2] + 2 * self.padding[0] - ks[0]) // self.stride[0] + 1
        shape[3] = (shape[3] + 2 * self.padding[1] - ks[1]) // self.stride[1] + 1

        rem_h = np.mod(shape[2], self.stride[0])
        if rem_h and not np.isnan(rem_h):
            if self.ceil_mode:
                shape[2] += np.mod(shape[2], self.stride[0])

        rem_w = np.mod(shape[3], self.stride[1])
        if rem_w and not np.isnan(rem_w):
            if self.ceil_mode:
                shape[3] += np.mod(shape[3], self.stride[1])
        return tuple(shape)


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

    .. _Theano's dimshuffle: https://github.com/Theano/Theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/var.py#L323-L356

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
