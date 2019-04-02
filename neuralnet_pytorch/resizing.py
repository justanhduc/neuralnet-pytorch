import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F

from neuralnet_pytorch import utils
from neuralnet_pytorch.layers import _NetMethod, Module, cuda_available, MultiInputModule

__all__ = ['UpsamplingLayer', 'AvgPool2d', 'MaxPool2d', 'Cat', 'Reshape', 'Flatten', 'DimShuffle', 'GlobalAvgPool2D']


class UpsamplingLayer(nn.Upsample, _NetMethod):
    def __init__(self, input_shape, size=None, scale_factor=None, mode='bilinear', align_corners=True, **kwargs):
        assert isinstance(scale_factor, (int, list, tuple)), 'scale_factor must be an int, a list or a tuple. ' \
                                                             'Received %s.' % type(scale_factor)

        self.input_shape = input_shape
        scale_factor = ((scale_factor,) * (len(input_shape) - 2)) if isinstance(scale_factor, int) else tuple(
            scale_factor)
        super().__init__(size, scale_factor, mode, align_corners)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        shape = [np.nan if s is None else s for s in self.input_shape]
        return (shape[:2] + self.new_shape) if self.size \
            else shape[:2] + [shape[i + 2] * self.scale_factor[i] for i in range(len(self.scale_factor))]

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class AvgPool2d(nn.AvgPool2d, _NetMethod):
    def __init__(self, input_shape, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=False):
        self.input_shape = input_shape
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else tuple(stride)
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        super().__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                         count_include_pad=count_include_pad)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

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

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class MaxPool2d(nn.MaxPool2d, _NetMethod):
    def __init__(self, input_shape, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        self.input_shape = input_shape
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = kernel_size if stride is None else (stride, stride) if isinstance(stride, int) else tuple(stride)
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        super().__init__(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices,
                         ceil_mode=ceil_mode)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

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

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class GlobalAvgPool2D(Module):
    def __init__(self, input_shape, keepdim=False):
        super().__init__(input_shape)
        self.keepdim = keepdim

    def forward(self, input):
        out = F.avg_pool2d(input, input.shape[2:], count_include_pad=True)
        return out if self.keepdim else T.flatten(out, 1)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        output_shape = tuple(self.input_shape[:2])
        return output_shape if not self.keepdim else output_shape + (1, 1)

    def __str__(self):
        return self.__class__.__name__ + '({}) -> {}'.format(self.input_shape, self.output_shape)


class Cat(MultiInputModule):
    def __init__(self, dim=1, *modules):
        super().__init__(*modules)
        self.dim = dim

    def forward(self, input):
        outputs = super().forward(input)
        return T.cat(outputs, dim=self.dim)

    @property
    @utils.validate
    def output_shape(self):
        if None in self.input_shape:
            return None

        shape_nan = [[np.nan if x is None else x for x in self.input_shape[i]] for i in range(len(self.input_shape))]
        depth = sum([shape_nan[i][self.dim] for i in range(len(self.input_shape))])
        shape = list(shape_nan[0])
        shape[self.dim] = depth
        return tuple(shape)

    def __repr__(self):
        return self.__class__.__name__ + '({}, dim={}) -> {}'.format(self.input_shape, self.dim, self.output_shape)


class Reshape(Module):
    def __init__(self, input_shape, shape):
        super().__init__(input_shape)
        self.new_shape = shape

    def forward(self, input):
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

    def __repr__(self):
        return self.__class__.__name__ + '({}, new_shape={}) -> {}'.format(
            self.input_shape, self.new_shape, self.output_shape)


class Flatten(Module):
    def __init__(self, input_shape, start_dim=0, end_dim=-1):
        super().__init__(input_shape)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
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

    def __repr__(self):
        return self.__class__.__name__ + '({}, start_dim={}, end_dim={}) -> {}'.format(
            self.input_shape, self.start_dim, self.end_dim, self.output_shape)


class DimShuffle(Module):
    def __init__(self, input_shape, pattern):
        super().__init__(input_shape)
        self.pattern = pattern

    def forward(self, input):
        return utils.dimshuffle(input, self.pattern)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        return tuple([self.input_shape[i] if i != 'x' else 1 for i in self.pattern])

    def __repr__(self):
        return self.__class__.__name__ + '({}, pattern={}) -> {}'.format(
            self.input_shape, self.pattern, self.output_shape)
