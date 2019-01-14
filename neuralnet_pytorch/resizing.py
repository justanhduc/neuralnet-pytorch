import numpy as np
from torch import nn

from neuralnet_pytorch import utils
from neuralnet_pytorch.layers import Layer, cuda_available

__all__ = ['UpsamplingLayer', 'AvgPool2d', 'MaxPool2d']


class UpsamplingLayer(nn.Upsample, Layer):
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
        shape = [np.nan if s is None else s for s in self.input_shape]
        return (shape[:2] + self.new_shape) if self.new_shape \
            else shape[:2] + [shape[i + 2] * self.scale_factor[i] for i in range(len(self.scale_factor))]


class AvgPool2d(nn.AvgPool2d, Layer):
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


class MaxPool2d(nn.MaxPool2d, Layer):
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
