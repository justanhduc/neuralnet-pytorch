import torch as T
import torch.nn as nn

from neuralnet_pytorch import utils
from neuralnet_pytorch.layers import _NetMethod, MultiMultiInputModule, MultiSingleInputModule
from neuralnet_pytorch.utils import cuda_available

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'InstanceNorm2d', 'AdaIN', 'MultiInputAdaIN']


class BatchNorm1d(nn.BatchNorm1d, _NetMethod):
    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)
        self.input_shape = input_shape
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.no_scale = no_scale
        self.kwargs = kwargs

        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)
            self.weight.requires_grad_(False)

    def forward(self, input):
        input = self.activation(super().forward(input), **self.kwargs)
        return input

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class BatchNorm2d(nn.BatchNorm2d, _NetMethod):
    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)
        self.input_shape = input_shape
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.no_scale = no_scale
        self.kwargs = kwargs

        super().__init__(self.input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        input = self.activation(super().forward(input), **self.kwargs)
        return input

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class LayerNorm(nn.LayerNorm, _NetMethod):
    def __init__(self, input_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)
        assert None not in input_shape[1:], 'All dims in input_shape must be specified except the first dim'
        self.input_shape = input_shape
        super().__init__(input_shape[1:], eps, elementwise_affine)
        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class InstanceNorm2d(nn.InstanceNorm2d, _NetMethod):
    def __init__(self, input_shape, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s' % type(
            input_shape)
        self.input_shape = input_shape
        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class AdaIN(MultiSingleInputModule):
    '''
    Adaptive Instance Normalization
    out1 = module1(input)
    out2 = module2(input)
    out = std(out2) * (out1 - mean(out1)) / std(out1) + mean(out2)
    '''
    def __init__(self, module1, module2, dim1=1, dim2=1):
        super().__init__(module1, module2)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        out1, out2 = super().forward(input)
        mean1, std1 = T.mean(out1, self.dim1, keepdim=True), T.sqrt(T.var(out1, self.dim1, keepdim=True) + 1e-8)
        mean2, std2 = T.mean(out2, self.dim2, keepdim=True), T.sqrt(T.var(out2, self.dim2, keepdim=True) + 1e-8)
        return std2 * (out1 - mean1) / std1 + mean2

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0]


class MultiInputAdaIN(MultiMultiInputModule):
    '''
    Adaptive Instance Normalization
    out1 = module1(input1)
    out2 = module2(input2)
    out = std(out2) * (out1 - mean(out1)) / std(out1) + mean(out2)
    '''

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
