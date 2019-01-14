import torch.nn as nn

from neuralnet_pytorch import utils
from neuralnet_pytorch.layers import Layer, cuda_available

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'InstanceNorm2d']


class BatchNorm1d(nn.BatchNorm1d, Layer):
    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
        self.input_shape = input_shape
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.no_scale = no_scale
        self.kwargs = kwargs

        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)

    def forward(self, input):
        input = self.activation(super().forward(input), **self.kwargs)
        return input

    def reset(self):
        super().reset_parameters()
        if self.no_scale:
            nn.init.constant_(self.weight, 1.)


class BatchNorm2d(nn.BatchNorm2d, Layer):
    def __init__(self, input_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, **kwargs):
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


class LayerNorm(nn.LayerNorm, Layer):
    def __init__(self, input_shape, eps=1e-5, elementwise_affine=True):
        assert None not in input_shape[1:], 'All dims in input_shape must be specified except the first dim'
        self.input_shape = input_shape
        super().__init__(input_shape[1:], eps, elementwise_affine)


class InstanceNorm2d(nn.InstanceNorm2d, Layer):
    def __init__(self, input_shape, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False):
        self.input_shape = input_shape
        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
