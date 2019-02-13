import torch.nn as nn

from neuralnet_pytorch import utils
from neuralnet_pytorch.layers import _NetMethod, cuda_available

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'InstanceNorm2d']


class BatchNorm1d(nn.BatchNorm1d, _NetMethod):
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
        assert None not in input_shape[1:], 'All dims in input_shape must be specified except the first dim'
        self.input_shape = input_shape
        super().__init__(input_shape[1:], eps, elementwise_affine)
        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)


class InstanceNorm2d(nn.InstanceNorm2d, _NetMethod):
    def __init__(self, input_shape, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, **kwargs):
        self.input_shape = input_shape
        super().__init__(input_shape[1], eps, momentum, affine, track_running_stats)
        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def __repr__(self):
        return super().__repr__() + ' -> {}'.format(self.output_shape)
