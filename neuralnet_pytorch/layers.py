"""
Written by Duc Nguyen (Mar 21, 18)
Updated Jan 13, 2019
"""
from functools import partial
from collections import OrderedDict

import numpy as np
import torch as T
import torch.nn as nn

from neuralnet_pytorch import utils
from neuralnet_pytorch.utils import cuda_available

__all__ = ['Conv2d', 'ConvNormAct', 'ConvTranspose2d', 'StackingConv', 'ResNetBasicBlock', 'FC', 'Wrapper',
           'ResNetBottleneckBlock', 'Activation', 'Sequential', 'Lambda', 'Module', 'Softmax', 'Sum', 'XConv',
           'GraphConv']


class _NetMethod:
    @property
    @utils.validate
    def output_shape(self):
        assert not hasattr(super(), 'output_shape')
        return None if self.input_shape is None else tuple(self.input_shape)

    @property
    def params(self):
        assert not hasattr(super(), 'params')
        return tuple(self.state_dict().values())

    @property
    def trainable(self):
        assert not hasattr(super(), 'trainable')
        return tuple([p for p in self.parameters() if p.requires_grad])

    @property
    def regularizable(self):
        assert not hasattr(super(), 'regularizable')

        if hasattr(self, 'weight'):
            return tuple([self.weight])
        else:
            r = []
            for m in self.children():
                r.extend(m.regularizable if hasattr(m, 'regularizable') else [])
            return tuple(r)

    def save(self, param_file):
        assert not hasattr(super(), 'save')
        params_np = utils.bulk_to_numpy(self.params)
        params_dict = OrderedDict(zip(list(self.state_dict().keys()), params_np))
        T.save(params_dict, param_file)
        print('Model weights dumped to %s' % param_file)

    def load(self, param_file, eval=True):
        assert not hasattr(super(), 'load')
        params_dict = T.load(param_file)
        if cuda_available:
            params_cuda = utils.bulk_to_cuda(params_dict.values())
            params_dict = OrderedDict(zip(list(params_dict.keys()), params_cuda))

        self.load_state_dict(params_dict)
        if eval:
            self.eval()
        print('Model weights loaded from %s' % param_file)

    def reset_parameters(self):
        assert not hasattr(super(), 'reset_parameters')
        pass


class Module(nn.Module, _NetMethod):
    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape


class Sequential(nn.Sequential, _NetMethod):
    def __init__(self, input_shape=None, *args):
        super().__init__(*args)
        self.input_shape = input_shape

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        layers = list(self.children())
        return layers[-1].output_shape if layers else self.input_shape

    @property
    def trainable(self):
        trainable = []
        for m in list(self.children()):
            trainable.extend(m.trainable)
        return tuple(trainable)

    @property
    def regularizable(self):
        regularizable = []
        for m in list(self.children()):
            regularizable.extend(m.regularizable)
        return tuple(regularizable)

    def reset_parameters(self):
        for m in self.children():
            m.reset_parameters()


def Wrapper(input_shape, layer, *args, **kwargs):
    class _Wrapper(layer, _NetMethod):
        def __init__(self):
            self.output_shape_tmp = kwargs.pop('output_shape', None)
            device = kwargs.pop('device', None)
            self.input_shape = input_shape

            super().__init__(*args, **kwargs)
            if cuda_available:
                self.cuda(device)

        @property
        @utils.validate
        def output_shape(self):
            if self.input_shape is None:
                return None

            if self.output_shape_tmp:
                return self.output_shape_tmp
            else:
                none_indices = [k for k in range(len(self.input_shape)) if self.input_shape[k] is None]
                shape = [1 if s is None else s for s in self.input_shape]
                dummy = T.zeros(*shape)
                if cuda_available:
                    dummy.cuda()
                dummy = self(dummy)
                output_shape = list(dummy.shape)
                for k in none_indices:
                    output_shape[k] = None
                return tuple(output_shape)

    return _Wrapper()


class Lambda(Module):
    def __init__(self, input_shape, func, output_shape=None, **kwargs):
        assert callable(func), 'The provided function must be callable'

        super().__init__(input_shape)
        self.output_shape_tmp = output_shape
        self.func = func
        self.kwargs = kwargs

    def forward(self, *input):
        return self.func(*input, **self.kwargs)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        if self.output_shape_tmp:
            return self.output_shape_tmp
        else:
            none_indices = [k for k in range(len(self.input_shape)) if self.input_shape[k] is None]
            shape = [1 if s is None else s for s in self.input_shape]
            dummy = T.zeros(*shape)
            if cuda_available:
                dummy.cuda()
            dummy = self.forward(dummy)
            output_shape = list(dummy.shape)
            for k in none_indices:
                output_shape[k] = None
            return tuple(output_shape)

    def __repr__(self):
        return self.__class__.__name__ + '({}, output_shape={})'.format(self.input_shape, self.output_shape)


class Conv2d(nn.Conv2d, _NetMethod):
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', dilation=1, groups=1,
                 bias=True, activation=None, weights_init=None, bias_init=None, **kwargs):
        assert isinstance(input_shape, list) or isinstance(input_shape,
                                                           tuple), 'input_shape must be list or tuple, got %s' % type(
            input_shape)
        assert len(input_shape) == 4, 'input_shape must have 4 elements, got %d' % len(input_shape)
        assert isinstance(out_channels, int) and isinstance(kernel_size, (int, list, tuple))
        assert isinstance(padding, (int, list, tuple,
                                    str)), 'border_mode should be either \'int\', ' '\'list\', \'tuple\' or \'str\', got {}'.format(
            type(padding))
        assert isinstance(stride, (int, list, tuple)), 'stride must be an int/list/tuple, got %s' % type(stride)

        self.input_shape = input_shape
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.no_bias = bias
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.border_mode = padding
        stride = tuple(stride) if isinstance(stride, (tuple, list)) else (stride, stride)
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        groups = groups
        self.kwargs = kwargs

        self.ks = [fs + (fs - 1) * (d - 1) for fs, d in zip(kernel_size, dilation)]
        if isinstance(padding, str):
            if padding == 'half':
                padding = [k >> 1 for k in self.ks]
            elif padding in ('valid', 'ref', 'rep'):
                padding = [0] * len(self.ks)
            elif padding == 'full':
                padding = [k - 1 for k in self.ks]
            else:
                raise NotImplementedError
        elif isinstance(padding, int):
            padding = (padding, padding)

        super().__init__(int(input_shape[1]), out_channels, kernel_size, stride, padding, dilation, bias=bias,
                         groups=groups)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        pad = nn.ReflectionPad2d(self.padding) if self.border_mode == 'ref' else nn.ReplicationPad2d(
            (self.ks[1] // 2, self.ks[1] // 2, self.ks[0] // 2,
             self.ks[0] // 2)) if self.border_mode == 'rep' else lambda x: x
        input = pad(input)
        input = self.activation(super().forward(input), **self.kwargs)
        return input

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

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


class FC(nn.Linear, _NetMethod):
    def __init__(self, input_shape, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 flatten=False, keepdim=False, **kwargs):
        assert None not in input_shape[1:], 'Shape of input must be known for FC layer'
        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.kwargs = kwargs

        super().__init__(int(np.prod(input_shape[1:])) if flatten else input_shape[-1], out_features, bias)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        if self.flatten:
            input = T.flatten(input, 1)

        output = self.activation(super().forward(input), **self.kwargs)
        return output.flatten() if self.out_features == 1 and not self.keepdim else output

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

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


class Softmax(FC):
    def __init__(self, input_shape, out_features, dim=1, weights_init=None, bias_init=None, **kwargs):
        self.dim = dim
        super().__init__(input_shape, out_features,
                         activation=lambda x, **kwargs: utils.function['softmax'](x, dim=dim, **kwargs),
                         weights_init=weights_init, bias_init=bias_init, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '({}, out_features={}, dim={})'.format(self.input_shape, self.out_features,
                                                                                self.dim)


class Activation(Module):
    def __init__(self, input_shape, activation='relu', **kwargs):
        super().__init__(input_shape)
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.kwargs = kwargs

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        return self.activation(input, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.activation)


class ConvNormAct(Sequential):
    def __init__(self, input_shape, out_channels, kernel_size, weights_init=None, bias=True, bias_init=None,
                 padding='half', stride=1, dilation=1, activation='relu', groups=1, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, no_scale=False, norm_method='bn', **kwargs):
        super().__init__(input_shape)
        from neuralnet_pytorch.normalization import BatchNorm2d
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.norm_method = norm_method
        self.conv = Conv2d(input_shape, out_channels, kernel_size, weights_init=weights_init, bias=bias,
                           bias_init=bias_init, padding=padding, stride=stride, dilation=dilation, activation=None,
                           groups=groups, **kwargs)
        if norm_method == 'bn':
            self.norm = BatchNorm2d(self.conv.output_shape, eps, momentum, affine, track_running_stats,
                                    no_scale=no_scale, activation=self.activation, **kwargs)
        else:
            raise NotImplementedError

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def __repr__(self):
        string = self.__class__.__name__ + '({}, {}, {}, padding={}, stride={}, activation={})'.format(
            self.input_shape, self.out_channels, self.kernel_size, self.padding, self.stride, self.activation)
        return string


class ResNetBasicBlock(Sequential):
    expansion = 1

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, activation='relu', downsample=None, groups=1,
                 block=None, weights_init=None, bias_init=None, norm_method='bn', **kwargs):
        super().__init__(input_shape)
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = utils.function[activation]
        self.groups = groups
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.norm_method = norm_method
        self.kwargs = kwargs

        self.block = self._make_block() if block is None else block()
        if stride > 1 or input_shape[1] != out_channels * self.expansion:
            if downsample:
                self.downsample = downsample
            else:
                self.downsample = ConvNormAct(input_shape, out_channels * self.expansion, 1, stride=stride, bias=False,
                                              weights_init=weights_init, bias_init=bias_init, activation='linear')
            self.add_module('downsample', self.downsample)
        else:
            if downsample:
                self.downsample = downsample
                self.add_module('downsample', self.downsample)
            else:
                self.downsample = lambda x: x

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def _make_block(self):
        block = Sequential(self.input_shape)
        if self.expansion != 1:
            block.add_module('pre',
                             ConvNormAct(block.output_shape, self.out_channels, 1, stride=1, bias=False,
                                         weights_init=self.weights_init, bias_init=self.bias_init, groups=self.groups))
        block.add_module('conv_norm_act_1',
                         ConvNormAct(block.output_shape, self.out_channels, self.kernel_size, bias=False,
                                     weights_init=self.weights_init, bias_init=self.bias_init,
                                     stride=self.stride, activation=self.activation, groups=self.groups,
                                     norm_method=self.norm_method, **self.kwargs))
        block.add_module('conv_norm_act_2',
                         ConvNormAct(block.output_shape, self.out_channels * self.expansion,
                                     1 if self.expansion != 1 else self.kernel_size, bias=False, stride=1,
                                     activation=None, groups=self.groups, weights_init=self.weights_init,
                                     norm_method=self.norm_method, bias_init=self.bias_init, **self.kwargs))
        return block

    def forward(self, input):
        res = input
        out = self.block(input)
        out += self.downsample(res)
        return self.activation(out, **self.kwargs)

    def __repr__(self):
        string = self.__class__.__name__ + '({}, {}, {}, stride={}, activation={})'.format(
            self.input_shape, self.out_channels, self.kernel_size, self.stride, self.activation)
        return string


class ResNetBottleneckBlock(ResNetBasicBlock):
    expansion = 4

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, activation='relu', downsample=None, groups=1,
                 block=None, weights_init=None, bias_init=None, norm_method='bn', **kwargs):
        super().__init__(input_shape, out_channels, kernel_size, stride, activation, downsample, groups, block=block,
                         weights_init=weights_init, bias_init=bias_init, norm_method=norm_method, **kwargs)

    def __repr__(self):
        string = self.__class__.__name__ + '({}, {}, {}, stride={}, activation={})'.format(
            self.input_shape, self.out_channels, self.kernel_size, self.stride, self.activation)
        return string


class ConvTranspose2d(nn.ConvTranspose2d, _NetMethod):
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', output_padding=0, bias=True,
                 dilation=1, weights_init=None, bias_init=None, activation='linear', groups=1, **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple. Received %s.' % type(
            input_shape)
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        stride = tuple(stride) if isinstance(stride, (tuple, list)) else (stride, stride)
        dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.kwargs = kwargs

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

        self.convtrans = nn.ConvTranspose2d(int(input_shape[0]), out_channels, kernel_size, stride, padding,
                                            output_padding, groups, bias, dilation)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, output_size=None):
        output = self.activation(super().forward(input, output_size=output_size), **self.kwargs)
        return output

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        shape = [np.nan if s is None else s for s in self.input_shape]
        h = ((shape[1] - 1) * self.stride[0]) + self.kernel_size[0] + \
            np.mod(shape[1] + 2 * self.padding[0] - self.kernel_size[0], self.stride[0]) - 2 * self.padding[0]
        w = ((shape[2] - 1) * self.stride[1]) + self.kernel_size[1] + \
            np.mod(shape[2] + 2 * self.padding[1] - self.kernel_size[1], self.stride[1]) - 2 * self.padding[1]
        return self.input_shape[0], self.filter_shape[1], h, w

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)


class StackingConv(Sequential):
    def __init__(self, input_shape, out_channels, kernel_size, num_layers, stride=1, padding='half', dilation=1,
                 bias=True, weights_init=None, bias_init=None, norm_method=None, activation='relu', groups=1, **kwargs):
        assert num_layers > 1, 'num_layers must be greater than 1, got %d' % num_layers
        super(StackingConv, self).__init__(input_shape)
        self.num_filters = out_channels
        self.filter_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.num_layers = num_layers
        self.norm_method = norm_method

        self.block = nn.Sequential()
        shape = tuple(input_shape)
        conv_layer = partial(ConvNormAct, norm_method=norm_method) if norm_method else Conv2d
        for num in range(num_layers - 1):
            layer = conv_layer(input_shape=shape, out_channels=out_channels, kernel_size=kernel_size,
                               weights_init=weights_init, bias_init=bias_init, stride=1, padding=padding,
                               dilation=dilation, activation=activation, groups=groups, bias=bias, **kwargs)
            self.add_module('stacking_conv_%d' % (num + 1), layer)
            shape = layer.output_shape
        self.add_module('stacking_conv_%d' % num_layers,
                        conv_layer(input_shape=shape, out_channels=out_channels, bias=bias, groups=groups,
                                   kernel_size=kernel_size, weights_init=weights_init, stride=stride, padding=padding,
                                   dilation=dilation, activation=activation, bias_init=bias_init, **kwargs))

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def __repr__(self):
        string = self.__class__.__name__ + '({}, {}, {}, num_layers={}, stride={}, activation={})'.format(
            self.input_shape, self.num_filters, self.filter_size, self.num_layers, self.stride, self.activation)
        return string


class Sum(Module):
    def __init__(self, input_shapes):
        super().__init__(input_shapes)

    def forward(self, *input):
        return sum(input)

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        return tuple(self.input_shape[0])

    def __repr__(self):
        return self.__class__.__name__ + '({}, output_shape={})'.format(self.input_shape, self.output_shape)


class DepthwiseSepConv2D(Sequential):
    """ Depthwise separable convolution"""

    def __init__(self, input_shape, out_channels, kernel_size, depth_mul, padding='half', activation=None):
        """
        :param input_shape: Shape of input features
        :param out_channels: Length of output features (first dimension)
        :param kernel_size: Size of convolutional kernel
        :param depth_mul: Depth multiplier for middle part of separable convolution
        :param activation: Activation function
        """
        super().__init__(input_shape)
        self.depth_mul = depth_mul
        self.activation = activation
        self.add_module('depthwise', Conv2d(self.output_shape, input_shape[1] * depth_mul, kernel_size, padding=padding,
                                            groups=input_shape[1]))
        self.add_module('pointwise', Conv2d(self.output_shape, out_channels, 1, activation=activation, padding=padding,
                                            bias=False))

    def __repr__(self):
        return self.__class__.__name__ + '({}, {}, {}, depth_mul={}, padding={}, activation={})'.format(
            self.input_shape, self[-1].out_channels, self[0].kernel_size, self.depth_mul, self[0].padding,
            self.activation)


class XConv(Module):
    def __init__(self, input_shape, feature_dim, out_channels, out_features, num_neighbors, depth_mul,
                 activation='relu', dropout=None, bn=True):
        super().__init__(input_shape)
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.out_features = out_features
        self.depth_mul = depth_mul
        self.activation = activation
        self.dropout = dropout
        self.bn = bn

        self.fcs = Sequential(input_shape)
        self.fcs.add_module('fc1', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout1', Wrapper(self.output_shape, T.nn.Dropout2d, p=dropout))
        self.fcs.add_module('fc2', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout2', Wrapper(self.output_shape, T.nn.Dropout2d, p=dropout))

        from neuralnet_pytorch.resizing import DimShuffle
        from neuralnet_pytorch.normalization import BatchNorm2d

        self.x_trans = Sequential(input_shape[:2] + (num_neighbors, input_shape[-1]))
        self.x_trans.add_module('dimshuffle1', DimShuffle(self.x_trans.output_shape, (0, 3, 1, 2)))
        self.x_trans.add_module('conv', Conv2d(self.x_trans.output_shape, num_neighbors ** 2, (1, num_neighbors),
                                               activation=activation, padding='valid'))
        self.x_trans.add_module('dimshuffle2', DimShuffle(self.x_trans.output_shape, (0, 2, 3, 1)))
        self.x_trans.add_module('fc1', FC(self.x_trans.output_shape, num_neighbors ** 2, activation='relu'))
        self.x_trans.add_module('fc2', FC(self.x_trans.output_shape, num_neighbors ** 2))

        self.end_conv = Sequential(input_shape[:2] + (num_neighbors, feature_dim + out_features))
        self.end_conv.add_module('dimshuffle1', DimShuffle(self.end_conv.output_shape, (0, 3, 1, 2)))
        self.end_conv.add_module('conv',
                                 DepthwiseSepConv2D(self.end_conv.output_shape, out_channels, (1, num_neighbors),
                                                    depth_mul=depth_mul, activation=None if bn else activation,
                                                    padding='valid'))
        if bn:
            self.end_conv.add_module('bn', BatchNorm2d(self.end_conv.output_shape, momentum=.9, activation=activation))
        self.end_conv.add_module('dimshuffle2', DimShuffle(self.end_conv.output_shape, (0, 2, 3, 1)))

    def forward(self, *input):
        rep_pt, pts, fts = input

        if fts is not None:
            assert rep_pt.size()[0] == pts.size()[0] == fts.size()[0]  # Check N is equal.
            assert rep_pt.size()[1] == pts.size()[1] == fts.size()[1]  # Check P is equal.
            assert pts.size()[2] == fts.size()[2] == self.num_neighbors  # Check K is equal.
            assert fts.size()[3] == self.feature_dim  # Check C_in is equal.
        else:
            assert rep_pt.size()[0] == pts.size()[0]  # Check N is equal.
            assert rep_pt.size()[1] == pts.size()[1]  # Check P is equal.
            assert pts.size()[2] == self.num_neighbors  # Check K is equal.
        assert rep_pt.size()[2] == pts.size()[3] == self.input_shape[-1]  # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = T.unsqueeze(rep_pt, dim=2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center  # (N, P, K, dims)

        # Individually lift each point into C_mid space.
        fts_lifted = self.fcs(pts_local)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = T.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.num_neighbors, self.num_neighbors)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        # Weight and permute fts_cat with the learned X.
        fts_X = T.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim=2)
        return fts_p

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[:2] + (self.out_channels,)

    def __repr__(self):
        return self.__class__.__name__ + \
               '({}, feature_dim={}, out_channels={}, out_features={}, num_neighbors={}, depth_mul={}, ' \
               'activation={}, dropout={}, bn={})'.format(
                   self.input_shape, self.feature_dim, self.out_channels, self.out_features, self.num_neighbors,
                   self.depth_mul, self.activation, self.dropout, self.bn)


class GraphConv(FC):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None):
        self.input_shape = input_shape
        super().__init__(input_shape, out_features, bias, activation=activation)

    def reset_parameters(self):
        if self.weights_init is None:
            stdv = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weights_init(self.weight)

        if self.bias is not None:
            if self.bias_init is None:
                self.bias.data.uniform_(-stdv, stdv)
            else:
                self.bias_init(self.bias)

    def forward(self, input, adj):
        support = T.mm(input, self.weight.t())
        output = T.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '({}, {})'.format(self.input_shape, self.out_features)
