"""
Written by Duc Nguyen (Mar 21, 18)
"""
__author__ = 'Duc Nguyen'

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc

from . import utils


def validate(func):
    """make sure output shape is a list of ints"""
    def func_wrapper(self):
        out = [int(x) if x is not None else x for x in func(self)]
        return tuple(out)
    return func_wrapper


class Layer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Layer, self).__init__()
        self.trainable = []
        self.regularizable = []

    @property
    @abc.abstractclassmethod
    def output_shape(self):
        return

    @abc.abstractclassmethod
    def forward(self, input):
        raise NotImplementedError

    @abc.abstractclassmethod
    def reset(self):
        raise NotImplementedError


class WrapperLayer(Layer):
    def __init__(self, pytorch_layer, input_shape):
        super(WrapperLayer, self).__init__()
        self._wrapped_layer = pytorch_layer
        self._forward = pytorch_layer.forward
        self._train = pytorch_layer.train
        self._eval = pytorch_layer.eval
        self.input_shape = input_shape
        if hasattr(pytorch_layer, 'weight'):
            self.trainable += [pytorch_layer.weight]
            self.regularizable += [pytorch_layer.weight]
        if hasattr(pytorch_layer, 'bias'):
            self.trainable += [pytorch_layer.bias]
        print(pytorch_layer)

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)
        return getattr(self._wrapped_layer, item)

    def forward(self, input):
        input = self._forward(input)
        return input

    @property
    def output_shape(self):
        shape = [1] + list(self.input_shape) if isinstance(self.input_shape, (list, tuple)) else [1, self.input_shape]
        dummy = F.Variable(T.zeros(*shape))
        dummy = self._forward(dummy)
        shape = dummy.shape[1:] if dummy.ndimension() == 4 else dummy.shape[1]
        return tuple(shape)

    def train(self, mode=True):
        self._train(mode)

    def eval(self):
        self._eval()

    def reset(self):
        pass


class Conv2DLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, init_mode='Xavier_normal', no_bias=True,
                 border_mode='half', stride=(1, 1), dilation=(1, 1), activation='relu', groups=1, show=True, **kwargs):
        """
        :param input_shape:
        :param num_filters:
        :param filter_size:
        :param init_mode: Xavier_normal, Xavier_uniform, He_normal, He_uniform
        :param no_bias:
        :param border_mode:
        :param stride:
        :param dilation:
        :param activation:
        :param groups:
        :param args:
        """
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 3, \
            'input_shape must have 3 elements. Received %d' % len(input_shape)
        assert isinstance(num_filters, int) and isinstance(filter_size, (int, list, tuple))
        assert isinstance(border_mode, (int, list, tuple, str)), 'border_mode should be either \'int\', ' \
                                                                 '\'list\', \'tuple\' or \'str\', got {}'.format(type(border_mode))
        assert isinstance(stride, (int, list, tuple))
        super(Conv2DLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.filter_shape = (num_filters, input_shape[0], filter_size[0], filter_size[1]) if isinstance(filter_size, (list, tuple)) \
            else (num_filters, input_shape[0], filter_size, filter_size)
        self.no_bias = no_bias
        self.activation = utils.function[activation]
        self.init_mode = init_mode
        self.border_mode = border_mode
        self.stride = tuple(stride) if isinstance(stride, (tuple, list)) else (stride, stride)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.kwargs = kwargs

        k1, k2 = self.filter_shape[2] + (self.filter_shape[2] - 1)*(self.dilation[0] - 1), \
                 self.filter_shape[3] + (self.filter_shape[3] - 1)*(self.dilation[1] - 1)
        if isinstance(self.border_mode, str):
            if self.border_mode == 'half':
                self.padding = (k1 // 2, k2 // 2)
            elif self.border_mode == 'valid':
                self.padding = (0, 0)
            elif self.border_mode == 'full':
                self.padding = (k1 - 1, k2 - 1)
            else:
                raise NotImplementedError
        elif isinstance(self.border_mode, (list, tuple)):
            self.padding = tuple(self.border_mode)
        elif isinstance(self.border_mode, int):
            self.padding = (self.border_mode, self.border_mode)
        else:
            raise NotImplementedError

        self.conv = nn.Conv2d(int(input_shape[0]), num_filters, filter_size, stride, self.padding, dilation, bias=not no_bias, groups=groups)
        params = list(self.conv.parameters())
        self.trainable = list(params)
        self.regularizable = [params[0]]

        utils.init[init_mode](self.conv.weight)
        self.W_values = self.conv.weight.data.numpy().copy()
        if not self.no_bias:
            self.b_values = np.zeros(self.filter_shape[0], dtype='float32')
            self.conv.bias.data = T.from_numpy(self.b_values)

        if show:
            print(self)

    def forward(self, input):
        input = self.activation(self.conv(input), **self.kwargs)
        return input

    @property
    @validate
    def output_shape(self):
        size = list(self.input_shape)
        assert len(size) == 3, "Shape must consist of 3 elements only"

        k1, k2 = self.filter_shape[2] + (self.filter_shape[2] - 1)*(self.dilation[0] - 1), \
                 self.filter_shape[3] + (self.filter_shape[3] - 1)*(self.dilation[1] - 1)

        size[1] = (size[1] - k1 + 2*self.padding[0]) // self.stride[0] + 1
        size[2] = (size[2] - k2 + 2*self.padding[1]) // self.stride[1] + 1

        size[0] = self.filter_shape[0]
        return tuple(size)

    def reset(self):
        self.conv.weight.data = T.from_numpy(self.W_values.copy())
        if not self.no_bias:
            self.conv.bias.data = T.from_numpy(self.b_values.copy())

    def __repr__(self):
        string = 'Conv2DLayer({}, {}, {}, stride={}, padding={}, activation={})'.format(self.input_shape[0],
                                                                                         self.filter_shape[0], (self.filter_shape[2], self.filter_shape[3]),
                                                                                         self.stride, self.padding, self.activation)
        return string


class FCLayer(Layer):
    def __init__(self, input_shape, num_nodes, init_mode='Xavier_normal', no_bias=False, activation='relu',
                 show=True, **kwargs):
        """
        :param input_shape:
        :param num_nodes:
        :param init_mode: Xavier_normal, Xavier_uniform, He_normal, He_uniform
        :param no_bias:
        :param activation:
        :param args:
        """
        super(FCLayer, self).__init__()

        self.input_shape = int(np.prod(input_shape))
        self.num_nodes = num_nodes
        self.init_mode = init_mode
        self.activation = utils.function[activation]
        self.no_bias = no_bias
        self.kwargs = kwargs

        self.fc = nn.Linear(self.input_shape, num_nodes, not no_bias)
        params = list(self.fc.parameters())
        self.trainable = list(params)
        self.regularizable = [params[0]]

        utils.init[init_mode](self.fc.weight)
        self.W_values = self.fc.weight.data.numpy().copy()
        if not self.no_bias:
            self.b_values = np.zeros(num_nodes, dtype='float32')
            self.fc.bias.data = T.from_numpy(self.b_values)

        if show:
            print(self.fc)

    def forward(self, input):
        input = self.activation(self.fc(input.view(-1, self.input_shape)), **self.kwargs)
        return input

    @property
    def output_shape(self):
        return self.num_nodes

    def __repr__(self):
        return self.fc.__repr__()

    def reset(self):
        self.fc.weight.data = T.from_numpy(self.W_values.copy())
        if not self.no_bias:
            self.fc.bias.data = T.from_numpy(self.b_values.copy())


class BNLayer(Layer):
    def __init__(self, input_shape, epsilon=1e-4, running_average_factor=1e-1, activation='relu', no_scale=False,
                 show=True, **kwargs):
        """
        :param input_shape:
        :param epsilon:
        :param running_average_factor:
        :param activation:
        :param no_scale:
        :param args:
        """
        super(BNLayer, self).__init__()

        self.input_shape = tuple(input_shape) if isinstance(input_shape, (tuple, list)) else input_shape
        self.epsilon = np.float32(epsilon)
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.no_scale = no_scale
        self.kwargs = kwargs

        self.bn = nn.BatchNorm1d(self.input_shape, epsilon, running_average_factor) if isinstance(input_shape, int) \
            else nn.BatchNorm2d(self.input_shape[0], epsilon, running_average_factor)
        params = list(self.bn.parameters())
        self.trainable = [params[1]] if self.no_scale else list(params)
        self.regularizable = [params[0]] if not self.no_scale else []

        self.W_values = self.bn.weight.data.numpy().copy()
        self.b_values = self.bn.bias.data.numpy().copy()

        if show:
            print(self.bn)

    def forward(self, input):
        input = self.activation(self.bn(input), **self.kwargs)
        return input

    @property
    @validate
    def output_shape(self):
        return self.input_shape

    def train(self, mode=True):
        self.bn.train(mode)

    def eval(self):
        self.bn.eval()

    def __repr__(self):
        return self.bn.__repr__()

    def reset(self):
        self.bn.weight.data = T.from_numpy(self.W_values.copy())
        self.bn.bias.data = T.from_numpy(self.b_values.copy())


class ActivationLayer(Layer):
    def __init__(self, input_shape, activation='relu'):
        super(ActivationLayer, self).__init__()


class ConvBNAct(Layer):
    def __init__(self, input_shape, num_filters, filter_size, init_mode='Xavier_normal', no_bias=True,
                 border_mode='half', stride=1, activation='relu', dilation=1, epsilon=1e-4, running_average_factor=1e-1,
                 no_scale=False, groups=1, show=True, **kwargs):
        super(ConvBNAct, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.stride = stride
        self.padding = border_mode
        self.conv = Conv2DLayer(input_shape, num_filters, filter_size, init_mode, no_bias, border_mode, stride,
                                dilation, activation='linear', groups=groups, show=False, **kwargs)
        self.bn = BNLayer(self.conv.output_shape, epsilon, running_average_factor, activation, no_scale, show=False, **kwargs)
        self.trainable += self.conv.trainable + self.bn.trainable
        self.regularizable += self.conv.regularizable + self.bn.regularizable

        if show:
            print(self)

    def forward(self, input):
        input = self.conv(input)
        input = self.bn(input)
        return input

    @property
    @validate
    def output_shape(self):
        return self.bn.output_shape

    def __repr__(self):
        string = 'ConvBNAct({}, {}, {}, padding={}, stride={}, activation={})'.format(self.input_shape[0],
                                                                                      self.num_filters, self.filter_size,
                                                                                      self.padding, self.stride, self.activation)
        return string

    def train(self, mode=True):
        self.bn.train(mode)

    def eval(self):
        self.bn.eval()

    def reset(self):
        self.conv.reset()
        self.bn.reset()


class ResNetBasicBlock(Layer):
    def __init__(self, input_shape, num_filters, filter_size=3, stride=1, activation='relu', groups=1, batch_norm=True, **kwargs):
        super(ResNetBasicBlock, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.activation = utils.function[activation]
        self.groups = groups
        self.batch_norm = batch_norm
        self.kwargs = kwargs

        base_layer = ConvBNAct if batch_norm else Conv2DLayer
        self.convbnact1 = base_layer(input_shape=input_shape, num_filters=num_filters, filter_size=filter_size,
                                     init_mode='He_normal', stride=stride, activation=activation, show=False, **kwargs)
        self.convbnact2 = base_layer(input_shape=self.convbnact1.output_shape, num_filters=num_filters,
                                     filter_size=filter_size, init_mode='He_normal', stride=1, activation='linear', show=False, **kwargs)
        self.trainable += self.convbnact1.trainable + self.convbnact2.trainable
        self.regularizable += self.convbnact1.regularizable + self.convbnact2.regularizable

        self.downsample = lambda x: x
        if stride > 1 or input_shape[0] != num_filters:
            self.downsample = base_layer(input_shape=input_shape, num_filters=num_filters, filter_size=1, stride=stride, no_bias=True, activation='linear')
            self.trainable += self.downsample.trainable
            self.regularizable += self.downsample.regularizable

        print(self)

    def forward(self, input):
        res = input
        input = self.convbnact1(input)
        input = self.convbnact2(input)
        input += self.downsample(res)
        return self.activation(input, **self.kwargs)

    @property
    @validate
    def output_shape(self):
        return self.convbnact2.output_shape

    def __repr__(self):
        string = 'ResNetBasicBlock({}, {}, {}, stride={}, activation={})'.format(self.input_shape[0],
                                                                                             self.num_filters, self.filter_size,
                                                                                             self.stride, self.activation)
        return string

    def train(self, mode=True):
        self.convbnact1.train(mode)
        self.convbnact2.train(mode)
        if self.stride > 1 or self.input_shape[0] != self.num_filters:
            self.downsample.train(mode)

    def eval(self):
        self.convbnact1.eval()
        self.convbnact2.eval()
        if self.stride > 1 or self.input_shape[0] != self.num_filters:
            self.downsample.eval()

    def reset(self):
        self.convbnact1.reset()
        self.convbnact2.reset()
        if self.stride > 1 or self.input_shape[0] != self.num_filters:
            self.downsample.reset()


class ConvTrans2DLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, init_mode='Xavier_normal', stride=2, padding='half', output_padding=0, dilation=1,
                 no_bias=False, activation='linear', groups=1, **kwargs):
        """
        :param input_shape:
        :param num_filters:
        :param filter_size:
        :param init_mode: Xavier_normal, Xavier_uniform, He_normal, He_uniform
        :param stride:
        :param padding:
        :param output_padding:
        :param dilation:
        :param no_bias:
        :param activation:
        :param groups:
        :param args:
        """
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple. Received %s.' % type(input_shape)
        assert len(input_shape) == 3, 'input_shape must have 3 elements. Received %d.' % len(input_shape)
        super(ConvTrans2DLayer, self).__init__()
        self.filter_shape = (input_shape[0], num_filters, filter_size[0], filter_size[1]) if isinstance(filter_size, (list, tuple)) \
            else (input_shape[0], num_filters, filter_size, filter_size)
        self.input_shape = tuple(input_shape)
        self.init_mode = init_mode
        self.stride = stride
        self.output_padding = output_padding
        self.dilation = dilation
        self.no_bias = no_bias
        self.groups = groups
        self.kwargs = kwargs
        self.activation = utils.function[activation]

        if isinstance(padding, str):
            if padding == 'half':
                self.padding = (self.filter_shape[2] // 2, self.filter_shape[3] // 2)
            elif padding == 'valid':
                self.padding = (0, 0)
            elif padding == 'full':
                self.padding = (self.filter_shape[2] - 1, self.filter_shape[3] - 1)
            else:
                raise NotImplementedError
        elif isinstance(padding, (int, list, tuple)):
            self.padding = padding
        else:
            raise NotImplementedError

        self.convtrans = nn.ConvTranspose2d(int(input_shape[0]), num_filters, filter_size, stride, self.padding,
                                            output_padding, groups, not no_bias, dilation)
        params = list(self.convtrans.parameters())
        self.trainable += params
        self.regularizable += [params[0]]

        utils.init[init_mode](self.convtrans.weight)
        self.W_values = self.convtrans.weight.data.numpy().copy()
        if not self.no_bias:
            self.b_values = np.zeros(self.filter_shape[0], dtype='float32')
            self.convtrans.bias.data = T.from_numpy(self.b_values)

    def forward(self, input):
        input = self.activation(self.convtrans(input), **self.kwargs)
        return input

    @property
    @validate
    def output_shape(self):
        stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        h = ((self.input_shape[1] - 1) * stride[0]) + self.filter_shape[2] + \
            np.mod(self.input_shape[1] + 2 * self.padding[0] - self.filter_shape[2], stride[0]) - 2 * self.padding[0]
        w = ((self.input_shape[2] - 1) * stride[1]) + self.filter_shape[3] + \
            np.mod(self.input_shape[2] + 2 * self.padding[1] - self.filter_shape[3], stride[1]) - 2 * self.padding[1]
        return self.filter_shape[1], h, w

    def __repr__(self):
        return self.convtrans.__repr__()


class UpsamplingLayer(Layer):
    def __init__(self, input_shape, new_shape=None, scale_factor=2, mode='bilinear'):
        """
        :param input_shape:
        :param new_shape:
        :param scale_factor:
        """
        assert len(input_shape) == 3, 'input_shape must have 3 elements. Received %d.' % len(input_shape)
        assert isinstance(scale_factor, (int, list, tuple)), 'scale_factor must be an int, a list or a tuple. ' \
                                                             'Received %s.' % type(scale_factor)
        super(UpsamplingLayer, self).__init__()
        self.input_shape = input_shape
        self.new_shape = new_shape
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(new_shape, mode=mode) if new_shape is not None else nn.Upsample(scale_factor=scale_factor, mode=mode)
        print(self.upsample)

    @property
    @validate
    def output_shape(self):
        return (self.input_shape[0], self.new_shape[0], self.new_shape[1]) if self.new_shape is not None \
            else (self.input_shape[0], self.input_shape[1]*self.scale_factor, self.input_shape[2]*self.scale_factor) \
            if isinstance(self.scale_factor, int) else (self.input_shape[0], self.input_shape[1]*self.scale_factor[0], self.input_shape[2]*self.scale_factor[1])

    def forward(self, input):
        input = self.upsample(input)
        return input

    def __repr__(self):
        return self.upsample.__repr__()

    def reset(self):
        pass


class StackingConv(Layer):
    def __init__(self, input_shape, num_filters, filter_size, num_layers, batch_norm=False, init_mode='Xavier_normal',
                 no_bias=True, border_mode='half', stride=(1, 1), dilation=(1, 1), activation='relu', groups=1, **kwargs):
        assert num_layers > 1, 'num_layers must be greater than 1, got %d' % num_layers
        super(StackingConv, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.activation = activation
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        self.block = nn.Sequential()
        shape = tuple(input_shape)
        conv_layer = ConvBNAct if batch_norm else Conv2DLayer
        for num in range(num_layers - 1):
            self.block.add_module('Stacking_conv%d' % (num+1), conv_layer(input_shape=shape, num_filters=num_filters,
                                                                          filter_size=filter_size, init_mode=init_mode,
                                                                          stride=1, border_mode=border_mode, dilation=dilation,
                                                                          activation=activation, no_bias=no_bias, show=False, **kwargs))
            shape = self.block[-1].output_shape
            self.trainable += self.block[-1].trainable
            self.regularizable += self.block[-1].regularizable
        self.block.add_module('Stacking_conv%d' % num_layers, conv_layer(input_shape=self.block[-1].output_shape,
                                                                         num_filters=num_filters, no_bias=no_bias,
                                                                         filter_size=filter_size, init_mode=init_mode,
                                                                         stride=stride, border_mode=border_mode,
                                                                         dilation=dilation, activation=activation, show=False, **kwargs))
        self.trainable += self.block[-1].trainable
        self.regularizable += self.block[-1].regularizable

        print(self)

    def forward(self, input):
        input = self.block(input)
        return input

    @property
    def output_shape(self):
        return self.block[-1].output_shape

    def __repr__(self):
        string = 'StackingConv({}, {}, {}, num_layers={}, stride={}, activation={})'.format(self.input_shape[0],
                                                                                            self.num_filters, self.filter_size,
                                                                                            self.num_layers, self.stride, self.activation)
        return string

    def train(self, mode=True):
        for layer in self.block:
            layer.train(mode)

    def eval(self):
        for layer in self.block:
            layer.eval()

    def reset(self):
        for layer in self.block:
            layer.reset()
