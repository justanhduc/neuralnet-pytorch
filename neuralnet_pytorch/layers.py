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

__all__ = ['Conv2d', 'ConvNormAct', 'ConvTranspose2d', 'StackingConv', 'ResNetBasicBlock',
           'ResNetBottleneckBlock', 'FC', 'WrapperLayer', 'Activation', 'Sequential', 'Layer', 'Lambda',
           'Module', 'Softmax']


class NetMethod:
    @property
    @utils.validate
    def output_shape(self):
        assert not hasattr(super(), 'output_shape')

        return tuple(self.input_shape)

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


class Layer(NetMethod):
    def __init__(self):
        self.input_shape = None


class Module(nn.Module, Layer, NetMethod):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def add_module(self, name, module):
        r"""Adds a child module to the current module. Modified to be able to add WrapperLayer.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            parameter (Module): child module to be added to the module.
        """
        if not isinstance(module, (nn.Module, WrapperLayer)) and module is not None:
            raise TypeError("{} is not a Module/Wrapper subclass".format(
                T.typename(module)))
        elif not isinstance(name, T._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(
                T.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module


class Sequential(nn.Sequential, NetMethod):
    def __init__(self, input_shape, *args):
        super().__init__(*args)
        self.input_shape = input_shape

    @property
    @utils.validate
    def output_shape(self):
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

    def add_module(self, name, module):
        r"""Adds a child module to the current module. Modified to be able to add WrapperLayer.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            parameter (Module): child module to be added to the module.
        """
        if not isinstance(module, (nn.Module, WrapperLayer)) and module is not None:
            raise TypeError("{} is not a Module/Wrapper subclass".format(
                T.typename(module)))
        elif not isinstance(name, T._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(
                T.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module


class WrapperLayer(Layer):
    def __init__(self, input_shape, layer, *args, **kwargs):
        super(WrapperLayer, self).__init__()
        self.input_shape = input_shape
        device = kwargs.pop('device', None)
        self.__wrapped_layer = layer(*args, **kwargs)
        if cuda_available:
            self.cuda(device)

    def __getattr__(self, item):
        try:
            return self.__wrapped_layer.__getattribute__(item)
        except AttributeError:
            return super(WrapperLayer, self).__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return self.__wrapped_layer(*args, **kwargs)

    def __repr__(self):
        return str(self.__wrapped_layer)

    @property
    @utils.validate
    def output_shape(self):
        shape = [1 if x is None else x for x in self.input_shape]
        dummy = T.zeros(*shape)
        dummy = self(dummy)
        return tuple(dummy.shape)


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
        return 'Lambda({}, output_shape={})'.format(self.input_shape, self.output_shape)


class Conv2d(nn.Conv2d, Layer):
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
        shape = [np.nan if s is None else s for s in self.input_shape]
        shape[2:] = [(s - self.ks[idx] + 2 * self.padding[idx]) // self.stride[idx] + 1 for idx, s in
                     enumerate(shape[2:])]
        shape[1] = self.out_channels
        return tuple(shape)

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)


class FC(nn.Linear, Layer):
    def __init__(self, input_shape, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 keepdim=False, **kwargs):
        assert None not in input_shape[1:], 'Shape of input must be known for FC layer'
        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.keepdim = keepdim
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else lambda x, **kwargs: activation(x)
        self.kwargs = kwargs

        super().__init__(int(np.prod(self.input_shape[1:])), out_features, bias)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        output = self.activation(super().forward(T.flatten(input, 1)), **self.kwargs)
        return output.flatten() if self.out_features == 1 and not self.keepdim else output

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0], self.out_features

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
        return 'Softmax({}, out_features={}, dim={})'.format(self.input_shape, self.out_features, self.dim)


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
        return 'Activation({})'.format(self.activation)


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

    @property
    @utils.validate
    def output_shape(self):
        return self.norm.output_shape

    def __repr__(self):
        string = 'ConvNormAct({}, {}, {}, padding={}, stride={}, activation={})'.format(self.input_shape,
                                                                                        self.out_channels,
                                                                                        self.kernel_size, self.padding,
                                                                                        self.stride, self.activation)
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

    @property
    @utils.validate
    def output_shape(self):
        return self.block.output_shape

    def __repr__(self):
        string = 'ResNetBasicBlock({}, {}, {}, stride={}, activation={})'.format(self.input_shape,
                                                                                 self.out_channels, self.kernel_size,
                                                                                 self.stride, self.activation)
        return string


class ResNetBottleneckBlock(ResNetBasicBlock):
    expansion = 4

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, activation='relu', downsample=None, groups=1,
                 block=None, weights_init=None, bias_init=None, norm_method='bn', **kwargs):
        super().__init__(input_shape, out_channels, kernel_size, stride, activation, downsample, groups, block=block,
                         weights_init=weights_init, bias_init=bias_init, norm_method=norm_method, **kwargs)

    def __repr__(self):
        string = 'ResNetBottleNeckBlock({}, {}, {}, stride={}, activation={})'.format(self.input_shape,
                                                                                      self.out_channels,
                                                                                      self.kernel_size, self.stride,
                                                                                      self.activation)
        return string


class ConvTranspose2d(nn.ConvTranspose2d, Layer):
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
        self.regularizable += tuple([self.bias])

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input):
        output_size = self.kwargs.pop('output_size', None)
        output = self.activation(super().forward(input, output_size=output_size), **self.kwargs)
        return output

    @property
    @utils.validate
    def output_shape(self):
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

    @property
    @utils.validate
    def output_shape(self):
        return self[self.num_layers - 1].output_shape

    def __repr__(self):
        string = 'StackingConv({}, {}, {}, num_layers={}, stride={}, activation={})'.format(self.input_shape,
                                                                                            self.num_filters,
                                                                                            self.filter_size,
                                                                                            self.num_layers,
                                                                                            self.stride,
                                                                                            self.activation)
        return string
