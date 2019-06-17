"""
Written by Duc Nguyen (Mar 21, 18)
Updated Jan 13, 2019
"""
from functools import partial
from collections import OrderedDict

import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.modules.utils import _pair

from neuralnet_pytorch import utils
from neuralnet_pytorch.utils import _image_shape, _matrix_shape, _pointset_shape
from neuralnet_pytorch.utils import cuda_available

__all__ = ['Conv2d', 'ConvNormAct', 'ConvTranspose2d', 'StackingConv', 'ResNetBasicBlock', 'FC', 'wrapper',
           'ResNetBottleneckBlock', 'Activation', 'Sequential', 'Lambda', 'Module', 'Softmax', 'Sum', 'XConv',
           'GraphConv', 'MultiSingleInputModule', 'MultiMultiInputModule', 'SequentialSum', 'ConcurrentSum',
           'Net']


class Net:
    def __init__(self, *args, **kwargs):
        self.optimizer = None
        self.scheduler = None
        self.stats = {
            'scalars': {},
            'images': {},
            'histograms': {},
            'pointclouds': {},
            'predictions': {}
        }

    def train_procedure(self, *args, **kwargs):
        raise NotImplementedError

    def learn(self, *args, **kwargs):
        raise NotImplementedError

    def eval_procedure(self, *args, **kwargs):
        raise NotImplementedError


class _LayerMethod:
    @property
    @utils.validate
    def output_shape(self):
        raise NotImplementedError

    @property
    def params(self):
        assert not hasattr(super(), 'params')
        return tuple(self.state_dict().values())

    @property
    def trainable(self):
        assert not hasattr(super(), 'trainable')
        params = []
        if hasattr(self, 'parameters'):
            params = [p for p in self.parameters() if p.requires_grad]
        return tuple(params)

    @property
    def regularizable(self):
        assert not hasattr(super(), 'regularizable')
        params = []
        if hasattr(self, 'weight'):
            params += [self.weight]

        for m in list(self.children()):
            if hasattr(m, 'regularizable'):
                params.extend(m.regularizable)

        return tuple(params)

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


@utils.add_simple_repr
class Module(nn.Module, _LayerMethod):
    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape


@utils.add_simple_repr
class MultiSingleInputModule(nn.Module, _LayerMethod):
    def __init__(self, *modules_or_tensors):
        assert all(isinstance(item, (nn.Module, T.Tensor)) for item in modules_or_tensors), \
            'All items in modules_or_tensors should be Pytorch modules or tensors'

        super().__init__()
        self.input_shape = []

        def foo(item):
            idx = len(list(self.children()))
            if isinstance(item, nn.Module):
                self.add_module('module%d' % idx, item)
                self.input_shape.append(item.output_shape)
            else:
                self.add_module('tensor%d' % idx, Lambda(lambda *args, **kwargs: item, input_shape=item.shape,
                                                         output_shape=item.shape))
                self.input_shape.append(item.shape)

        list(map(foo, modules_or_tensors))
        self.input_shape = tuple(self.input_shape)

    def forward(self, input, *args, **kwargs):
        outputs = [module(input) for name, module in self.named_children()]
        return tuple(outputs)

    def trainable(self):
        return tuple()

    def params(self):
        return tuple()

    @property
    def regularizable(self):
        return tuple()


class MultiMultiInputModule(MultiSingleInputModule):
    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, *input, **kwargs):
        input_it = iter(input)
        outputs = [module(next(input_it)) if name.startswith('module') else module()
                   for name, module in self.named_children()]
        return tuple(outputs)


@utils.add_simple_repr
class Sequential(nn.Sequential, _LayerMethod):
    def __init__(self, *args, input_shape=None):
        super().__init__(*args)
        self.input_shape = input_shape

    def forward(self, input, *args, **kwargs):
        for module in self._modules.values():
            input = module(input, *args, **kwargs)
        return input

    @property
    @utils.validate
    def output_shape(self):
        if self.input_shape is None:
            return None

        layers = list(self.children())
        return layers[-1].output_shape if layers else self.input_shape

    def reset_parameters(self):
        for m in self.children():
            m.reset_parameters()


def wrapper(layer: nn.Module, input_shape=None, *args, **kwargs):
    assert isinstance(layer, nn.Module), 'layer must be a subclass of Pytorch\'s Module'

    @utils.add_simple_repr
    class _Wrapper(layer, _LayerMethod):
        def __init__(self):
            self.input_shape = input_shape
            self.output_shape_tmp = kwargs.pop('output_shape', None)
            device = kwargs.pop('device', None)

            super().__init__(*args, **kwargs)
            if cuda_available:
                self.cuda(device)

        def forward(self, input, *args, **kwargs):
            return super().forward(input)

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

    return _Wrapper


class Lambda(Module):
    def __init__(self, func, input_shape=None, output_shape=None, **kwargs):
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

    def extra_repr(self):
        s = '{}'.format(self.func.__name__)
        return s


@utils.add_simple_repr
class Conv2d(nn.Conv2d, _LayerMethod):
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', dilation=1, groups=1,
                 bias=True, activation=None, weights_init=None, bias_init=None, **kwargs):
        input_shape = _image_shape(input_shape)
        assert input_shape[1] is not None, 'Shape at dimension 1 (zero-based index) must be known'

        self.input_shape = input_shape
        kernel_size = _pair(kernel_size)
        self.no_bias = bias
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.border_mode = padding
        dilation = _pair(dilation)
        groups = groups
        self.kwargs = kwargs

        self.ks = [fs + (fs - 1) * (d - 1) for fs, d in zip(kernel_size, dilation)]
        if isinstance(padding, str):
            if padding == 'half':
                padding = [k >> 1 for k in self.ks]
            elif padding in ('valid', 'ref', 'rep'):
                padding = (0,) * len(self.ks)
            elif padding == 'full':
                padding = [k - 1 for k in self.ks]
            else:
                raise NotImplementedError
        elif isinstance(padding, int):
            pass
        else:
            raise ValueError('padding must be a str/tuple/int, got %s' % type(padding))

        super().__init__(int(input_shape[1]), out_channels, kernel_size, stride, tuple(padding), dilation, bias=bias,
                         groups=groups)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
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

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


@utils.add_simple_repr
class ConvTranspose2d(nn.ConvTranspose2d, _LayerMethod):
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', output_padding=0, bias=True,
                 dilation=1, weights_init=None, bias_init=None, activation='linear', groups=1, output_size=None, **kwargs):
        input_shape = _image_shape(input_shape)
        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.output_size = _pair(output_size)
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

        super().__init__(int(input_shape[1]), out_channels, kernel_size, stride, padding, output_padding, groups, bias,
                         dilation)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, output_size=None, *args, **kwargs):
        output = self.activation(super().forward(
            input, output_size=self.output_size if output_size is None else output_size), **self.kwargs)
        return output

    @property
    @utils.validate
    def output_shape(self):
        if self.output_size is not None:
            return (self.input_shape[0], self.out_channels) + self.output_size

        shape = [np.nan if s is None else s for s in self.input_shape]
        _, _, h_in, w_in = shape
        h_out = (h_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + \
                self.output_padding[0] + 1
        w_out = (w_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + \
                self.output_padding[1] + 1
        return self.input_shape[0], self.out_channels, h_out, w_out

    def reset_parameters(self):
        super().reset_parameters()
        if self.weights_init:
            self.weights_init(self.weight)

        if self.bias is not None and self.bias_init:
            self.bias_init(self.bias)

    def extra_repr(self):
        s = 'activation={}'.format(self.activation.__name__)
        return s


@utils.add_simple_repr
class FC(nn.Linear, _LayerMethod):
    def __init__(self, input_shape, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 flatten=False, keepdim=True, **kwargs):
        input_shape = _matrix_shape(input_shape)
        assert input_shape[-1] is not None, 'Shape at the last position (zero-based index) must be known'

        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.kwargs = kwargs

        super().__init__(int(np.prod(input_shape[1:])) if flatten else input_shape[-1], out_features, bias)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        if self.flatten:
            input = T.flatten(input, 1)

        output = self.activation(super().forward(input), **self.kwargs)
        return output.flatten(-2) if self.out_features == 1 and not self.keepdim else output

    @property
    @utils.validate
    def output_shape(self):
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

    def extra_repr(self):
        s = super().extra_repr()
        s += ', activation={}'.format(self.activation.__name__)
        return s


class Softmax(FC):
    def __init__(self, input_shape, out_features, dim=1, weights_init=None, bias_init=None, **kwargs):
        self.dim = dim
        kwargs['dim'] = dim
        super().__init__(input_shape, out_features, activation='softmax', weights_init=weights_init, bias_init=bias_init,
                         **kwargs)


@utils.add_simple_repr
@utils.no_dim_change_op
class Activation(Module):
    def __init__(self, activation='relu', input_shape=None, **kwargs):
        super().__init__(input_shape)
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.kwargs = kwargs

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        return self.activation(input, **self.kwargs)

    def extra_repr(self):
        s = 'activation={}'.format(self.activation.__name__)
        return s


@utils.add_custom_repr
class ConvNormAct(Sequential):
    def __init__(self, input_shape, out_channels, kernel_size, weights_init=None, bias=True, bias_init=None,
                 padding='half', stride=1, dilation=1, activation='relu', groups=1, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, no_scale=False, norm_method='bn', **kwargs):
        super().__init__(input_shape=input_shape)
        from neuralnet_pytorch.normalization import BatchNorm2d, InstanceNorm2d, LayerNorm

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.norm_method = norm_method
        self.conv = Conv2d(input_shape, out_channels, kernel_size, weights_init=weights_init, bias=bias,
                           bias_init=bias_init, padding=padding, stride=stride, dilation=dilation, activation=None,
                           groups=groups, **kwargs)

        norm_method = BatchNorm2d if norm_method == 'bn' else InstanceNorm2d if norm_method == 'in' \
            else LayerNorm if norm_method == 'ln' else norm_method
        assert isinstance(norm_method, Module)
        self.norm = norm_method(self.conv.output_shape, eps, momentum, affine, track_running_stats,
                                no_scale=no_scale, activation=self.activation, **kwargs)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.conv.padding != (0,) * len(self.conv.padding):
            s += ', padding={padding}'
        if self.conv.dilation != (1,) * len(self.conv.dilation):
            s += ', dilation={dilation}'
        if self.conv.output_padding != (0,) * len(self.conv.output_padding):
            s += ', output_padding={output_padding}'
        if self.conv.groups != 1:
            s += ', groups={groups}'
        if self.conv.bias is None:
            s += ', bias=False'

        s = s.format(**self.conv.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@utils.add_custom_repr
class FCNormAct(Sequential):
    def __init__(self, input_shape, out_features, bias=True, weights_init=None, bias_init=None, flatten=False,
                 keepdim=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, activation=None,
                 no_scale=False, norm_method='bn', **kwargs):
        super().__init__(input_shape=input_shape)
        from neuralnet_pytorch.normalization import BatchNorm1d, InstanceNorm1d, LayerNorm, FeatureNorm1d

        self.out_features = out_features
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)

        self.fc = FC(self.input_shape, out_features, bias, weights_init=weights_init, bias_init=bias_init,
                     flatten=flatten, keepdim=keepdim)

        norm_method = BatchNorm1d if norm_method == 'bn' else InstanceNorm1d if norm_method == 'in' \
            else LayerNorm if norm_method == 'ln' else FeatureNorm1d if norm_method == 'fn' else norm_method
        assert isinstance(norm_method, Module)
        self.norm = norm_method(self.conv.output_shape, eps, momentum, affine, track_running_stats,
                                no_scale=no_scale, activation=self.activation, **kwargs)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def extra_repr(self):
        s = '{in_features}, {out_features}'
        if self.flatten:
            s += 'flatten={flatten}'
        if not self.keepdim:
            s += 'keepdim={keepdim}'

        s = s.format(**self.conv.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@utils.add_custom_repr
class ResNetBasicBlock(Module):
    expansion = 1

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, dilation=(1, 1), activation='relu',
                 downsample=None, groups=1, block=None, weights_init=None, norm_method='bn', **kwargs):
        input_shape = _image_shape(input_shape)

        super().__init__(input_shape=input_shape)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.groups = groups
        self.weights_init = weights_init
        self.norm_method = norm_method
        self.kwargs = kwargs

        self.block = self._make_block() if block is None else block(**kwargs)
        if downsample is not None:
            assert isinstance(Module, downsample), 'downsample must be an instance of Module, got %s' % type(downsample)
            self.downsample = downsample
        else:
            if stride > 1 or input_shape[1] != out_channels * self.expansion:
                self.downsample = ConvNormAct(input_shape, out_channels * self.expansion, 1, stride=stride, bias=False,
                                              weights_init=weights_init, activation='linear')
            else:
                self.downsample = Lambda(lambda x: x, output_shape=input_shape, input_shape=input_shape)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def _make_block(self):
        block = Sequential(input_shape=self.input_shape)
        if self.expansion != 1:
            block.add_module('pre',
                             ConvNormAct(block.output_shape, self.out_channels, 1, stride=1, bias=False,
                                         weights_init=self.weights_init, groups=self.groups))
        block.add_module('conv_norm_act_1',
                         ConvNormAct(block.output_shape, self.out_channels, self.kernel_size, bias=False,
                                     weights_init=self.weights_init, stride=self.stride, activation=self.activation,
                                     groups=self.groups, norm_method=self.norm_method, **self.kwargs))
        block.add_module('conv_norm_act_2',
                         ConvNormAct(block.output_shape, self.out_channels * self.expansion,
                                     1 if self.expansion != 1 else self.kernel_size, bias=False, stride=1,
                                     activation=None, groups=self.groups, weights_init=self.weights_init,
                                     norm_method=self.norm_method, **self.kwargs))
        return block

    def forward(self, input, *args, **kwargs):
        res = input
        out = self.block(input)
        out += self.downsample(res)
        return self.activation(out, **self.kwargs)

    def extra_repr(self):
        s = ('{input_shape}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s

    @property
    @utils.validate
    def output_shape(self):
        return self.downsample.output_shape


class ResNetBottleneckBlock(ResNetBasicBlock):
    expansion = 4

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, dilation=(1, 1), activation='relu',
                 downsample=None, groups=1, block=None, weights_init=None, norm_method='bn', **kwargs):
        super().__init__(input_shape, out_channels, kernel_size, stride=stride, dilation=dilation, activation=activation,
                         downsample=downsample, groups=groups, block=block, weights_init=weights_init,
                         norm_method=norm_method, **kwargs)


@utils.add_custom_repr
class StackingConv(Sequential):
    def __init__(self, input_shape, out_channels, kernel_size, num_layers, stride=1, padding='half', dilation=(1, 1),
                 bias=True, weights_init=None, bias_init=None, norm_method=None, activation='relu', groups=1, **kwargs):
        assert num_layers > 1, 'num_layers must be greater than 1, got %d' % num_layers
        input_shape = _image_shape(input_shape)

        super(StackingConv, self).__init__(input_shape=input_shape)
        self.num_filters = out_channels
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.num_layers = num_layers
        self.norm_method = norm_method

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

    def extra_repr(self):
        s = ('{input_shape}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_layers={num_layers}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


class Sum(MultiSingleInputModule):
    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, input, *args, **kwargs):
        outputs = super().forward(input)
        return sum(outputs)

    @property
    @utils.validate
    def output_shape(self):
        if None in self.input_shape:
            return None

        shapes_transposed = [item for item in zip(*self.input_shape)]
        input_shape_none_filtered = list(map(utils.get_non_none, shapes_transposed))

        return tuple(input_shape_none_filtered)


class SequentialSum(Sum):
    def __init__(self, *modules):
        super().__init__(*modules)

    def forward(self, input, *args, **kwargs):
        outputs = []
        output = input
        for name, module in self.named_children():
            if name.startswith('tensor'):
                outputs.append(module())
            else:
                output = module(output)
                outputs.append(output)

        return sum(outputs)


class ConcurrentSum(MultiMultiInputModule):
    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, *input, **kwargs):
        outputs = super().forward(*input)
        return sum(outputs)

    @property
    @utils.validate
    def output_shape(self):
        if None in self.input_shape:
            return None

        shapes_transposed = [item for item in zip(*self.input_shape)]
        input_shape_none_filtered = list(map(utils.get_non_none, shapes_transposed))

        return tuple(input_shape_none_filtered)


@utils.add_custom_repr
class DepthwiseSepConv2D(Sequential):
    """ Depthwise separable convolution"""

    def __init__(self, input_shape, out_channels, kernel_size, depth_mul, padding='half', dilation=(1, 1),
                 activation=None):
        """
        :param input_shape: Shape of input features
        :param out_channels: Length of output features (first dimension)
        :param kernel_size: Size of convolutional kernel
        :param depth_mul: Depth multiplier for middle part of separable convolution
        :param activation: Activation function
        """
        input_shape = _image_shape(input_shape)

        super().__init__(input_shape=input_shape)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depth_mul = depth_mul
        self.padding = padding
        self.dilation = dilation
        self.activation = utils.function[activation] if isinstance(activation, str) or activation is None \
            else utils._wrap(activation)
        self.add_module('depthwise', Conv2d(self.output_shape, input_shape[1] * depth_mul, kernel_size, padding=padding,
                                            dilation=dilation, groups=input_shape[1]))
        self.add_module('pointwise', Conv2d(self.output_shape, out_channels, 1, activation=activation, padding=padding,
                                            dilation=dilation, bias=False))

    def extra_repr(self):
        s = ('{input_shape}, {out_channels}, kernel_size={kernel_size}'
             ', depth_mul={depth_mul}')
        if self.padding != 'half':
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@utils.add_custom_repr
class XConv(Module):
    def __init__(self, input_shape, feature_dim, out_channels, out_features, num_neighbors, depth_mul,
                 activation='relu', dropout=None, bn=True):
        input_shape = _pointset_shape(input_shape)

        super().__init__(input_shape)
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.out_features = out_features
        self.depth_mul = depth_mul
        self.activation = activation
        self.dropout = dropout
        self.bn = bn

        self.fcs = Sequential(input_shape=input_shape)
        self.fcs.add_module('fc1', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout1', wrapper(self.output_shape, T.nn.Dropout2d, p=dropout))
        self.fcs.add_module('fc2', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout2', wrapper(self.output_shape, T.nn.Dropout2d, p=dropout))

        from neuralnet_pytorch.resizing import DimShuffle
        from neuralnet_pytorch.normalization import BatchNorm2d

        self.x_trans = Sequential(input_shape=input_shape[:2] + (num_neighbors, input_shape[-1]))
        self.x_trans.add_module('dimshuffle1', DimShuffle(self.x_trans.output_shape, (0, 3, 1, 2)))
        self.x_trans.add_module('conv', Conv2d(self.x_trans.output_shape, num_neighbors ** 2, (1, num_neighbors),
                                               activation=activation, padding='valid'))
        self.x_trans.add_module('dimshuffle2', DimShuffle(self.x_trans.output_shape, (0, 2, 3, 1)))
        self.x_trans.add_module('fc1', FC(self.x_trans.output_shape, num_neighbors ** 2, activation='relu'))
        self.x_trans.add_module('fc2', FC(self.x_trans.output_shape, num_neighbors ** 2))

        self.end_conv = Sequential(input_shape=input_shape[:2] + (num_neighbors, feature_dim + out_features))
        self.end_conv.add_module('dimshuffle1', DimShuffle(self.end_conv.output_shape, (0, 3, 1, 2)))
        self.end_conv.add_module('conv',
                                 DepthwiseSepConv2D(self.end_conv.output_shape, out_channels, (1, num_neighbors),
                                                    depth_mul=depth_mul, activation=None if bn else activation,
                                                    padding='valid'))
        if bn:
            self.end_conv.add_module('bn', BatchNorm2d(self.end_conv.output_shape, momentum=.9, activation=activation))
        self.end_conv.add_module('dimshuffle2', DimShuffle(self.end_conv.output_shape, (0, 2, 3, 1)))

    def forward(self, *input, **kwargs):
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

    def extra_repr(self):
        s = ('{input_shape}, feature_dim={feature_dim}, out_channels={out_channels}, out_features={out_features}, '
             'num_neighbors={num_neighbors}, depth_mul={depth_mul}, dropout={dropout}, bn={bn}')

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


class GraphConv(FC):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None):
        super().__init__(input_shape, out_features, bias, activation=activation)

    def reset_parameters(self):
        if self.weights_init is None:
            stdv = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weights_init(self.weight)

        if self.bias is not None:
            if self.bias_init is None:
                stdv = 1. / np.sqrt(self.weight.size(1))
                self.bias.data.uniform_(-stdv, stdv)
            else:
                self.bias_init(self.bias)

    def forward(self, input, adj, *args, **kwargs):
        support = T.mm(input, self.weight.t())
        output = T.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output
