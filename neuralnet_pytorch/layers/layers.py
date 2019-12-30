from functools import partial
from collections import OrderedDict

import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .. import utils
from ..utils import _image_shape, _matrix_shape, _pointset_shape
from ..utils import cuda_available

__all__ = ['Conv2d', 'ConvNormAct', 'ConvTranspose2d', 'StackingConv', 'ResNetBasicBlock', 'FC', 'wrapper',
           'ResNetBottleneckBlock', 'Activation', 'Sequential', 'Lambda', 'Module', 'Softmax', 'Sum', 'XConv',
           'GraphConv', 'MultiSingleInputModule', 'MultiMultiInputModule', 'SequentialSum', 'ConcurrentSum',
           'Net', 'DepthwiseSepConv2D', 'FCNormAct', 'BatchGraphConv', 'GraphXConv', 'SingleMultiInputModule']


class Net:
    """
    This abstract class is useful when you want to use
    :meth:`~neuralnet_pytorch.monitor.Monitor.run_training`.
    For a start, subclass this as the first parent.
    Then specify your optimization and schedule methods using :attr:`~optim`.
    You can specify your training procedure in :attr:`~train_procedure` and
    use :meth:`~learn` to perform optimization.
    If :meth:`~eval_procedure` is specified,
    Use :attr:`~stats` to collect your interested statistics from your training
    and evaluation.
    These statistics can be printed out or displayed in Tensorboard via
    :class:`~neuralnet_pytorch.monitor.Monitor`.

    Parameters
    ----------
    args
        arguments to be passed to `super`.
    kwargs
        keyword arguments to be passed to `super`.

    Attributes
    ----------
    optim
        a dictionary that contains the optimizer and scheduler for optimization.
    stats
        a dictionary to hold the interested statistics from training and evaluation.
        For each ``'train'`` and ``'eval'`` keys, an other dictionary with several
        built-in keys.
        The possible keys are: ``'scalars'``, ``'images'``, ``'histograms'``,
        and  ``'pointclouds'``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stats = {'train': {
                'scalars': {},
                'images': {},
                'histograms': {},
                'pointclouds': {}
            },
            'eval': {
                'scalars': {},
                'images': {},
                'histograms': {},
                'pointclouds': {}
            }
        }

    def train_procedure(self, *args, **kwargs):
        """
        Your training instructions can be specified here.
        This can return the loss to be optimized in :meth:`~learn`.
        You can use :attr:`~stats` to record the interested statistics.
        """

        raise NotImplementedError

    def learn(self, optim, *args, **kwargs):
        """
        The optimization can be defined here.
        Usually, it includes zeroing gradients, optimizing the loss,
        and collect statistics.
        """

        raise NotImplementedError

    def eval_procedure(self, *args, **kwargs):
        """
        If specified, an evaluation will be performed for your model.
        Use :attr:`~stats` to collect statistics.
        """

        raise NotImplementedError


class _LayerMethod:
    """
    This mixin class contains various attributes to extend :mod:`torch` modules.
    """

    @property
    @utils.validate
    def output_shape(self):
        """
        Returns the output shape of the module.
        """

        raise NotImplementedError

    @property
    def params(self):
        """
        Return a tuple of all the parameters in the module.
        """

        assert not hasattr(super(), 'params')
        return tuple(self.state_dict().values())

    @property
    def trainable(self):
        """
        Return a tuple of all parameters with :attr:`requires_grad` set to `True`.
        """

        assert not hasattr(super(), 'trainable')
        params = []
        if hasattr(self, 'parameters'):
            params = [p for p in self.parameters() if p.requires_grad]
        return tuple(params)

    @property
    def regularizable(self):
        """
        Returns a tuple of parameters to be regularized.
        """

        assert not hasattr(super(), 'regularizable')
        params = []
        if hasattr(self, 'weight'):
            if self.weight.requires_grad:
                params += [self.weight]

        for m in list(self.children()):
            if hasattr(m, 'regularizable'):
                params.extend(m.regularizable)

        return tuple(params)

    def save(self, param_file):
        """
        Save the weights of the model in :class:`numpy.nrdarray` format.

        :param param_file:
            path to the weight file.
        """

        assert not hasattr(super(), 'save')
        params_np = utils.bulk_to_numpy(self.params)
        params_dict = OrderedDict(zip(list(self.state_dict().keys()), params_np))
        T.save(params_dict, param_file)
        print('Model weights dumped to %s' % param_file)

    def load(self, param_file, eval=True):
        """
        Load the `numpy.ndarray` weights from file.

        :param param_file:
            path to the weight file.
        :param eval:
            whether to use evaluation mode or not.
        """

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
        """
        This overloads the :meth:`torch.Module.reset_parameters` of the module.
        Used for custom weight initialization.
        """

        assert not hasattr(super(), 'reset_parameters')
        pass


@utils.add_simple_repr
class Module(nn.Module, _LayerMethod):
    """
    Similar to :class:`torch.nn.Module`, but extended by
    :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.
    All the usages in native Pytorch are preserved.

    Parameters
    ----------
    input_shape
        shape of the tensor to be input to the modules.
        Can be a list, tuple, nested list/tuple or an integer.
    """

    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape


@utils.add_simple_repr
class MultiSingleInputModule(Module):
    """
    This is an abstract class.
    This class computes the results of multiple modules given an input tensor,
    then fuses the results.

    Parameters
    ----------
    modules_or_tensors
        a list of modules or tensors whose results are fused together.

    Attributes
    ----------
    input_shape
        a list of input shapes of the incoming modules and tensors.
    """

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
        outputs = [module(input, *args, **kwargs) for name, module in self.named_children()]
        return tuple(outputs)

    @property
    def trainable(self):
        return tuple()

    @property
    def params(self):
        return tuple()

    @property
    def regularizable(self):
        return tuple()


class MultiMultiInputModule(MultiSingleInputModule):
    """
    Similar to :class:`MultiSingleInputModule`, but each module has its own input tensor.
    """

    def __init__(self, *modules_or_tensors):
        super().__init__(*modules_or_tensors)

    def forward(self, *input, **kwargs):
        input_it = iter(input)
        outputs = [module(next(input_it), **kwargs) if name.startswith('module') else module()
                   for name, module in self.named_children()]
        return tuple(outputs)


class SingleMultiInputModule(Module):
    def __init__(self, module):
        super().__init__(module.output_shape)
        self.module = module

    @property
    @utils.validate
    def output_shape(self):
        return self.module.output_shape

    def forward(self, *input, **kwargs):
        return tuple([self.module(inp, **kwargs) for inp in input])

    @property
    def trainable(self):
        return tuple()

    @property
    def params(self):
        return tuple()

    @property
    def regularizable(self):
        return tuple()


@utils.add_simple_repr
class Sequential(nn.Sequential, _LayerMethod):
    """
    Similar to :class:`torch.nn.Sequential`, but extended by
    :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.
    All the usages in native Pytorch are preserved.

    Parameters
    ----------
    args
        a list of modules as in :class:`torch.nn.Sequential`.
    input_shape
        shape of the input tensor. If ``None``, the functionality is
        the same as :class:`torch.nn.Sequential`.
    """

    def __init__(self, *args, input_shape=None):
        super().__init__(*args)
        self.input_shape = input_shape

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start else 0
            modules = list(self._modules.items())
            return Sequential(OrderedDict(modules[idx]), input_shape=modules[start][1].input_shape)
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

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


def wrapper(input_shape=None, output_shape=None, *args, **kwargs):
    """
    A class decorator to wrap any :mod:`torch` module.

    :param input_shape:
        shape of the input to the module.
        Can be ``None``.
    :param output_shape:
        shape of the output tensor.
        If ``None``, the output shape is calculated by performing a forward pass.
    :param args:
        extra arguments needed by the module.
    :param kwargs:
        extra keyword arguments needed by the module.
    :return:
        The input module extended by :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Examples
    --------
    You can use this function directly on any :mod:`torch` module

    >>> import torch.nn as nn
    >>> import neuralnet_pytorch as nnt
    >>> dropout = nnt.wrapper(p=.2)(nn.Dropout2d)() # because wrapper returns a class!

    Alternatively, you can use it as a decorator

    .. code-block:: python

        import torch.nn as nn
        import neuralnet_pytorch as nnt

        @nnt.wrapper(# optional arguments for input and output shapes)
        class Foo(nn.Module):
            ...

        foo = Foo()
    """
    assert input_shape is None or isinstance(input_shape, (int, list, tuple)), 'Unknown type of input_shape'
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    def decorator(module: nn.Module):
        assert issubclass(module, nn.Module), 'module must be a subclass of Pytorch\'s Module'

        @utils.add_simple_repr
        class _Wrapper(module, _LayerMethod):
            def __init__(self):
                self.input_shape = input_shape
                self.output_shape_tmp = output_shape
                device = kwargs.pop('device', None)

                super().__init__(*args, **kwargs)
                if cuda_available:
                    self.cuda(device)

            def forward(self, input, *args, **kwargs):
                return super().forward(input, *args, **kwargs)

            @property
            @utils.validate
            def output_shape(self):
                if self.input_shape is None and self.output_shape_tmp is None:
                    return None

                if self.output_shape_tmp is not None:
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

        _Wrapper.__name__ = module.__name__
        _Wrapper.__doc__ = module.__doc__
        _Wrapper.__module__ = module.__module__
        return _Wrapper
    return decorator


class Lambda(Module):
    """
    Wraps a function as a :class:`~neuralnet_pytorch.layers.Module`.

    Parameters
    ----------
    func
        a callable function.
    input_shape
        shape of the input tensor.
    output_shape
        shape of the output tensor.
        If ``None``, the output shape is calculated by performing a forward pass.
    kwargs
        keyword arguments required by `func`.

    Examples
    --------
    You can easily wrap a :mod:`torch` function

    .. code-block:: python

        import torch as T
        import neuralnet_pytorch as nnt

        a, b = T.rand(3, 1), T.rand(3, 2)
        cat = nnt.Lambda(T.cat, dim=1)
        c = cat((a, b))
        print(c.shape)

    Also, it works for any self-defined function as well

    .. code-block:: python

        import neuralnet_pytorch as nnt

        def foo(x, y):
            return x + y

        a = T.rand(3, 3)
        print(a)
        foo_sum = nnt.Lambda(foo, y=1.)
        res = foo_sum(a)
        print(res)
    """

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
        if self.input_shape is None and self.output_shape_tmp is None:
            return None

        if self.output_shape_tmp is not None:
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
    """
    Extends :class:`torch.nn.Conv2d` with :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', dilation=1, groups=1,
                 bias=True, activation=None, weights_init=None, bias_init=None, **kwargs):
        input_shape = _image_shape(input_shape)
        assert input_shape[1] is not None, 'Shape at dimension 1 (zero-based index) must be known'

        self.input_shape = input_shape
        kernel_size = _pair(kernel_size)
        self.no_bias = bias
        self.activation = utils.function(activation, **kwargs)
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.border_mode = padding
        dilation = _pair(dilation)
        groups = groups

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
            padding = _pair(padding)
        else:
            raise ValueError('padding must be a str/tuple/int, got %s' % type(padding))

        super().__init__(int(input_shape[1]), out_channels, kernel_size, stride, tuple(padding), dilation, bias=bias,
                         groups=groups)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        if self.border_mode in ('ref', 'rep'):
            padding = (self.ks[1] // 2, self.ks[1] // 2, self.ks[0] // 2, self.ks[0] // 2)
            pad = nn.ReflectionPad2d(padding) if self.border_mode == 'ref' else nn.ReplicationPad2d(padding)
        else:
            pad = lambda x: x

        input = pad(input)
        input = self.activation(super().forward(input))
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
    """
    Extends :class:`torch.nn.ConvTranspose2d` with :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        ``dilation * (kernel_size - 1) - padding`` zero-padding
        will be added to both sides of each dimension in the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'`` which are common padding schemes.
        Default: ``'half'``.
    output_padding
        additional size added to one side of each dimension in the output shape. Default: 0
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    dilation
        spacing between kernel elements. Default: 1.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    output_size
        size of the output tensor. If ``None``, the shape is automatically determined.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', output_padding=0, groups=1,
                 bias=True, dilation=1, activation='linear', weights_init=None, bias_init=None, output_size=None,
                 **kwargs):
        input_shape = _image_shape(input_shape)
        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.activation = utils.function(activation, **kwargs)
        self.output_size = _pair(output_size) if output_size is not None else None

        kernel_size = _pair(kernel_size)
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
            input, output_size=self.output_size if output_size is None else output_size))
        return output

    @property
    @utils.validate
    def output_shape(self):
        if self.output_size is not None:
            assert len(self.output_size) == 2, \
                'output_size should have exactly 2 elements, got %d' % len(self.output_size)
            return (self.input_shape[0], self.out_channels) + tuple(self.output_size)

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
    """
    AKA fully connected layer in deep learning literature.
    This class extends :class:`torch.nn.Linear` by :class:`~neuralnet_pytorch.layers.layers._LayerMethod`.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    bias : bool
        if set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    flatten : bool
        whether to flatten input tensor into 2D matrix. Default: ``False``.
    keepdim : bool
        whether to keep the output dimension when `out_features` is 1.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 flatten=False, keepdim=True, **kwargs):
        input_shape = _matrix_shape(input_shape)
        assert input_shape[-1] is not None, 'Shape at the last position (zero-based index) must be known'

        self.input_shape = input_shape
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function(activation, **kwargs)

        super().__init__(int(np.prod(input_shape[1:])) if flatten else input_shape[-1], out_features, bias)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        if self.flatten:
            input = T.flatten(input, 1)

        output = self.activation(super().forward(input))
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
    """
    A special case of :class:`~neuralnet_pytorch.layers.FC` with softmax activation function.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    dim : int
        dimension to apply softmax. Default: 1.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, dim=1, weights_init=None, bias_init=None, **kwargs):
        self.dim = dim
        kwargs['dim'] = dim
        super().__init__(input_shape, out_features, activation='softmax', weights_init=weights_init, bias_init=bias_init,
                         **kwargs)


@utils.add_simple_repr
@utils.no_dim_change_op
class Activation(Module):
    """
    Applies a non-linear function to the incoming input.

    Parameters
    ----------
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    input_shape
        shape of the input tensor. Can be ``None``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, activation='relu', input_shape=None, **kwargs):
        super().__init__(input_shape)
        self.activation = utils.function(activation, **kwargs)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def forward(self, input, *args, **kwargs):
        return self.activation(input)

    def extra_repr(self):
        s = 'activation={}'.format(self.activation.__name__)
        return s


@utils.add_custom_repr
class ConvNormAct(Sequential):
    """
    Fuses convolution, normalization and activation together.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum : float
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    norm_method
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding='half', dilation=1, groups=1, bias=True,
                 activation='relu', weights_init=None, bias_init=None, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, no_scale=False, norm_method='bn', **kwargs):
        super().__init__(input_shape=input_shape)
        from neuralnet_pytorch.layers.normalization import BatchNorm2d, InstanceNorm2d, LayerNorm

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.activation = utils.function(activation, **kwargs)
        self.norm_method = norm_method
        self.conv = Conv2d(input_shape, out_channels, kernel_size, weights_init=weights_init, bias=bias,
                           bias_init=bias_init, padding=padding, stride=stride, dilation=dilation, activation=None,
                           groups=groups, **kwargs)

        norm_method = BatchNorm2d if norm_method == 'bn' else InstanceNorm2d if norm_method == 'in' \
            else LayerNorm if norm_method == 'ln' else norm_method
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
    """
    Fuses fully connected, normalization and activation together.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    bias : bool
        if set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    flatten : bool
        whether to flatten input tensor into 2D matrix. Default: ``False``.
    keepdim : bool
        whether to keep the output dimension when `out_features` is 1.
    eps
        a value added to the denominator for numerical stability.
        Default: 1e-5.
    momentum : float
        the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1.
    affine
        a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``.
    track_running_stats
        a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics and always uses batch
        statistics in both training and eval modes. Default: ``True``.
    no_scale: bool
        whether to use a trainable scale parameter. Default: ``True``.
    norm_method
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, weights_init=None, bias_init=None,
                 flatten=False, keepdim=True, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 no_scale=False, norm_method='bn', **kwargs):
        super().__init__(input_shape=input_shape)
        from neuralnet_pytorch.layers.normalization import BatchNorm1d, InstanceNorm1d, LayerNorm, FeatureNorm1d

        self.out_features = out_features
        self.flatten = flatten
        self.keepdim = keepdim
        self.activation = utils.function(activation, **kwargs)

        self.fc = FC(self.input_shape, out_features, bias, weights_init=weights_init, bias_init=bias_init,
                     flatten=flatten, keepdim=keepdim)

        norm_method = BatchNorm1d if norm_method == 'bn' else InstanceNorm1d if norm_method == 'in' \
            else LayerNorm if norm_method == 'ln' else FeatureNorm1d if norm_method == 'fn' else norm_method
        self.norm = norm_method(self.fc.output_shape, eps, momentum, affine, track_running_stats,
                                no_scale=no_scale, activation=self.activation, **kwargs)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def extra_repr(self):
        s = '{in_features}, {out_features}'
        if self.flatten:
            s += 'flatten={flatten}'
        if not self.keepdim:
            s += 'keepdim={keepdim}'

        s = s.format(**self.fc.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


@utils.add_custom_repr
class ResNetBasicBlock(Module):
    """
    A basic block to build ResNet (https://arxiv.org/abs/1512.03385).

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    downsample
        a module to process the residual branch when output shape is different from input shape.
        If ``None``, a simple :class:`ConvNormAct` is used.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    block
        a function to construct the main branch of the block.
        If ``None``, a simple block as described in the paper is used.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    norm_method
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.

    Attributes
    ----------
    expansion : int
        expansion coefficients of the number of output channels.
        Default: 1.
    """

    expansion = 1

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, padding='half', dilation=1, activation='relu',
                 downsample=None, groups=1, block=None, weights_init=None, norm_method='bn', **kwargs):
        input_shape = _image_shape(input_shape)

        super().__init__(input_shape=input_shape)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = _pair(dilation)
        self.activation = utils.function(activation, **kwargs)
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
                                              padding=padding, weights_init=weights_init, activation='linear')
            else:
                self.downsample = Lambda(lambda x: x, output_shape=input_shape, input_shape=input_shape)

        if cuda_available:
            self.cuda(kwargs.pop('device', None))

    def _make_block(self):
        block = Sequential(input_shape=self.input_shape)
        if self.expansion != 1:
            block.add_module('pre',
                             ConvNormAct(block.output_shape, self.out_channels, 1, stride=1, bias=False,
                                         padding=self.padding, weights_init=self.weights_init, groups=self.groups))
        block.add_module('conv_norm_act_1',
                         ConvNormAct(block.output_shape, self.out_channels, self.kernel_size, bias=False,
                                     padding=self.padding, weights_init=self.weights_init, stride=self.stride,
                                     activation=self.activation, groups=self.groups, norm_method=self.norm_method,
                                     **self.kwargs))
        block.add_module('conv_norm_act_2',
                         ConvNormAct(block.output_shape, self.out_channels * self.expansion,
                                     1 if self.expansion != 1 else self.kernel_size, bias=False, stride=1,
                                     padding=self.padding, activation=None, groups=self.groups,
                                     weights_init=self.weights_init, norm_method=self.norm_method, **self.kwargs))
        return block

    def forward(self, input, *args, **kwargs):
        res = input
        out = self.block(input)
        out += self.downsample(res)
        return self.activation(out)

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
    """
        A bottleneck block to build ResNet (https://arxiv.org/abs/1512.03385).

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    downsample
        a module to process the residual branch when output shape is different from input shape.
        If ``None``, a simple :class:`ConvNormAct` is used.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    block
        a function to construct the main branch of the block.
        If ``None``, a simple block as described in the paper is used.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    norm_method
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    kwargs
        extra keyword arguments to pass to activation.

    Attributes
    ----------
    expansion : int
        expansion coefficients of the number of output channels.
        Default: 4.
    """

    expansion = 4

    def __init__(self, input_shape, out_channels, kernel_size=3, stride=1, padding='half', dilation=1, activation='relu',
                 downsample=None, groups=1, block=None, weights_init=None, norm_method='bn', **kwargs):
        super().__init__(input_shape, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         activation=activation, downsample=downsample, groups=groups, block=block,
                         weights_init=weights_init, norm_method=norm_method, **kwargs)


@utils.add_custom_repr
class StackingConv(Sequential):
    """
    Stacks multiple convolution layers together.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size
        size of the convolving kernel.
    num_layer : int
        number of convolutional layers.
    stride
        stride of the convolution. Default: 1.
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    norm_method
        normalization method to be used. Choices are ``'bn'``, ``'in'``, and ``'ln'``.
        Default: ``'bn'``.
    groups : int
        number of blocked connections from input channels to output channels. Default: 1.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, num_layers, stride=1, padding='half', dilation=1,
                 bias=True, activation='relu', weights_init=None, bias_init=None, norm_method=None, groups=1, **kwargs):
        assert num_layers > 1, 'num_layers must be greater than 1, got %d' % num_layers
        input_shape = _image_shape(input_shape)

        super().__init__(input_shape=input_shape)
        self.num_filters = out_channels
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation = utils.function(activation, **kwargs)
        self.num_layers = num_layers
        self.norm_method = norm_method

        shape = tuple(input_shape)
        conv_layer = partial(ConvNormAct, norm_method=norm_method) if norm_method is not None else Conv2d
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
    """
    Sums the outputs of multiple modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

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
    """
    Sums the intermediate outputs of multiple sequential modules given an input tensor.
    A subclass of :class:`~neuralnet_pytorch.layers.Sum`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.ConcurrentSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

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
    """
    Sums the outputs of multiple modules given input tensors.
    A subclass of :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`.

    See Also
    --------
    :class:`~neuralnet_pytorch.layers.MultiSingleInputModule`
    :class:`~neuralnet_pytorch.layers.MultiMultiInputModule`
    :class:`~neuralnet_pytorch.layers.Sum`
    :class:`~neuralnet_pytorch.layers.SequentialSum`
    :class:`~neuralnet_pytorch.resizing.Cat`
    :class:`~neuralnet_pytorch.resizing.SequentialCat`
    :class:`~neuralnet_pytorch.resizing.ConcurrentCat`
    """

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
    """
    Performs depthwise separable convolution in image processing.

    Parameters
    ----------
    input_shape
        shape of the 4D input image.
        If a single integer is passed, it is treated as the number of input channels
        and other sizes are unknown.
    out_channels : int
        number of channels produced by the convolution.
    kernel_size : int
        size of the convolving kernel.
    depth_mul
        depth multiplier for intermediate result of depthwise convolution
    padding
        zero-padding added to both sides of the input.
        Besides tuple/list/integer, it accepts ``str`` such as ``'half'`` and ``'valid'``,
        which is similar the Theano, and ``'ref'`` and ``'rep'``, which are common padding schemes.
        Default: ``'half'``.
    dilation
        spacing between kernel elements. Default: 1.
    bias : bool
        if ``True``, adds a learnable bias to the output. Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_channels, kernel_size, depth_mul=1, stride=1, padding='half', dilation=1,
                 bias=True, activation=None, **kwargs):
        input_shape = _image_shape(input_shape)

        super().__init__(input_shape=input_shape)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depth_mul = depth_mul
        self.padding = padding
        self.dilation = dilation
        self.activation = utils.function(activation, **kwargs)
        self.depthwise = Conv2d(self.output_shape, input_shape[1] * depth_mul, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=input_shape[1], bias=bias)
        self.pointwise = Conv2d(self.output_shape, out_channels, 1, activation=activation, padding=padding,
                                dilation=dilation, bias=False, **kwargs)

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
    """
    Performs X-Convolution on unordered set as in `this paper`_.

    .. _this paper: https://papers.nips.cc/paper/7362-pointcnn-convolution-on-x-transformed-points.pdf

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    feature_dim : int
        dimension of the input features.
    out_channels : int
        number of channels produced by the convolution.
    out_features : int
        size of each output sample.
    num_neighbors : int
        size of the convolving kernel.
    depth_mul
        depth multiplier for intermediate result of depthwise convolution
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    dropout : bool
        whether to use dropout.
    bn : bool
        whether to use batch normalization.
    kwargs
        extra keyword arguments to pass to activation.
    """
    def __init__(self, input_shape, feature_dim, out_channels, out_features, num_neighbors, depth_mul,
                 activation='relu', dropout=None, bn=True, **kwargs):
        input_shape = _pointset_shape(input_shape)

        super().__init__(input_shape)
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.out_features = out_features
        self.depth_mul = depth_mul
        self.activation = utils.function(activation, **kwargs)
        self.dropout = dropout
        self.bn = bn

        self.fcs = Sequential(input_shape=input_shape)
        self.fcs.add_module('fc1', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout1', wrapper(self.output_shape, nn.Dropout2d, p=dropout))
        self.fcs.add_module('fc2', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout2', wrapper(self.output_shape, nn.Dropout2d, p=dropout))

        from neuralnet_pytorch.layers.resizing import DimShuffle
        from neuralnet_pytorch.layers.normalization import BatchNorm2d

        self.x_trans = Sequential(input_shape=input_shape[:2] + (num_neighbors, input_shape[-1]))
        self.x_trans.add_module('dimshuffle1', DimShuffle(self.x_trans.output_shape, (0, 3, 1, 2)))
        self.x_trans.add_module('conv', Conv2d(self.x_trans.output_shape, num_neighbors ** 2, (1, num_neighbors),
                                               activation=activation, padding='valid', **kwargs))
        self.x_trans.add_module('dimshuffle2', DimShuffle(self.x_trans.output_shape, (0, 2, 3, 1)))
        self.x_trans.add_module('fc1', FC(self.x_trans.output_shape, num_neighbors ** 2, activation='relu', **kwargs))
        self.x_trans.add_module('fc2', FC(self.x_trans.output_shape, num_neighbors ** 2, **kwargs))

        self.end_conv = Sequential(input_shape=input_shape[:2] + (num_neighbors, feature_dim + out_features))
        self.end_conv.add_module('dimshuffle1', DimShuffle(self.end_conv.output_shape, (0, 3, 1, 2)))
        self.end_conv.add_module('conv',
                                 DepthwiseSepConv2D(self.end_conv.output_shape, out_channels, (1, num_neighbors),
                                                    depth_mul=depth_mul, activation=None if bn else activation,
                                                    padding='valid', **kwargs))
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
    Performs graph convolution as described in https://arxiv.org/abs/1609.02907.
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, **kwargs):
        super().__init__(input_shape, out_features, bias, activation=activation, **kwargs)

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


class BatchGraphConv(GraphConv):
    """
    Performs graph convolution as described in https://arxiv.org/abs/1609.02907 on a batch of graphs.
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        The first dim should be batch size.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, **kwargs):
        input_shape = _pointset_shape(input_shape)
        super().__init__(input_shape, out_features, bias, activation=activation, **kwargs)

    def forward(self, input, adj, *args, **kwargs):
        """
        Performs graphs convolution.

        :param input:
            a ``list``/``tuple`` of 2D matrices.
        :param adj:
            a block diagonal matrix whose diagonal consists of
            adjacency matrices of the input graphs.
        :return:
            a batch of embedded graphs.
        """

        shapes = [input_.shape[0] for input_ in input]
        X = T.cat(input, 0)
        output = super().forward(X, adj)
        output = T.split(output, shapes)
        return output


@utils.add_custom_repr
class GraphXConv(Module):
    """
    Performs GraphX Convolution as described here_.
    **Disclaimer:** I am the first author of the paper.

    .. _here:
        http://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_GraphX-Convolution_for_Point_Cloud_Deformation_in_2D-to-3D_Conversion_ICCV_2019_paper.html

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        The first dim should be batch size.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    out_instances : int
        resolution of the output point clouds.
        If not specified, output will have the same resolution as input.
        Default: ``None``.
    rank
        if specified and smaller than `num_out_points`, the mixing matrix will
        be broken into a multiplication of two matrices of sizes
        ``(num_out_points, rank)`` and ``(rank, input_shape[1])``.
    bias
        whether to use bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, out_instances=None, rank=None, bias=True, activation=None,
                 weights_init=None, bias_init=None, **kwargs):
        input_shape = _pointset_shape(input_shape)
        super().__init__(input_shape=input_shape)

        self.out_features = out_features
        self.out_instances = out_instances if out_instances else input_shape[-2]
        if rank:
            assert rank <= self.out_instances // 2, 'rank should be smaller than half of num_out_points'

        self.rank = rank
        self.activation = utils.function(activation, **kwargs)
        pattern = list(range(len(input_shape)))
        pattern[-1], pattern[-2] = pattern[-2], pattern[-1]
        self.pattern = pattern
        self.weights_init = weights_init
        self.bias_init = bias_init

        self.weight = nn.Parameter(T.Tensor(out_features, input_shape[-1]))
        if self.rank is None:
            self.mixing = nn.Parameter(T.Tensor(self.out_instances, input_shape[-2]))
        else:
            self.mixing_u = nn.Parameter(T.Tensor(self.rank, input_shape[-2]))
            self.mixing_v = nn.Parameter(T.Tensor(self.out_instances, self.rank))

        if bias:
            self.bias = nn.Parameter(T.Tensor(out_features))
            self.mixing_bias = nn.Parameter(T.Tensor(self.out_instances))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mixing_bias', None)

        self.reset_parameters()
        if cuda_available:
            self.cuda()

    def forward(self, input):
        output = utils.dimshuffle(input, self.pattern)
        mixing = T.mm(self.mixing_v, self.mixing_u) if self.rank else self.mixing
        output = T.matmul(output, mixing.t())
        if self.mixing_bias is not None:
            output = output + self.mixing_bias

        output = utils.dimshuffle(output, self.pattern)
        output = T.matmul(output, self.weight.t())
        if self.bias is not None:
            output = output + self.bias

        return self.activation(output)

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[:-2] + (self.out_instances, self.out_features)

    def reset_parameters(self):
        super().reset_parameters()
        weights_init = partial(nn.init.kaiming_uniform_, a=np.sqrt(5)) if self.weights_init is None \
            else self.weights_init

        weights_init(self.weight)
        if self.rank:
            weights_init(self.mixing_u)
            weights_init(self.mixing_v)
        else:
            weights_init(self.mixing)

        if self.bias is not None:
            if self.bias_init is None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                bias_init = partial(nn.init.uniform_, a=-bound, b=bound)
            else:
                bias_init = self.bias_init

            bias_init(self.bias)
            nn.init.zeros_(self.mixing_bias)

    def extra_repr(self):
        s = '{input_shape}, out_features={out_features}, out_instances={out_instances}, rank={rank}'
        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s
