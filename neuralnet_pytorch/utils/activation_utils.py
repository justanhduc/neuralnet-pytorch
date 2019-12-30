import torch as T
import torch.nn.functional as F
from functools import partial, update_wrapper

__all__ = ['relu', 'linear', 'lrelu', 'tanh', 'sigmoid', 'elu', 'selu', 'softmax', 'function']


def relu(x: T.Tensor, **kwargs):
    """
    ReLU activation.
    """

    return T.relu(x)


def linear(x: T.Tensor, **kwargs):
    """
    Linear activation.
    """

    return x


def lrelu(x: T.Tensor, **kwargs):
    """
    Leaky ReLU activation.
    """

    return F.leaky_relu(x, kwargs.get('negative_slope', .2), kwargs.get('inplace', False))


def tanh(x: T.Tensor, **kwargs):
    """
    Hyperbolic tangent activation.
    """

    return T.tanh(x)


def sigmoid(x: T.Tensor, **kwargs):
    """
    Sigmoid activation.
    """

    return T.sigmoid(x)


def elu(x: T.Tensor, **kwargs):
    """
    ELU activation.
    """

    return F.elu(x, kwargs.get('alpha', 1.), kwargs.get('inplace', False))


def softmax(x: T.Tensor, **kwargs):
    """
    Softmax activation.
    """

    return T.softmax(x, kwargs.get('dim', None))


def selu(x: T.Tensor, **kwargs):
    """
    SELU activation.
    """

    return T.selu(x)


act = {
    'relu': relu,
    'linear': linear,
    None: linear,
    'lrelu': lrelu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'elu': elu,
    'softmax': softmax,
    'selu': selu
}


def function(activation, **kwargs):
    """
    returns the `activation`. Can be ``str`` or ``callable``.
    For ``str``, possible choices are
    ``None``, ``'linear'``, ``'relu'``, ``'lrelu'``,
    ``'tanh'``, ``'sigmoid'``, ``'elu'``, ``'softmax'``,
    and ``'selu'``.

    :param activation:
        name of the activation function.
    :return:
        activation function
    """

    func = partial(activation, **kwargs) if callable(activation) else partial(act[activation], **kwargs)
    return update_wrapper(func, activation) if callable(activation) else update_wrapper(func, act[activation])
