import numpy as np
import numbers
from torch._six import container_abcs
from itertools import repeat as repeat_

from . import root_logger

__all__ = ['validate', 'no_dim_change_op', 'add_simple_repr', 'add_custom_repr', 'deprecated', 'get_non_none']


def _make_input_shape(m, n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x

        return tuple(repeat_(None, m)) + (x, ) + tuple(repeat_(None, n))
    return parse


def validate(func):
    """
    A decorator to make sure output shape is a tuple of ``int`` s.
    """

    def wrapper(self):
        shape = func(self)

        if shape is None:
            return None

        if isinstance(shape, numbers.Number):
            return int(shape)

        out = [None if x is None or np.isnan(x) else int(x) for x in func(self)]
        return tuple(out)

    return wrapper


def no_dim_change_op(cls):
    """
    A decorator to overwrite :attr:`~neuralnet_pytorch.layers._LayerMethod.output_shape`
    to an op that does not change the tensor shape.

    :param cls:
        a subclass of :class:`~neuralnet_pytorch.layers.Module`.
    """

    @validate
    def output_shape(self):
        return None if self.input_shape is None else tuple(self.input_shape)

    cls.output_shape = property(output_shape)
    return cls


def add_simple_repr(cls):
    """
     A decorator to add a simple repr to the designated class.

    :param cls:
        a subclass of :class:`~neuralnet_pytorch.layers.Module`.
     """

    def _repr(self):
        return super(cls, self).__repr__() + ' -> {}'.format(self.output_shape)

    setattr(cls, '__repr__', _repr)
    return cls


def add_custom_repr(cls):
    """
    A decorator to add a custom repr to the designated class.
    User should define extra_repr for the decorated class.

    :param cls:
        a subclass of :class:`~neuralnet_pytorch.layers.Module`.
    """

    def _repr(self):
        return self.__class__.__name__ + '({}) -> {}'.format(self.extra_repr(), self.output_shape)

    setattr(cls, '__repr__', _repr)
    return cls


def deprecated(new_func, version):
    def _deprecated(func):
        """prints out a deprecation warning"""

        def func_wrapper(*args, **kwargs):
            root_logger.warning('%s is deprecated and will be removed in version %s. Use %s instead.' %
                                (func.__name__, version, new_func.__name__), exc_info=True)
            return func(*args, **kwargs)

        return func_wrapper
    return _deprecated


def get_non_none(array):
    """
    Gets the first item that is not ``None`` from the given array.

    :param array:
        an arbitrary array that is iterable.
    :return:
        the first item that is not ``None``.
    """

    try:
        e = next(item for item in array if item is not None)
    except StopIteration:
        e = None
    return e
