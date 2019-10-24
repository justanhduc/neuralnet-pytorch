import numpy as np
import torch as T
import torch.nn.functional as F
import abc
import threading
import warnings
from torch.utils.data.dataloader import default_collate
import numbers
from queue import Queue
from scipy.stats import truncnorm
from torch._six import container_abcs
from itertools import repeat as repeat_
from functools import wraps, partial, update_wrapper

try:
    from slackclient import SlackClient
except (ModuleNotFoundError, ImportError):
    from slack import RTMClient as SlackClient

cuda_available = T.cuda.is_available()


def _wrap(f):
    @wraps(f)
    def wrapper(x, *args, **kwargs):
        return f(x)

    return wrapper


def _make_input_shape(m, n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat_(None, m)) + (x, ) + tuple(repeat_(None, n))
    return parse


_image_shape = _make_input_shape(1, 2)
_matrix_shape = _make_input_shape(1, 0)
_pointset_shape = _make_input_shape(2, 0)


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
            warnings.warn('%s is deprecated and will be removed in version %s. Use %s instead.' %
                          (func.__name__, version, new_func.__name__), DeprecationWarning)
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


class ThreadsafeIter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    Parameters
    ----------
    it
        an iterable object.

    Attributes
    ----------
    lock
        a thread lock.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(generator):
    """
    A decorator that takes a generator function and makes it thread-safe.

    :param generator:
        a generator object.
    :return:
        a thread-safe generator.
    """

    def safe_generator(*agrs, **kwargs):
        return ThreadsafeIter(generator(*agrs, **kwargs))
    return safe_generator


class DataLoader(metaclass=abc.ABCMeta):
    """
    A lightweight data loader. Works comparably to
    Pytorch's :class:`torch.utils.data.Dataloader`
    when the workload is light, but it
    initializes much faster.
    It is totally compatible with :class:`torch.utils.data.Dataset`
    and can be a drop-in for :class:`torch.utils.data.Dataloader`.

    Parameters
    ----------
    dataset
        an instance of :class:`torch.utils.data.Dataset`.
    batch_size
        batch size
    shuffle
        whether to shuffle in each iteration.
    num_workers
        number of threads to be used.
    collate_fn
        function to specify how a batch is loaded.
    num_cached
        number of batches to be cached.
    kwargs
        arguments what will not be used.
        For compatibility with :class:`torch.utils.data.Dataloader` only.

    Attributes
    ----------
    batches
        contains batches of data when iterating over the dataset.
    _indices
        contains the indeices of samples of the dataset.
    num_batches
        the number of batches in the dataset.
    """

    def __init__(self, dataset, batch_size, shuffle=False, num_workers=10, collate_fn=None, num_cached=10, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_cached = num_cached
        self.num_workers = num_workers
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.batches = None
        self._indices = None
        self.num_batches = len(self.dataset) // self.batch_size

        if len(kwargs.keys()) > 0:
            warnings.warn(str(kwargs.keys()) +
                          " are present for compatibility with "
                          "`torch.utils.data.DataLoader` interface"
                          " and will be ignored.",
                          stacklevel=2)

    def __iter__(self):
        self.batches = self._get_batches()
        return self

    def __next__(self):
        if self.batches is None:
            self.batches = self._get_batches()
        return self.batches.__next__()

    def _get_batches(self):
        batches = self._generator()
        batches = self._generate_in_background(batches)
        for it, batch in enumerate(batches):
            yield batch

    def _generate_in_background(self, generator):
        generator = ThreadsafeIter(generator)
        queue = Queue(maxsize=self.num_cached)

        # define producer (putting items into queue)
        def producer():
            for item in generator:
                queue.put(item)
            queue.join()
            queue.put(None)

        # start producer (in a background thread)
        for i in range(self.num_workers):
            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

        # run as consumer (read items from queue, in current thread)
        while True:
            item = queue.get()
            if item is None:
                break
            yield item
            queue.task_done()

    def _generator(self):
        assert isinstance(self.dataset[0],
                          (list, tuple, np.ndarray)), 'Dataset should consist of lists/tuples or Numpy ndarray'
        self._indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self._indices)

        for i in range(self.num_batches):
            slice_ = self._indices[i * self.batch_size:(i + 1) * self.batch_size]
            batch = [self.dataset[s] for s in slice_]
            batch = self.collate_fn(batch)
            yield batch


def truncated_normal(tensor, a=-1, b=1, mean=0., std=1.):
    """
    Initializes a tensor from a truncated normal distribution.

    :param tensor:
        a :class:`torch.Tensor`.
    :param a:
        lower bound of the truncated normal.
    :param b:
        higher bound of the truncated normal.
    :param mean:
        mean of the truncated normal.
    :param std:
        standard deviation of the truncated normal.
    :return:
        ``None``.
    """

    values = truncnorm.rvs(a, b, loc=mean, scale=std, size=list(tensor.shape))
    with T.no_grad():
        tensor.data.copy_(T.tensor(values))


def rgb2gray(img):
    """
    Converts a batch of RGB images to gray.

    :param img:
        a batch of RGB image tensors.
    :return:
        a batch of gray images.
    """

    if len(img.shape) != 4:
        raise ValueError('Input images must have four dimensions, not %d' % len(img.shape))

    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)


def rgb2ycbcr(img):
    """
    Converts a batch of RGB images to YCbCr.

    :param img:
        a batch of RGB image tensors.
    :return:
        a batch of YCbCr images.
    """

    if len(img.shape) != 4:
        raise ValueError('Input images must have four dimensions, not %d' % len(img.shape))

    Y = 0. + .299 * img[:, 0] + .587 * img[:, 1] + .114 * img[:, 2]
    Cb = 128. - .169 * img[:, 0] - .331 * img[:, 1] + .5 * img[:, 2]
    Cr = 128. + .5 * img[:, 0] - .419 * img[:, 1] - .081 * img[:, 2]
    return T.cat((Y.unsqueeze(1), Cb.unsqueeze(1), Cr.unsqueeze(1)), 1)


def ycbcr2rgb(img):
    """
    Converts a batch of YCbCr images to RGB.

    :param img:
        a batch of YCbCr image tensors.
    :return:
        a batch of RGB images.
    """

    if len(img.shape) != 4:
        raise ValueError('Input images must have four dimensions, not %d' % len(img.shape))

    R = img[:, 0] + 1.4 * (img[:, 2] - 128.)
    G = img[:, 0] - .343 * (img[:, 1] - 128.) - .711 * (img[:, 2] - 128.)
    B = img[:, 0] + 1.765 * (img[:, 1] - 128.)
    return T.cat((R.unsqueeze(1), G.unsqueeze(1), B.unsqueeze(1)), 1)


def rgba2rgb(img):
    """
    Converts a batch of RGBA images to RGB.

    :param img:
        an batch of RGBA image tensors.
    :return:
        a batch of RGB images.
    """

    r = img[..., 0, :, :]
    g = img[..., 1, :, :]
    b = img[..., 2, :, :]
    a = img[..., 3, :, :]

    shape = img.shape[:-3] + (3,) + img.shape[-2:]
    out = T.zeros(*shape).to(img.device)
    out[..., 0, :, :] = (1 - a) * r + a * r
    out[..., 1, :, :] = (1 - a) * g + a * g
    out[..., 2, :, :] = (1 - a) * b + a * b
    return out


def gram_matrix(x):
    """
    Computes the Gram matrix given a 4D tensor.

    :param x:
        a 4D tensor.
    :return:
        the Gram matrix of `x`.
    """

    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = T.mm(features, features.t())
    return G.div(a * b * c * d)


def var(x, dim=None, unbiased=True, keepdim=False):
    """
    Calculates the variance of `x` along `dim`.
    Exists because :mod:`torch.var` sometimes causes some error in backward pass.

    :param x:
        a tensor.
    :param dim:
        the dimension along which to calculate the variance.
        Can be ``int``/``list``/``tuple``.
    :param unbiased:
        whether to use an unbiased estimate.
        Default: ``True``.
    :param keepdim:
        whether to keep the reduced dims as ``1``.
        Default: ``False``.
    :return:
        the variance of `x`
    """

    if dim is None:
        dim = tuple(i for i in range(len(x.shape)))

    if isinstance(dim, numbers.Number):
        dim = (int(dim),)

    mean = T.mean(x, dim, keepdim=True)
    dim_prod = np.prod([x.shape[i] for i in dim])
    if unbiased:
        dim_prod -= 1

    var = T.sum((x - mean) ** 2, dim, keepdim=keepdim) / dim_prod
    return var


def std(x, dim=None, unbiased=True, keepdim=False):
    """
    Calculates the standard deviation of `x` along `dim`.
    Exists because :mod:`torch.std` sometimes causes some error in backward pass.

    :param x:
        a tensor.
    :param dim:
        the dimension along which to calculate the variance.
        Can be ``int``/``list``/``tuple``.
    :param unbiased:
        whether to use an unbiased estimate.
        Default: ``True``.
    :param keepdim:
        whether to keep the reduced dims as ``1``.
        Default: ``False``.
    :return:
        the standard deviation of `x`
    """

    return T.sqrt(var(x, dim, unbiased, keepdim) + 1e-8)


def batch_set_value(params, values):
    """
    Sets values of a tensor to another.

    :param params:
        a :class:`torch.Tensor`.
    :param values:
        a :class:`torch.Tensor` of the same shape as `params`.
    :return:
        ``None``.
    """

    for p, v in zip(params, values):
        p.data.copy_(T.from_numpy(v))


def to_numpy(x):
    """
    Moves a tensor to :mod:`numpy`.

    :param x:
        a :class:`torch.Tensor`.
    :return:
        a :class:`numpy.ndarray`.
    """

    return x.cpu().detach().data.numpy()


def to_cuda(x):
    """
    Moves a :mod:`numpy` to tensor.

    :param x:
        a :class:`numpy.ndarray`.
    :return:
        a :class:`torch.Tensor`.
    """

    return T.from_numpy(x).cuda()


def to_cuda_sparse(coo):
    """
    Moves a sparse matrix to cuda tensor.

    :param x:
        a :class:`scipy.sparse.coo.coo_matrix`.
    :return:
        a :class:`torch.Tensor`.
    """

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = T.LongTensor(list(indices))
    v = T.FloatTensor(values)
    shape = coo.shape
    return T.sparse.FloatTensor(i, v, T.Size(shape))


def bulk_to_numpy(xs):
    """
    Moves a list of tensors to :class:`numpy.ndarray`.

    :param xs:
        a list/tuple of :class:`torch.Tensor` s.
    :return:
        a tuple of :class:`numpy.ndarray` s.
    """

    return tuple([to_numpy(x) for x in xs])


def bulk_to_cuda(xs):
    """
    Moves a list of :class:`numpy.ndarray` to tensors.

    :param xs:
        a tuple of :class:`numpy.ndarray` s.
    :return:
        a list/tuple of :class:`torch.Tensor` s.
    """

    return tuple([to_cuda(x) for x in xs])


def bulk_to_cuda_sparse(xs):
    """
    Moves a list sparse matrices to cuda tensor.

    :param x:
        a list/tuple of :class:`scipy.sparse.coo.coo_matrix`.
    :return:
        a :class:`torch.Tensor`.
    """

    return tuple([to_cuda_sparse(x) for x in xs])


def batch_to_cuda(batch):
    batch_cuda = [b.cuda() if not isinstance(b, (list, tuple))
                  else [bb.cuda() for bb in b] for b in batch]
    return batch_cuda


def dimshuffle(x, pattern):
    """
    Reorders the dimensions of this variable, optionally inserting broadcasted dimensions.
    Inspired by `Theano's dimshuffle`_.

    .. _Theano's dimshuffle:
        https://github.com/Theano/Theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/var.py#L323-L356

    :param x:
        Input tensor.
    :param pattern:
        List/tuple of int mixed with `x` for broadcastable dimensions.
    :return:
        a tensor whose shape matches `pattern`.

    Examples
    --------
    To create a 3D view of a [2D] matrix, call ``dimshuffle(x, [0,'x',1])``.
    This will create a 3D view such that the
    middle dimension is an implicit broadcasted dimension.  To do the same
    thing on the transpose of that matrix, call ``dimshuffle(x, [1, 'x', 0])``.

    See Also
    --------
    :func:`~neuralnet_pytorch.utils.shape_padleft`
    :func:`~neuralnet_pytorch.utils.shape_padright`
    """

    assert isinstance(pattern, (list, tuple)), 'pattern must be a list/tuple'
    no_expand_pattern = [x for x in pattern if x != 'x']
    y = x.permute(*no_expand_pattern)
    shape = list(y.shape)
    for idx, e in enumerate(pattern):
        if e == 'x':
            shape.insert(idx, 1)
    return y.view(*shape)


def shape_padleft(x, n_ones=1):
    """
    Reshape `x` by left-padding the shape with `n_ones` 1s.
    Inspired by `Theano's shape_padleft`_.

    .. _Theano's shape_padleft:
        https://github.com/Theano/Theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/basic.py#L4539-L4553

    :param x:
        variable to be reshaped.
    :param n_ones:
        number of 1s to pad.

    See Also
    --------
    :func:`~neuralnet_pytorch.utils.dimshuffle`
    :func:`~neuralnet_pytorch.utils.shape_padright`
    """

    pattern = ('x',) * n_ones + tuple(range(x.ndimension()))
    return dimshuffle(x, pattern)


def shape_padright(x, n_ones=1):
    """
    Reshape `x` by right-padding the shape with `n_ones` 1s.
    Inspired by `Theano's shape_padright`_.

    .. _Theano's shape_padright:
        https://github.com/Theano/Theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/basic.py#L4557

    :param x:
        variable to be reshaped.
    :param n_ones:
        number of 1s to pad.

    See Also
    --------
    :func:`~neuralnet_pytorch.utils.dimshuffle`
    :func:`~neuralnet_pytorch.utils.shape_padleft`
    """

    pattern = tuple(range(x.ndimension())) + ('x',) * n_ones
    return dimshuffle(x, pattern)


def ravel_index(index, shape):
    """
    Finds the linear index of `index` of a tensor of `shape`
    when it is flattened.

    :param index:
        a tuple of ``int`` s.
    :param shape:
        shape of the tensor.
    :return:
        the linear index of the element having `index`.

    Examples
    --------

    >>> import torch as T
    >>> import numpy as np
    >>> import neuralnet_pytorch as nnt
    >>> shape = (2, 3, 5)
    >>> a = T.arange(np.prod(shape)).view(*shape)
    >>> index = (1, 1, 3)
    >>> print(a[index])  # 23
    >>> linear_index = nnt.utils.ravel_index(index, shape)
    >>> print(a.flatten()[linear_index])  # 23
    """

    assert len(index) == len(shape), 'indices and shape must have the same length'
    shape = T.tensor(shape)
    if cuda_available:
        shape = shape.cuda()

    return sum([T.tensor(index[i], dtype=shape.dtype) * T.prod(shape[i + 1:]) for i in range(len(shape))])


def tile(x, dims):
    """
    Repeats `x` along `dims`.

    :param x:
        a :mod:`torch.Tensor`.
    :param dims:
        the number of times to tile this tensor along each dimension.
    :return:
        the tiled tensor.
    """

    return x.repeat(*dims)


def repeat(input, repeats, dim=None):
    """
    Repeats elements of a tensor like `numpy.repeat`.

    :param input:
        a :mod:`torch.Tensor`.
    :param repeats:
        the number of times to repeat this tensor along `dim`.
    :param dim:
        the dimension to repeat.
        If not specified, the method is applied to the flattened tensor.
        Default: ``None``.
    :return:
        the repeated tensor.
    """

    return T.repeat_interleave(input, repeats, dim)


def block_diag(*blocks):
    """
    Modified from scipy.linalg.block_diag.
    Creates a block diagonal matrix from provided arrays.
    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    :param blocks:
        an iterator of tensors, up to 2-D
        a 1-D tensor of length `n` is treated as a 2-D array
        with shape `(1,n)`.
    :return:
        a tensor with `A`, `B`, `C`, ... on the diagonal.
        Has the same dtype as `A`.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Examples
    --------

    >>> from neuralnet_pytorch.utils import block_diag
    >>> A = T.tensor([[1, 0],
    ...               [0, 1]])
    >>> B = T.tensor([[3, 4, 5],
    ...               [6, 7, 8]])
    >>> C = T.tensor([[7]])
    >>> block_diag(A, B, C)
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(T.tensor([1.0]), T.tensor([2, 3]), T.tensor([[4, 5], [6, 7]]))
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])
    """

    assert all(a.ndimension() >= 2 for a in blocks), 'All tensors must be at least of rank 2'

    shapes = np.array([a.shape for a in blocks])
    out = T.zeros(*list(np.sum(shapes, axis=0))).to(blocks[0].device)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = blocks[i]
        r = r + rr
        c = c + cc
    return out


def smooth(x, beta=.9, window='hanning'):
    """
    Smoothens the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    :param x:
        the input signal.
    :param beta:
        the weighted moving average coeff. Window length is :math:`1 / (1 - \\beta)`.
    :param window:
        the type of window from ```flat```, ```hanning```, ```hamming```,
        ```bartlett```, and ```blackman```.
        Flat window will produce a moving average smoothing.
    :return:
        the smoothed signal.

    Examples
    --------

    .. code-block:: python

        t = linspace(-2, 2, .1)
        x = sin(t) + randn(len(t)) * .1
        y = smooth(x)

    """

    x = np.array(x)
    assert x.ndim == 1, 'smooth only accepts 1 dimension arrays'
    assert 0 < beta < 1, 'Input vector needs to be bigger than window size'

    window_len = int(1 / (1 - beta))
    if window_len < 3 or x.shape[0] < window_len:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if isinstance(window, str):
        assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'], \
            'Window is on of \'flat\', \'hanning\', \'hamming\', \'bartlett\', \'blackman\''

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
    else:
        window = np.array(window)
        assert window.ndim == 1, 'Window must be a 1-dim array'
        w = window

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y if y.shape[0] == x.shape[0] else y[(window_len // 2 - 1):-(window_len // 2)]


def get_bilinear_weights(x, y, h, w, border_mode='nearest'):
    """
    Returns bilinear weights used in bilinear interpolation.

    :param x:
        floating point coordinates along the x-axis.
    :param y:
        floating point coordinates along the y-axis.
    :param h:
        height of the 2D array.
    :param w:
        width of the 2D array
    :param border_mode:
        strategy to deal with borders.
        Choices are ```nearest``` (default), ```mirror```, and ```wrap```.
    :return:
        the weights for bilinear interpolation and the integer coordinates.
    """

    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    if border_mode == 'nearest':
        x0 = T.clamp(x0_f, 0, w - 1)
        x1 = T.clamp(x1_f, 0, w - 1)
        y0 = T.clamp(y0_f, 0, h - 1)
        y1 = T.clamp(y1_f, 0, h - 1)
    elif border_mode == 'mirror':
        w = 2 * (w - 1)
        x0 = T.min(x0_f % w, -x0_f % w)
        x1 = T.min(x1_f % w, -x1_f % w)
        h = 2 * (h - 1)
        y0 = T.min(y0_f % h, -y0_f % h)
        y1 = T.min(y1_f % h, -y1_f % h)
    elif border_mode == 'wrap':
        x0 = T.fmod(x0_f, w)
        x1 = T.fmod(x1_f, w)
        y0 = T.fmod(y0_f, h)
        y1 = T.fmod(y1_f, h)
    else:
        raise ValueError("border_mode must be one of "
                         "'nearest', 'mirror', 'wrap'")
    x0, x1, y0, y1 = [v.long() for v in (x0, x1, y0, y1)]

    wxy = dimshuffle((x1_f - x) * (y1_f - y), (0, 'x'))
    wx1y = dimshuffle((x1_f - x) * (1. - (y1_f - y)), (0, 'x'))
    w1xy = dimshuffle((1. - (x1_f - x)) * (y1_f - y), (0, 'x'))
    w1x1y = dimshuffle((1. - (x1_f - x)) * (1. - (y1_f - y)), (0, 'x'))
    return wxy, wx1y, w1xy, w1x1y, x0, x1, y0, y1


def interpolate_bilinear(im, x, y, output_shape=None, border_mode='nearest'):
    """
    Returns a batch of interpolated images. Used for Spatial Transformer Network.
    Works like `torch.grid_sample`.

    :param im:
        a batch of input images
    :param x:
        floating point coordinates along the x-axis.
        Should be in the range [-1, 1].
    :param y:
        floating point coordinates along the y-axis.
        Should be in the range [-1, 1].
    :param output_shape:
        output shape. A tuple of height and width.
        If not specified, output will have the same shape as input.
    :param border_mode:
        strategy to deal with borders.
        Choices are ```nearest``` (default), ```mirror```, and ```wrap```.
    :return:
        the bilinear interpolated batch of images.
    """
    if im.ndim != 4:
        raise TypeError('im should be a 4D Tensor image, got %dD' % im.ndim)

    output_shape = output_shape if output_shape else im.shape[2:]
    x, y = x.flatten(), y.flatten()
    n, c, h, w = im.shape
    h_out, w_out = output_shape

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (w - 1)
    y = (y + 1) / 2 * (h - 1)
    wxy, wx1y, w1xy, w1x1y, x0, x1, y0, y1 = get_bilinear_weights(
        x, y, h, w, border_mode=border_mode)

    base = T.arange(n) * w * h
    base = T.reshape(base, (-1, 1))
    base = repeat(base, (1, h_out * w_out))
    base = base.flatten()

    base_y0 = base + y0 * w
    base_y1 = base + y1 * w
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = T.reshape(dimshuffle(im, (0, 2, 3, 1)), (-1, c))
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    output = wxy * pixel_a + wx1y * pixel_b + w1xy * pixel_c + w1x1y * pixel_d
    output = T.reshape(output, (n, h_out, w_out, c))
    return dimshuffle(output, (0, 3, 1, 2))


def batch_pairwise_dist(x, y):
    """
    Calculates the pair-wise distance between two sets of points.

    :param x:
        a tensor of shape (m, nx, d).
    :param y:
        a tensor of shape (m, ny, d).
    :return:
        the exhaustive distance tensor between every pair of points in `x` and `y`.
    """

    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = T.bmm(x, x.transpose(2, 1))
    yy = T.bmm(y, y.transpose(2, 1))
    zz = T.bmm(x, y.transpose(2, 1))

    if cuda_available:
        dtype = T.cuda.LongTensor
    else:
        dtype = T.LongTensor

    diag_ind_x = T.arange(0, num_points_x).type(dtype)
    diag_ind_y = T.arange(0, num_points_y).type(dtype)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def time_cuda_module(f, *args, **kwargs):
    """
    Measures the time taken by a Pytorch module.

    :param f:
        a Pytorch module.
    :param args:
        arguments to be passed to `f`.
    :param kwargs:
        keyword arguments to be passed to `f`.
    :return:
        the time (in second) that `f` takes.
    """

    start = T.cuda.Event(enable_timing=True)
    end = T.cuda.Event(enable_timing=True)

    start.record()
    f(*args, **kwargs)
    end.record()

    # Waits for everything to finish running
    T.cuda.synchronize()

    total = start.elapsed_time(end)
    print('Took %fms' % total)
    return total


def slack_message(username, message, channel, token, **kwargs):
    """
    Sends a slack message to the specified chatroom.

    :param username:
        Slack username.
    :param message:
        message to be sent.
    :param channel:
        Slack channel.
    :param token:
        Slack chatroom token.
    :param kwargs:
        additional keyword arguments to slack's :meth:`api_call`.
    :return:
        ``None``.
    """

    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, text=message, username=username, **kwargs)


def relu(x, **kwargs):
    """
    ReLU activation.
    """

    return T.relu(x)


def linear(x, **kwargs):
    """
    Linear activation.
    """

    return x


def lrelu(x, **kwargs):
    """
    Leaky ReLU activation.
    """

    return F.leaky_relu(x, kwargs.get('negative_slope', .2), kwargs.get('inplace', False))


def tanh(x, **kwargs):
    """
    Hyperbolic tangent activation.
    """

    return T.tanh(x)


def sigmoid(x, **kwargs):
    """
    Sigmoid activation.
    """

    return T.sigmoid(x)


def elu(x, **kwargs):
    """
    ELU activation.
    """

    return F.elu(x, kwargs.get('alpha', 1.), kwargs.get('inplace', False))


def softmax(x, **kwargs):
    """
    Softmax activation.
    """

    return T.softmax(x, kwargs.get('dim', None))


def selu(x, **kwargs):
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
    returns the `activation`.
    Possible choices are
    ``None``, ```linear```, ```relu```, ```lrelu```,
    ```tanh```, ```sigmoid```, ```elu```, ```softmax```,
    and ```selu```.

    :param activation:
        name of the activation function.
    :return:
        activation function
    """

    func = partial(act[activation], **kwargs)
    return update_wrapper(func, act[activation])
