from .. import cuda_ext_available

import torch as T
import numpy as np
import numbers

__all__ = ['dimshuffle', 'shape_padleft', 'shape_padright', 'swapaxes', 'ravel_index', 'tile', 'repeat', 'block_diag',
           'block_diag_sparse', 'get_bilinear_weights', 'interpolate_bilinear', 'batch_pairwise_dist', 'gram_matrix',
           'var', 'std']


def dimshuffle(x: T.Tensor, pattern):
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
    :func:`~neuralnet_pytorch.utils.swapaxes`
    """

    assert isinstance(pattern, (list, tuple)), 'pattern must be a list/tuple'
    no_expand_pattern = [x for x in pattern if x != 'x']
    y = x.permute(*no_expand_pattern)
    shape = list(y.shape)
    for idx, e in enumerate(pattern):
        if e == 'x':
            shape.insert(idx, 1)
    return y.view(*shape)


def shape_padleft(x: T.Tensor, n_ones=1):
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
    :func:`~neuralnet_pytorch.utils.swapaxes`
    """

    pattern = ('x',) * n_ones + tuple(range(x.ndimension()))
    return dimshuffle(x, pattern)


def shape_padright(x: T.Tensor, n_ones=1):
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
    :func:`~neuralnet_pytorch.utils.swapaxes`
    """

    pattern = tuple(range(x.ndimension())) + ('x',) * n_ones
    return dimshuffle(x, pattern)


def swapaxes(y: T.Tensor, axis1, axis2):
    """
    Swaps two given axes in the input tensor.
    If the input is of shape :math:`(n_1, n_2, ..., n_{axis1}, ..., n_{axis2}, ...)`,
    the output will be :math:`(n_1, n_2, ..., n_{axis2}, ..., n_{axis1}, ...)`.
    Can be seen as a generalization of transpose.
    Taken from `Theano's swapaxes`_.

    .. _Theano's swapaxes:
        http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor._tensor_py_operators.swapaxes

    :param y:
        a tensor.
    :param axis1:
        an axis to be swapped.
    :param axis2:
        another axis to be swapped.
    :return:
        the axis-swapped tensor.

    See Also
    --------
    :func:`~neuralnet_pytorch.utils.dimshuffle`
    :func:`~neuralnet_pytorch.utils.shape_padleft`
    :func:`~neuralnet_pytorch.utils.shape_padright`
    """

    ndim = y.ndimension()
    li = list(range(0, ndim))
    li[axis1], li[axis2] = li[axis2], li[axis1]
    return dimshuffle(y, li)


def ravel_index(indices: T.Tensor, shape):
    """
    Finds the linear index of `index` of a tensor of `shape`
    when it is flattened.

    :param indices:
        a tensor containing indices to be linearized.
    :param shape:
        shape of the tensor w.r.t the index tensor.
    :return:
        the linear index of the element having `index`.

    Examples
    --------

    >>> import torch as T
    >>> import numpy as np
    >>> import neuralnet_pytorch as nnt
    >>> shape = (2, 3, 5)
    >>> a = T.arange(np.prod(shape)).view(*shape)
    >>> indices = T.tensor([[1, 0, 1, 1], [0, 1, 2, 1], [1, 0, 4, 3]]).long()
    >>> print(a[list(indices)])
    tensor([16.,  5., 29., 23.])
    >>> linear_indices = nnt.utils.ravel_index(indices, shape)
    >>> print(linear_indices)
    tensor([16,  5, 29, 23])
    >>> print(a.flatten()[linear_indices])
    tensor([16.,  5., 29., 23.])
    """

    assert indices.shape[0] == len(shape), 'indices and shape must have the same length'
    shape = T.tensor(shape).to(indices.device)
    return sum([indices[i].long() * T.prod(shape[i + 1:]) for i in range(len(shape))])


def tile(x: T.Tensor, dims):
    """
    Repeats `x` along `dims`.
    Behaves like :func:`numpy.tile`.

    :param x:
        a :mod:`torch.Tensor`.
    :param dims:
        the number of times to tile this tensor along each dimension.
    :return:
        the tiled tensor.
    """

    return x.repeat(*dims)


def repeat(input: T.Tensor, repeats, dim=None):
    """
    Repeats elements of a tensor like :func:`numpy.repeat`.

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
        an iterator of tensors, up to 2-D.
        A 1-D tensor of length `n` is treated as a 2-D array
        with shape `(1,n)`.
    :return:
        a tensor with `A`, `B`, `C`, ... on the diagonal.
        Has the same dtype as `A`.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    See Also
    --------
    :func:`~neuralnet_pytorch.utils.block_diag_sparse`

    Examples
    --------

    >>> from neuralnet_pytorch.utils import block_diag
    >>> A = T.tensor([[1, 0],
    ...               [0, 1]])
    >>> B = T.tensor([[3, 4, 5],
    ...               [6, 7, 8]])
    >>> C = T.tensor([[7]])
    >>> block_diag(A, B, C)
    tensor([[1 0 0 0 0 0]
            [0 1 0 0 0 0]
            [0 0 3 4 5 0]
            [0 0 6 7 8 0]
            [0 0 0 0 0 7]])
    >>> block_diag(T.tensor([1.0]), T.tensor([2, 3]), T.tensor([[4, 5], [6, 7]]))
    tensor([[ 1.,  0.,  0.,  0.,  0.],
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


def block_diag_sparse(a: T.Tensor, dense=False):
    """
    Creates a sparse block diagonal matrix from the provided array.
    Given the input tensor of size ``(n, r, c)``, the output will have
    the matrices of the last two indices arranged on the diagonal::

        [[a[0], 0, 0],
         [0, a[1], 0],
         [0, 0, a[2]]]

    :param a:
        a tensor of size ``(n, r, c)``.
    :param dense:
        whether to return a dense matrix.
        Default: ``False``.
    :return:
        a tensor with `a[0]`, `a[1]`, `a[2]`, ... on the diagonal.
        Has the same dtype as `a`.

    Notes
    -----
    This function is for square matrices only. For general cases,
    use :func:`~neuralnet_pytorch.utils.block_diag`.

    See Also
    --------
    :func:`~neuralnet_pytorch.utils.block_diag`

    Examples
    --------

    >>> from neuralnet_pytorch.utils import block_diag_sparse
    >>> import numpy as np
    >>> a = T.arange(3 * 2 * 4).view(3, 2, 4)
    >>> block_diag_sparse(a)
    tensor(indices=tensor([[ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,
                             3,  3,  4,  4,  4,  4,  5,  5,  5,  5],
                           [ 0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  4,  5,
                             6,  7,  8,  9, 10, 11,  8,  9, 10, 11]]),
           values=tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
           size=(6, 12), nnz=24, layout=torch.sparse_coo)
    >>> block_diag_sparse(a, dense=True)
    tensor([[ 0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 4,  5,  6,  7,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  8,  9, 10, 11,  0,  0,  0,  0],
            [ 0,  0,  0,  0, 12, 13, 14, 15,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0, 16, 17, 18, 19],
            [ 0,  0,  0,  0,  0,  0,  0,  0, 20, 21, 22, 23]])
    """
    assert len(a.shape) == 3, \
        'Input tensor must have 3 dimensions with the last two being matrices, got %d'.format(len(a.shape))

    n, r, c = a.shape
    y = T.arange(r)
    x = T.arange(c)
    yy, xx = T.meshgrid(y, x)

    xxs = T.stack([xx] * n)
    yys = T.stack([yy] * n)
    transl_x = T.arange(n) * c
    transl_y = T.arange(n) * r
    xxs_transl = xxs + transl_x[..., None, None]
    yys_transl = yys + transl_y[..., None, None]

    x_flat = xxs_transl.flatten()
    y_flat = yys_transl.flatten()
    indices = T.stack((y_flat, x_flat))

    a_sp = T.sparse_coo_tensor(indices.long(), a.flatten(),
                               size=T.Size((n * r, n * c)), dtype=a.dtype)
    return a_sp.to_dense() if dense else a_sp


def get_bilinear_weights(x: T.Tensor, y: T.Tensor, h: int, w: int, border_mode='nearest'):
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
        Choices are ``'nearest'`` (default), ``'mirror'``, and ``'wrap'``.
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


def interpolate_bilinear(im: T.Tensor, x: T.Tensor, y: T.Tensor, output_shape=None, border_mode='nearest'):
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
        Choices are ``'nearest'`` (default), ``'mirror'``, and ``'wrap'``.
    :return:
        the bilinear interpolated batch of images.
    """

    if im.ndimension() != 4:
        raise TypeError('im should be a 4D Tensor image, got %dD' % im.ndimension())

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


def batch_pairwise_dist(x: T.Tensor, y: T.Tensor, c_code=cuda_ext_available):
    """
    Calculates the pair-wise distance between two sets of points.

    :param x:
        a tensor of shape ``(m, nx, d)``.
    :param y:
        a tensor of shape ``(m, ny, d)``.
    :param c_code:
        whether to use a C++ implementation.
        Default: ``True``.
    :return:
        the exhaustive distance tensor between every pair of points in `x` and `y`.
    """

    assert x.ndimension() in (2, 3) and y.ndimension() in (2, 3), \
        'Input point clouds must be 2D or 3D tensors'
    if x.ndimension() == 2:
        x = x[None]

    if y.ndimension() == 2:
        y = y[None]

    if c_code:
        from ..extensions import batch_pairwise_dist
        return batch_pairwise_dist(x, y)
    else:
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = T.bmm(x, x.transpose(2, 1))
        yy = T.bmm(y, y.transpose(2, 1))
        zz = T.bmm(x, y.transpose(2, 1))

        diag_ind_x = T.arange(0, num_points_x).to(device=x.device, dtype=T.long)
        diag_ind_y = T.arange(0, num_points_y).to(device=x.device, dtype=T.long)

        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


def gram_matrix(x: T.Tensor) -> T.Tensor:
    """
    Computes the Gram matrix given a 4D tensor.

    :param x:
        a 4D tensor.
    :return:
        the Gram matrix of `x`.
    """

    b, c, h, w = x.shape
    features = x.view(b, c, -1)
    G = T.bmm(features, features.transpose(-1, -2))
    return G.div(np.prod(x.shape[1:]))


def var(x: T.Tensor, dim=None, unbiased=True, keepdim=False):
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


def std(x: T.Tensor, dim=None, unbiased=True, keepdim=False):
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
