import torch as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import neuralnet_pytorch as nnt
from neuralnet_pytorch import utils
from neuralnet_pytorch.extensions.dist_emd import emd

__all__ = ['huber_loss', 'first_derivative_loss', 'lp_loss', 'ssim', 'psnr', 'chamfer_loss', 'emd_loss', 'tv_reg',
           'spectral_norm']


def huber_loss(x, y, reduction='mean'):
    """
    An alias for :func:`torch.nn.functional.smooth_l1_loss`.
    """

    return F.smooth_l1_loss(x, y, reduction)


def first_derivative_loss(x, y, p=2):
    """
    Calculates lp loss between the first derivatives of the inputs.

    :param x:
        a :class:`torch.Tensor`.
    :param y:
        a :class:`torch.Tensor` of the same shape as x.
    :param p:
        order of the norm.
    :return:
        the scalar loss between the first derivatives of the inputs.
    """

    if x.ndimension() != 4 and y.ndimension() != 4:
        raise TypeError('y and y_pred should have four dimensions')

    kern_x = T.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')).requires_grad_(False)
    kern_x = T.flip(kern_x.expand(y.shape[1], y.shape[1], 3, 3), (0, 1)).to(x.device)

    kern_y = T.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')).requires_grad_(False)
    kern_y = T.flip(kern_y.expand(y.shape[1], y.shape[1], 3, 3), (0, 1)).to(x.device)

    x_grad_x = F.conv2d(x, kern_x, padding=1)
    x_grad_y = F.conv2d(x, kern_y, padding=1)
    x_grad = T.sqrt(x_grad_x ** 2 + x_grad_y ** 2 + 1e-10)

    y_grad_x = F.conv2d(y, kern_x, padding=1)
    y_grad_y = F.conv2d(y, kern_y, padding=1)
    y_grad = T.sqrt(y_grad_x ** 2 + y_grad_y ** 2 + 1e-10)
    return lp_loss(x_grad, y_grad, p)


def lp_loss(x, y, p=2, reduction='mean'):
    """
    Calculates p-norm of (x - y).

    :param x:
        a :class:`torch.Tensor`.
    :param y:
        a :class:`torch.Tensor` of the same shape as x.
    :param p:
        order of the norm.
    :param reduction:
        ```mean``` or ```sum```.
    :return:
        the p-norm of (x - y).
    """

    if y.ndimension() != x.ndimension():
        raise TypeError('y should have the same shape as y_pred', ('y', y.data.type(), 'y_pred', x.data.type()))

    if p == 1:
        return F.l1_loss(x, y, reduction=reduction)
    elif p == 2:
        return F.mse_loss(x, y, reduction=reduction)
    else:
        return T.mean(T.abs(x - y) ** p)


def chamfer_loss(xyz1, xyz2, reduce='sum', c_code=True):
    """
    Calculates the Chamfer distance between two batches of point clouds.
    The Pytorch code is adapted from DenseLidarNet_.
    The CUDA code is adapted from AtlasNet_.

    .. _DenseLidarNet: https://github.com/345ishaan/DenseLidarNet/blob/master/code/chamfer_loss.py
    .. _AtlasNet: https://github.com/ThibaultGROUEIX/AtlasNet/tree/master/extension

    :param xyz1:
        a point cloud of shape (b, n1, k).
    :param xyz2:
        a point cloud of shape (b, n2, k).
    :param reduce:
        ```mean``` or ```sum```. Default: ```sum```.
    :param c_code:
        whether to use CUDA implementation.
        This version is much more memory-friendly and slightly faster.
    :return:
        the Chamfer distance between the inputs.
    """

    assert reduce in ('mean', 'sum'), 'Unknown reduce method'
    reduce = T.sum if reduce == 'sum' else T.mean

    if c_code:
        from .extensions import chamfer_distance
        dist1, dist2 = chamfer_distance(xyz1, xyz2)
    else:
        P = utils.batch_pairwise_dist(xyz1, xyz2)
        dist2, _ = T.min(P, 1)
        dist1, _ = T.min(P, 2)
    loss_2 = reduce(dist2)
    loss_1 = reduce(dist1)
    return loss_1 + loss_2


def emd_loss(xyz1, xyz2, reduce='mean'):
    assert len(xyz1.shape) == len(xyz2.shape) == 3, 'Inputs should have 3 dimensions'
    assert reduce in ('mean', 'sum'), 'Reduce method should be mean or sum'

    emd_dist = emd(xyz1, xyz2) + emd(xyz2, xyz1)
    return T.mean(emd_dist) if reduce == 'mean' else T.sum(emd_dist)


def _fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / np.sum(g)


def ssim(img1, img2, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, cs_map=False):
    """
    Returns the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    :param img1:
        a 4D :class:`torch.Tensor`.
    :param img2:
        a 4D :class:`torch.Tensor` of the same shape as `img1`.
    :param max_val:
        the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
    :param filter_size:
        size of blur kernel to use (will be reduced for small images).
    :param filter_sigma:
        standard deviation for Gaussian blur kernel (will be reduced
        for small images).
    :param k1:
        constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
    :param k2:
        constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
    :return:
        pair containing the mean SSIM and contrast sensitivity between `img1` and `img2`.
    :raise:
        RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """

    if img1.ndimension() != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img1.ndimension())

    _, _, height, width = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min((filter_size, height, width))

    # Scale down sigma if a smaller filter size is used.
    sigma = (size * filter_sigma / filter_size) if filter_size else 1.

    if filter_size:
        window = T.flip(T.tensor(_fspecial_gauss(size, sigma)), (0, 1)).view(1, 1, size, size)\
            .requires_grad_(False).to(device=img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, window)
        mu2 = F.conv2d(img2, window)
        sigma11 = F.conv2d(img1 * img1, window)
        sigma22 = F.conv2d(img2 * img2, window)
        sigma12 = F.conv2d(img1 * img2, window)
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu1 = mu1 * mu1
    mu2 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu1
    sigma22 -= mu2
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = T.mean((((2.0 * mu12 + c1) * v1) / ((mu1 + mu2 + c1) * v2 + 1e-10)))
    output = ssim if not cs_map else (ssim, T.mean(v1 / v2))
    return output


def psnr(x, y):
    """
    Peak-signal-to-noise ratio for [0,1] images.

    :param x:
        a :class:`torch.Tensor`.
    :param y:
        a :class:`torch.Tensor` of the same shape as `x`.
    """

    return -10 * T.log(T.mean((y - x) ** 2)) / np.log(10.)


def tv_reg(y):
    return T.sum(T.abs(y[..., :-1] - y[..., 1:])) + T.sum(T.abs(y[..., :-1, :] - y[..., 1:, :]))


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    """
    Applies :func:`torch.nn.utils.spectral_norm` recursively to `module` and all of
    its submodules.

    :param module:
        containing module.
    :param name:
        name of weight parameter.
        Default: ```weight```.
    :param n_power_iterations:
        number of power iterations to calculate spectral norm.
    :param eps:
        epsilon for numerical stability in calculating norms.
    :param dim:
        dimension corresponding to number of outputs,
        the default is ``0``, except for modules that are instances of
        ConvTranspose{1,2,3}d, when it is ``1``.
    :return:
        the original module with the spectral norm hook.
    """

    if hasattr(module, 'weight'):
        if dim is None:
            dim = 1 if isinstance(module, nnt.ConvTranspose2d) else 0

        if not isinstance(module, (nn.modules.batchnorm._BatchNorm,
                                   nn.GroupNorm,
                                   nn.LayerNorm)):
            module = nn.utils.spectral_norm(module, name, n_power_iterations, eps, dim)

        return module
    else:
        for mod_name, mod in module.named_children():
            mod = spectral_norm(mod, name, n_power_iterations, eps, dim)
            module.__setattr__(mod_name, mod)
        return module
