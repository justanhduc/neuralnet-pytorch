import torch as T

from . import tensor_utils as tutils

__all__ = ['rgb2gray', 'rgb2ycbcr', 'rgba2rgb', 'ycbcr2rgb', 'pc2vox', 'pc2vox_fast']


def rgb2gray(img: T.Tensor):
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


def rgb2ycbcr(img: T.Tensor):
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


def ycbcr2rgb(img: T.Tensor):
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


def rgba2rgb(img: T.Tensor):
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


def pc2vox(pc: T.Tensor, vox_size=32, sigma=.005, analytical_gauss_norm=True):
    """
    Converts a centered point cloud to voxel representation.

    :param pc:
        a batch of centered point clouds.
    :param vox_size:
        resolution of the voxel field.
        Default: 32.
    :param sigma:
        std of the Gaussian blur that determines the area of effect of each point.
    :param analytical_gauss_norm:
        whether to use a analytically precomputed normalization term.
    :return:
        the voxel representation of input.
    """
    assert pc.ndimension() in (2, 3), 'Point cloud must be a 2D a 3D tensor'
    if pc.ndimension() == 2:
        pc = pc[None]

    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    vox_size = vox_size

    rng = T.linspace(-1.0, 1.0, vox_size).to(pc.device)
    xg, yg, zg = T.meshgrid(rng, rng, rng)  # [G,G,G]

    x_big = tutils.shape_padright(x, 3)  # [B,N,1,1,1]
    y_big = tutils.shape_padright(y, 3)  # [B,N,1,1,1]
    z_big = tutils.shape_padright(z, 3)  # [B,N,1,1,1]

    xg = tutils.shape_padleft(xg, 2)  # [1,1,G,G,G]
    yg = tutils.shape_padleft(yg, 2)  # [1,1,G,G,G]
    zg = tutils.shape_padleft(zg, 2)  # [1,1,G,G,G]

    # squared distance
    sq_distance = (x_big - xg) ** 2. + (y_big - yg) ** 2. + (z_big - zg) ** 2.

    # compute gaussian
    func = T.exp(-sq_distance / (2. * sigma ** 2))  # [B,N,G,G,G]

    # normalise gaussian
    if analytical_gauss_norm:
        # should work with any grid sizes
        magic_factor = 1.78984352254  # see estimate_gauss_normaliser
        sigma_normalised = sigma * vox_size
        normaliser = 1. / (magic_factor * (sigma_normalised ** 3.))
        func *= normaliser
    else:
        normaliser = T.sum(func, (2, 3, 4), keepdim=True)
        func /= normaliser

    summed = T.sum(func, dim=1)  # [B,G,G G]
    voxels = T.clamp(summed, 0., 1.)
    return voxels


def pc2vox_fast(pc: T.Tensor, voxel_size=32, grid_size=1., filter_outlier=True, c_code=True):
    """
    A fast conversion from a centered point cloud to voxel representation.

    :param pc:
        a batch of centered point clouds.
    :param voxel_size:
        resolution of the voxel field.
        Default: 32.
    :param grid_size:
        range of the point clouds.
        Default: 1.
    :param filter_outlier:
        whether to filter point outside of `grid_size`.
        Default: ``True``.
    :param c_code:
        whether to use a C++ implementation.
        Default: ``True``.
    :return:
        the voxel representation of input.
    """

    assert pc.ndimension() in (2, 3), 'Point cloud must be a 2D a 3D tensor'
    if pc.ndimension() == 2:
        pc = pc[None]

    if c_code:
        from ..extensions import pc2vox
        voxel = pc2vox(pc, voxel_size, grid_size, filter_outlier)
    else:
        b, n, _ = pc.shape
        half_size = grid_size / 2.
        valid = (pc >= -half_size) & (pc <= half_size)
        valid = T.all(valid, 2)
        pc_grid = (pc + half_size) * (voxel_size - 1.)
        indices_floor = T.floor(pc_grid)
        indices = indices_floor.long()
        batch_indices = T.arange(b).to(pc.device)
        batch_indices = tutils.shape_padright(batch_indices, 2)
        batch_indices = tutils.tile(batch_indices, (1, n, 1))
        indices = T.cat((batch_indices, indices), 2)
        indices = T.reshape(indices, (-1, 4))

        r = pc_grid - indices_floor
        rr = (1. - r, r)
        if filter_outlier:
            valid = valid.flatten()
            indices = indices[valid]

        def interpolate_scatter3d(pos):
            updates = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
            updates = updates.flatten()

            if filter_outlier:
                updates = updates[valid]

            indices_shift = T.tensor([[0] + pos]).to(pc.device)
            indices_loc = indices + indices_shift
            out_shape = (b,) + (voxel_size,) * 3
            voxels = T.zeros(*out_shape).to(pc.device).flatten()
            voxels.scatter_add_(0, tutils.ravel_index(indices_loc.t(), out_shape), updates)
            return voxels.view(*out_shape)

        voxel = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
        voxel = sum(voxel)
        voxel = T.clamp(voxel, 0., 1.)
    return voxel
