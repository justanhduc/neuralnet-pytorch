import torch as T

__all__ = ['rgb2gray', 'rgb2ycbcr', 'rgba2rgb', 'ycbcr2rgb']


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
