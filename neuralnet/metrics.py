import torch as T
import torch.nn.functional as F
import numpy as np


def huber_loss(x, y, thres=1., sum=False):
    if y.ndimension() != x.ndimension():
        raise TypeError('y should have the same shape as y_pred', ('y', y.data.type(), 'y_pred', x.data.type()))
    e = T.abs(x - y)
    larger_than_equal_to = .5 * thres ** 2 + thres * (e - thres)
    less_than = .5 * e**2
    mask = e >= thres
    mask = mask.type(e.data.type())
    return T.mean(mask * larger_than_equal_to + (1. - mask) * less_than) if not sum \
        else T.sum(mask * larger_than_equal_to + (1. - mask) * less_than)


def _flip_tensor(filters):
    flipped = filters.numpy().copy()

    for i in range(len(filters.size())):
        flipped = np.flip(flipped, i) #Reverse given tensor on dimention i
    return T.from_numpy(flipped.copy())


def first_derivative_error(x, y, p=2, sum=False):
    if x.ndimension() != 4 and y.ndimension() != 4:
        raise TypeError('y and y_pred should have four dimensions')
    kern_x = T.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32'))
    kern_x = F.Variable(_flip_tensor(kern_x.expand(y.shape[1], y.shape[1], 3, 3)))

    kern_y = T.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32'))
    kern_y = F.Variable(_flip_tensor(kern_y.expand(y.shape[1], y.shape[1], 3, 3)))

    if T.cuda.is_available():
        kern_x = kern_x.cuda()
        kern_y = kern_y.cuda()

    x_grad_x = F.conv2d(x, kern_x, padding=1)
    x_grad_y = F.conv2d(x, kern_y, padding=1)
    x_grad = T.sqrt(x_grad_x ** 2 + x_grad_y ** 2 + 1e-10)

    y_grad_x = F.conv2d(y, kern_x, padding=1)
    y_grad_y = F.conv2d(y, kern_y, padding=1)
    y_grad = T.sqrt(y_grad_x ** 2 + y_grad_y ** 2 + 1e-10)
    return norm_error(x_grad, y_grad, p, sum=sum)


def norm_error(x, y, p=2, sum=False):
    if y.ndimension() != x.ndimension():
        raise TypeError('y should have the same shape as y_pred', ('y', y.data.type(), 'y_pred', x.data.type()))
    return T.mean(T.abs(x - y) ** p) if not sum else T.sum(T.abs(x - y) ** p)


def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / np.sum(g)


def ssim(img1, img2, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, cs_map=False):
    """Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
    Raises:
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
        window = F.Variable(_flip_tensor(T.from_numpy(_fspecial_gauss(size, sigma))).view(1, 1, size, size)).type(T.FloatTensor)
        if T.cuda.is_available():
            window = window.cuda()
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


def _laplacian_pyramid(img, levels=3, size=5, sigma=2.):
    pyr = []
    current = img
    gauss_kern = F.Variable(T.from_numpy(_fspecial_gauss(size, sigma)).expand(current.shape[1], 1, size, size)).type(T.FloatTensor)
    if T.cuda.is_available():
        gauss_kern = gauss_kern.cuda()
    for l in range(levels):
        gauss = F.conv2d(current, gauss_kern, stride=1, padding=size//2, groups=current.shape[1])
        diff = current - gauss
        pyr.append(diff)
        current = F.avg_pool2d(gauss, 2)
    pyr.append(current)
    return pyr


def _gaussian_pyramid(img, levels=3, size=5, sigma=2.):
    pyr = []
    current = img
    gauss_kern = F.Variable(T.from_numpy(_fspecial_gauss(size, sigma)).expand(current.shape[1], 1, size, size)).type(T.FloatTensor)
    if T.cuda.is_available():
        gauss_kern = gauss_kern.cuda()
    for l in range(levels):
        gauss = F.conv2d(current, gauss_kern, stride=1, padding=size // 2, groups=current.shape[1])
        pyr.append(gauss)
        current = F.avg_pool2d(gauss, 2)
    pyr.append(current)
    return pyr


def pyramid_loss(img1, img2, type='laplacian', levels=3, size=5, sigma=2., p=1, loss_sum=False, weights=None):
    if weights is None:
        weights = [1 / (levels + 1)]
    pyramid = _laplacian_pyramid if type == 'laplacian' else _gaussian_pyramid
    py1 = pyramid(img1, levels, size, sigma)
    py2 = pyramid(img2, levels, size, sigma)
    if len(weights) == 1:
        weights = weights * (levels + 1)
    elif 1 < len(weights) < levels + 1:
        raise NotImplementedError
    losses = [norm_error(a, b, p, loss_sum) * w for a, b, w in zip(py1, py2, weights)]
    return sum(losses)


def psnr(x, y):
    """PSNR for [0,1] images"""
    return -10 * T.log(norm_error(x, y)) / np.log(10.)
