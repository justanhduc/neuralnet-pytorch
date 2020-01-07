import numpy as np

__all__ = ['is_outlier', 'smooth']


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
        the type of window from ``'flat'``, ``'hanning'``, ``'hamming'``,
        ``'bartlett'``, and ``'blackman'``.
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


def is_outlier(x: np.ndarray, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise. Adapted from https://stackoverflow.com/a/11886564/4591601.

    :param x:
        an ``nxd`` array of observations
    :param thresh:
        the modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

    :return:
        a ``nx1`` boolean array.

    References
    ----------
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", The ASQC Basic References in Quality Control:
    Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """

    if len(x) == 1:
        return np.array([False])

    if len(x.shape) == 1:
        x = x[:, None]

    median = np.median(x, axis=0)
    diff = np.sum((x - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh
