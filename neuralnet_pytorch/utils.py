import os
import torch as T
import torch.nn.functional as F
import json
import yaml
import numpy as np


def validate(func):
    """make sure output shape is a list of ints"""

    def func_wrapper(self):
        shape = [None if x is None else None if np.isnan(x) else x for x in func(self)]
        out = [int(x) if x is not None else x for x in shape]
        return tuple(out)

    return func_wrapper


class ConfigParser(object):
    def __init__(self, config_file, type='json', **kwargs):
        super(ConfigParser, self).__init__()
        self.config_file = config_file
        self.config = self.load_configuration()
        self.type = type

    def load_configuration(self):
        try:
            with open(self.config_file) as f:
                data = json.load(f) if type == 'json' else yaml.load(f)
            print('Config file loaded successfully')
        except:
            raise NameError('Unable to open config file!!!')
        return data


def lrelu(x, **kwargs):
    alpha = kwargs.get('alpha', 0.1)
    return F.leaky_relu(x, alpha, True)


def rgb2gray(img):
    if img.ndimension() != 4:
        raise ValueError('Input images must have four dimensions, not %d' % img.ndimension())
    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)


def rgb2ycbcr(img):
    if img.ndimension() != 4:
        raise ValueError('Input images must have four dimensions, not %d' % img.ndimension())
    Y = 0. + .299 * img[:, 0] + .587 * img[:, 1] + .114 * img[:, 2]
    Cb = 128. - .169 * img[:, 0] - .331 * img[:, 1] + .5 * img[:, 2]
    Cr = 128. + .5 * img[:, 0] - .419 * img[:, 1] - .081 * img[:, 2]
    return T.cat((Y.unsqueeze(1), Cb.unsqueeze(1), Cr.unsqueeze(1)), 1)


def ycbcr2rgb(img):
    if img.ndimension() != 4:
        raise ValueError('Input images must have four dimensions, not %d' % img.ndimension())
    R = img[:, 0] + 1.4 * (img[:, 2] - 128.)
    G = img[:, 0] - .343 * (img[:, 1] - 128.) - .711 * (img[:, 2] - 128.)
    B = img[:, 0] + 1.765 * (img[:, 1] - 128.)
    return T.cat((R.unsqueeze(1), G.unsqueeze(1), B.unsqueeze(1)), 1)


def batch_get_value(params):
    return tuple([to_numpy(p) for p in params])


def batch_set_value(params, values):
    for p, v in zip(params, values):
        p.data.copy_(T.from_numpy(v))


def to_numpy(x):
    return x.cpu().detach().data.numpy()


def to_cuda(x):
    return T.from_numpy(x).cuda()


def bulk_to_numpy(xs):
    return tuple([to_numpy(x) for x in xs])


def bulk_to_cuda(xs):
    return tuple([to_cuda(x) for x in xs])


function = {'relu': lambda x, **kwargs: F.relu(x, True), 'linear': lambda x, **kwargs: x, None: lambda x, **kwargs: x,
            'lrelu': lambda x, **kwargs: lrelu(x, **kwargs), 'tanh': lambda x, **kwargs: F.tanh(x)}
