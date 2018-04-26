import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import json


def adam(params, **kwargs):
    lr = kwargs.get('lr', 1e-3)
    beta1 = kwargs.get('beta1', 0.9)
    beta2 = kwargs.get('beta2', 0.999)
    eps = kwargs.get('epsilon', 1e-8)
    wd = kwargs.get('weight_decay', 0)
    return optim.Adam(params, lr, (beta1, beta2), eps, wd)


def sgd(params, **kwargs):
    lr = kwargs.get('lr', 1e-3)
    mom = kwargs.get('momentum', 0.)
    ntr = kwargs.get('nesterov', False)
    wd = kwargs.get('weight_decay', 0)
    damp = kwargs.get('dampening', 0)
    return optim.SGD(params, lr, mom, damp, wd, ntr)


def rmsprop(params, **kwargs):
    lr = kwargs.get('lr', 1e-2)
    mom = kwargs.get('momentum', 0.)
    eps = kwargs.get('epsilon', 1e-8)
    alpha = kwargs.get('alpha', 0.99)
    wd = kwargs.get('weight_decay', 0)
    centered = kwargs.get('centered', False)
    return optim.RMSprop(params, lr, alpha, eps, wd, mom, centered)


class ConfigParser(object):
    def __init__(self, config_file, **kwargs):
        super(ConfigParser, self).__init__()
        self.config_file = config_file
        self.config = self.load_configuration()

    def load_configuration(self):
        try:
            with open(self.config_file) as f:
                data = json.load(f)
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


def var_to_numpy(var):
    return var.cpu().data.numpy() if T.cuda.is_available() else var.data.numpy()


def numpy_to_var(numpy):
    var = F.Variable(T.from_numpy(numpy)).type(T.FloatTensor)
    if T.cuda.is_available():
        var = var.cuda()
    return var


function = {'relu': lambda x, **kwargs: F.relu(x, True), 'linear': lambda x, **kwargs: x,
            'lrelu': lambda x, **kwargs: lrelu(x, **kwargs), 'tanh': lambda x, **kwargs: F.tanh(x)}
init = {'He_normal': nn.init.kaiming_normal, 'He_uniform': nn.init.kaiming_uniform,
        'Xavier_normal': nn.init.xavier_normal, 'Xavier_uniform': nn.init.xavier_uniform}
loss = {'ce': F.cross_entropy, 'mse': F.mse_loss, 'bce': F.binary_cross_entropy, 'bcel': F.binary_cross_entropy_with_logits,
        'l1': F.l1_loss}
optimizer = {'adam': adam, 'sgd': sgd, 'rmsprop': rmsprop}
