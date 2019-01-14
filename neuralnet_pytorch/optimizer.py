from torch import optim


def adam(params, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., **kwargs):
    closure = kwargs.pop('closure', None)
    return optim.Adam(params, lr, (beta1, beta2), eps, weight_decay).step(closure)


def amsgrad(params, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., **kwargs):
    closure = kwargs.pop('closure', None)
    return optim.Adam(params, lr, (beta1, beta2), eps, weight_decay, amsgrad=True).step(closure)


def sgd(params, lr, momentum=0., dampening=0., weight_decay=0., nesterov=False, **kwargs):
    closure = kwargs.pop('closure', None)
    return optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov).step(closure)


def rmsprop(params, lr, alpha, momentum, eps, weght_decay, centered, **kwargs):
    closure = kwargs.pop('closure', None)
    return optim.RMSprop(params, lr, alpha, eps, weght_decay, momentum, centered).step(closure)
