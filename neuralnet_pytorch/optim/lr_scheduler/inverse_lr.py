import torch.optim as optim


class InverseLR(optim.lr_scheduler.LambdaLR):
    """Decreases lr every iteration by the inverse of gamma times iteration plus 1.
    :math:`\\text{lr} = \\text{lr} / (1 + \\gamma * t)`.

    Parameters
    ----------
    optimizer
        wrapped optimizer.
    gamma
        decrease coefficient.
    last_epoch : int
        the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, lambda it: 1. / (1. + gamma * it), last_epoch)
