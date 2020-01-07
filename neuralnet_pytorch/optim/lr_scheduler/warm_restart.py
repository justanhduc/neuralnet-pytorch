from torch.optim.lr_scheduler import CosineAnnealingLR


class WarmRestart(CosineAnnealingLR):
    """
    Step should be used in batch iteration loop.
    Putting step in the epoch loop results in wrong behavior of the restart.
    One must not pass the iteration number to step.

    Parameters
    ----------
    optimizer
        wrapped optimizer.
    T_max
        maximum number of iterations.
    T_mul
        multiplier for `T_max`.
    eta_min
        minimum learning rate.
        Default: 0.
    last_epoch
        the index of last epoch.
        Default: -1.
    """

    def __init__(self, optimizer, T_max, T_mul=1, eta_min=0, last_epoch=-1):
        self.T_mul = T_mul
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mul
        return super().get_lr()
