from torch.optim.lr_scheduler import CosineAnnealingLR


class WarmRestart(CosineAnnealingLR):
    """
    Step should be used in the inner loop, i.e., the iteration loop.
    Putting step in the epoch loop results in wrong behavior of the restart.
    One must not pass the iteration number to step.
    """

    def __init__(self, optimizer, T_max, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return super().get_lr()
