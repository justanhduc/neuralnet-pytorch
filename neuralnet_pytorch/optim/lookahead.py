from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    """
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610.

    Parameters
    ----------
    optimizer
        an usual optimizer such as Adam.
    la_steps
        number of lookahead steps.
        Default: 5.
    alpha
        linear interpolation coefficient.
        Default: 0.8.
    pullback_momentum
        either ``'reset'``, ``'pullback'``, or ``None``.
        Default: ``None``.
    """

    def __init__(self, optimizer, la_steps=5, alpha=0.8, pullback_momentum=None):
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.alpha = alpha
        self._total_la_steps = la_steps
        assert pullback_momentum in ['reset', 'pullback', None]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == 'pullback':
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def la_step(self):
        return self._la_step

    @la_step.setter
    def la_step(self, value):
        self._la_step = value

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.la_step += 1

        if self.la_step >= self._total_la_steps:
            self.la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == 'pullback':
                        internal_momentum = self.optimizer.state[p]['momentum_buffer']
                        self.optimizer.state[p]['momentum_buffer'] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state['cached_mom'])
                        param_state['cached_mom'] = self.optimizer.state[p]['momentum_buffer']
                    elif self.pullback_momentum == 'reset':
                        self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

        return loss
