import torch as T
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = ['NAdam', 'WarmRestart', 'InverseLR']


class NAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 decay=lambda x, t: x * (1. - .5 * .96 ** (t / 250.))):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.decay = decay

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = T.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = T.zeros_like(p.data)
                    # Beta1 accumulation
                    state['beta1_cum'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                beta1_t = self.decay(beta1, state['step'])
                beta1_tp1 = self.decay(beta1, state['step'] + 1.)
                beta1_cum = state['beta1_cum'] * beta1_t

                g_hat_t = grad / (1. - beta1_cum)
                exp_avg.mul_(beta1).add_(1. - beta1, grad)
                m_hat_t = exp_avg / (1. - beta1_cum * beta1_tp1)

                exp_avg_sq.mul_(beta2).addcmul_(1. - beta2, grad, grad)
                v_hat_t = exp_avg_sq / (1. - beta2 ** state['step'])
                m_bar_t = (1. - beta1) * g_hat_t + beta1_tp1 * m_hat_t

                denom = v_hat_t.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], m_bar_t, denom)
                state['beta1_cum'] = beta1_cum

        return loss


class WarmRestart(CosineAnnealingLR):
    """
    step should be used in the inner loop, i.e., the iteration loop. Putting step in the epoch loop results in wrong
    behavior of the restart.
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


class InverseLR(optim.lr_scheduler.LambdaLR):
    """Decreases lr every iteration by the inverse of gamma times iteration plus 1

    lr = lr / (1 + gamma * t)
    """
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, lambda it: 1. / (1. + gamma * it), last_epoch)
