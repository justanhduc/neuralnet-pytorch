import torch as T
import math
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = ['NAdam', 'AdaBound', 'WarmRestart', 'InverseLR']


class NAdam(optim.Adam):
    """
    Adaptive moment with Nesterov gradients.

    http://cs229.stanford.edu/proj2015/054_report.pdf

    Parameters
    ----------
    params
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr
        learning rate (default: 1e-3)
    betas
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay
        weight decay (L2 penalty) (default: 0)
    decay
        a decay scheme for `betas[0]`.
        Default: :math:`\\beta * (1 - 0.5 * 0.96^{\\frac{t}{250}})`
        where `t` is the training step.
    """

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


class AdaBound(optim.Optimizer):
    """
    Implements AdaBound algorithm proposed in
    `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.

    .. _Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX

    Parameters
    ----------
    params
        iterable of parameters to optimize or dicts defining.
        parameter groups
    lr
        Adam learning rate. Default: 1e-3.
    betas
        coefficients used for computing running averages of gradient
        and its square. Default: (0.9, 0.999).
    final_lr
        final (SGD) learning rate. Default: 0.1.
    gamma
        convergence speed of the bound functions. Default: 1e-3.
    eps
        term added to the denominator to improve
        numerical stability. Default: 1e-8.
    weight_decay
        weight decay (L2 penalty). Default: 0.
    amsbound : bool
        whether to use the AMSBound variant of this algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = T.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = T.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = T.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    T.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = T.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


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
