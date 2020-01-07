import torch as T
from torch import optim


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
