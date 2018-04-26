import torch.nn.functional as F
import numpy as np

from neuralnet_pytorch import utils


class Optimizer(utils.ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Optimizer, self).__init__(config_file, **kwargs)

        self.cost_function = self.config['optimization']['cost_function']
        self.class_weights = self.config['optimization']['class_weights']
        self.method = self.config['optimization']['method']
        self.learning_rate = self.config['optimization']['learning_rate']
        self.momentum = self.config['optimization']['momentum']
        self.epsilon = self.config['optimization']['epsilon']
        self.gamma = self.config['optimization']['gamma']
        self.rho = self.config['optimization']['rho']
        self.beta1 = self.config['optimization']['beta1']
        self.beta2 = self.config['optimization']['beta2']
        self.nesterov = self.config['optimization']['nesterov']
        self.regularization = self.config['optimization']['regularization']
        self.regularization_type = self.config['optimization']['regularization_type']
        self.regularization_coeff = self.config['optimization']['regularization_coeff']
        self.decrease_factor = np.float32(self.config['optimization']['decrease_factor'])
        self.final_learning_rate = self.config['optimization']['final_learning_rate']
        self.last_iter_to_decrease = self.config['optimization']['last_iter_to_decrease']
        self.optimizer = None

    def compute_cost(self, pred, target, **kwargs):
        return utils.loss[self.method](pred, target, **kwargs)

    def get_optimizer(self, params):
        print('Using %s optimizer' % self.method)
        self.optimizer = utils.optimizer[self.method](params, **{'lr': self.learning_rate, 'momentum': self.momentum,
                                                                 'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                                                                 'nesterov': self.nesterov})

    def learn(self, loss=None, pred=None, target=None, **kwargs):
        cost = loss if loss is not None else self.compute_cost(pred, target, **kwargs)
        cost.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
