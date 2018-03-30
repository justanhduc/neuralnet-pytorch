import torch as T
import torch.nn as nn
import abc
import numpy as np

from neuralnet import optimization
from neuralnet import training


class Model(optimization.Optimizer, training.Training, nn.Sequential, metaclass=abc.ABCMeta):
    def __init__(self, config_file, **kwargs):
        super(Model, self).__init__(config_file, **kwargs)
        self.params = []
        self.trainable = []
        self.regularizable = []

    def get_all_params(self):
        self.params = list(self.parameters())

    def get_trainable(self):
        for layer in self:
            self.trainable += layer.trainable

    def get_regularizable(self):
        for layer in self:
            self.regularizable += layer.regularizable

    def save_params(self):
        np.savez(self.param_file, **{p.name: p.numpy() for p in self.params})
        print('Model weights dumped to %s' % self.param_file)

    def load_params(self):
        weights = np.load(self.param_file)
        for p in self.params:
            try:
                p.data = T.from_numpy(weights[p.name])
            except:
                NameError('There is no saved weight for %s' % p.name)
        print('Model weights loaded from %s' % self.param_file)

    def reset(self):
        for layer in self:
            layer.reset()
