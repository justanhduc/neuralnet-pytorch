import torch as T
import torch.optim as optim
import torch.nn as nn
from gin import config

import neuralnet_pytorch as nnt
from neuralnet_pytorch import zoo

# optimizers
config.external_configurable(optim.SGD, 'sgd', module='T.optim')
config.external_configurable(optim.Adam, 'adam', module='T.optim')
config.external_configurable(optim.Adadelta, 'adadelta', module='T.optim')
config.external_configurable(optim.Adagrad, 'adagrad', module='T.optim')
config.external_configurable(optim.SparseAdam, 'sparse_adam', module='T.optim')
config.external_configurable(optim.Adamax, 'adamax', module='T.optim')
config.external_configurable(optim.AdamW, 'adamw', module='T.optim')
config.external_configurable(optim.ASGD, 'asgd', module='T.optim')
config.external_configurable(optim.LBFGS, 'lbfgs', module='T.optim')
config.external_configurable(optim.RMSprop, 'rmsprop', module='T.optim')
config.external_configurable(optim.Rprop, 'rprop', module='T.optim')
config.external_configurable(nnt.optim.AdaBound, 'adabound', module='nnt')
config.external_configurable(nnt.optim.Lookahead, 'lookahead', module='nnt')
config.external_configurable(nnt.optim.NAdam, 'nadam', module='nnt')

try:
    import apex
    config.external_configurable(apex.optimizers.FusedAdam, 'fusedadam', module='apex.optimizers')
    config.external_configurable(apex.optimizers.FusedSGD, 'fusedsgd', module='apex.optimizers')
    config.external_configurable(apex.optimizers.FusedNovoGrad, 'fusednovograd', module='apex.optimizers')
    config.external_configurable(apex.optimizers.FusedLAMB, 'fusedlamb', module='apex.optimizers')
    print('Apex Fused Optimizers is availble for GIN')
except ImportError:
    pass

# lr scheduler
config.external_configurable(optim.lr_scheduler.LambdaLR, 'lambda_lr', module='T.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.StepLR, 'step_lr', module='T.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.MultiStepLR, 'multistep_lr', module='T.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.ExponentialLR, 'exp_lr', module='T.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.CosineAnnealingLR, 'cosine_lr', module='T.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.ReduceLROnPlateau, 'plateau_lr', module='T.optim.lr_scheduler')
config.external_configurable(optim.lr_scheduler.CyclicLR, 'cyclic_lr', module='T.optim.lr_scheduler')
config.external_configurable(nnt.optim.lr_scheduler.InverseLR, 'inverse_lr')
config.external_configurable(nnt.optim.lr_scheduler.WarmRestart, 'warm_restart')

# losses
config.external_configurable(nn.L1Loss, 'l1loss', module='T.nn')
config.external_configurable(nn.MSELoss, 'mseloss', module='T.nn')
config.external_configurable(nn.CrossEntropyLoss, 'celoss', module='T.nn')
config.external_configurable(nn.BCELoss, 'bceloss', module='T.nn')
config.external_configurable(nn.BCEWithLogitsLoss, 'bcelogit_loss', module='T.nn')
config.external_configurable(nn.NLLLoss, 'nllloss', module='T.nn')
config.external_configurable(nn.KLDivLoss, 'kldloss', module='T.nn')
config.external_configurable(nn.SmoothL1Loss, 'huberloss', module='T.nn')
config.external_configurable(nn.CosineEmbeddingLoss, 'cosineembed_loss', module='T.nn')

# activations
config.external_configurable(nn.ELU, 'elu', module='T.nn')
config.external_configurable(nn.Hardshrink, 'hardshrink', module='T.nn')
config.external_configurable(nn.Hardtanh, 'hardtanh', module='T.nn')
config.external_configurable(nn.LeakyReLU, 'lrelu', module='T.nn')
config.external_configurable(nn.LogSigmoid, 'logsig', module='T.nn')
config.external_configurable(nn.MultiheadAttention, 'multihead_att', module='T.nn')
config.external_configurable(nn.PReLU, 'prelu', module='T.nn')
config.external_configurable(nn.ReLU, 'relu', module='T.nn')
config.external_configurable(nn.RReLU, 'rrelu', module='T.nn')
config.external_configurable(nn.SELU, 'selu', module='T.nn')
config.external_configurable(nn.CELU, 'celu', module='T.nn')
config.external_configurable(nn.Sigmoid, 'sigmoid', module='T.nn')
config.external_configurable(nn.Softplus, 'softplus', module='T.nn')
config.external_configurable(nn.Softshrink, 'softshrink', module='T.nn')
config.external_configurable(nn.Softsign, 'softsign', module='T.nn')
config.external_configurable(nn.Tanh, 'tanh', module='T.nn')
config.external_configurable(nn.Tanhshrink, 'tanhshrink', module='T.nn')
config.external_configurable(nn.Threshold, 'threshold', module='T.nn')

# constants
config.constant('float16', T.float16)
config.constant('float32', T.float32)
config.constant('float64', T.float64)
config.constant('int8', T.int8)
config.constant('int16', T.int16)
config.constant('int32', T.int32)
config.constant('int64', T.int64)
config.constant('complex32', T.complex32)
config.constant('complex64', T.complex64)
config.constant('complex128', T.complex128)
config.constant('float', T.float)
config.constant('short', T.short)
config.constant('long', T.long)
config.constant('half', T.half)
config.constant('uint8', T.uint8)
config.constant('int', T.int)

# model zoo
config.external_configurable(zoo.ResNet, 'ResNet', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnet18, 'resnet18', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnet34, 'resnet34', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnet50, 'resnet50', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnet101, 'resnet101', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnet152, 'resnet152', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnext50_32x4d, 'resnext50_32x4d', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.resnext101_32x8d, 'resnext101_32x8d', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.wide_resnet50_2, 'wide_resnet50_2', module='neuralnet_pytorch.zoo.resnet')
config.external_configurable(zoo.wide_resnet101_2, 'wide_resnet101_2', module='neuralnet_pytorch.zoo.resnet')
