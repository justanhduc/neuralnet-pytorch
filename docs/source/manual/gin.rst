.. _gin:
.. currentmodule:: neuralnet_pytorch


========================
:mod:`gin` -- Gin-config
========================

This page presents a list of preserved keywords for
`Gin-config <https://github.com/google/gin-config>`_.

.. contents:: Contents
   :depth: 2

Optimizer Keywords
==================
::

    'adabound' - neuralnet_pytorch.optim.AdaBound
    'adadelta' - torch.optim.Adadelta
    'adagrad' - torch.optim.Adagrad
    'adam' - torch.optim.Adam
    'adamax' - torch.optim.Adamax
    'adamw' - torch.optim.AdamW
    'asgd' - torch.optim.ASGD
    'lbfgs' - torch.optim.LBFGS
    'lookahead' - neuralnet_pytorch.optim.Lookahead
    'nadam' - neuralnet_pytorch.optim.NAdam
    'rmsprop' - torch.optim.RMSprop
    'rprop' - torch.optim.Rprop
    'sgd' - torch.optim.SGD
    'sparse_adam' - torch.optim.SparseAdam

If `Automatic Mixed Precision <https://github.com/NVIDIA/apex>`_ is installed,
there are a few extra choices coming from the package ::

    'fusedadam' - apex.optimizers.FusedAdam
    'fusedlamb' - apex.optimizers.FusedLAMB
    'fusednovograd' - apex.optimizers.FusedNovoGrad
    'fusedsgd' - apex.optimizers.FusedSGD

Learning Rate Scheduler Keywords
================================
::

    'lambda_lr' - torch.optim.lr_scheduler.LambdaLR
    'step_lr' - torch.optim.lr_scheduler.StepLR
    'multistep_lr' - torch.optim.lr_scheduler.MultiStepLR
    'exp_lr' - torch.optim.lr_scheduler.ExponentialLR
    'cosine_lr' - torch.optim.lr_scheduler.CosineAnnealingLR
    'plateau_lr' - torch.optim.lr_scheduler.ReduceLROnPlateau
    'cyclic_lr' - torch.optim.lr_scheduler.CyclicLR
    'inverse_lr' - neuralnet_pytorch.optim.lr_scheduler.InverseLR
    'warm_restart' - neuralnet_pytorch.optim.lr_scheduler.WarmRestart

Loss Keywords
=============
::

    'l1loss' - torch.nn.L1Loss
    'mseloss' - torch.nn.MSELoss
    'celoss' - torch.nn.CrossEntropyLoss
    'bceloss' - torch.nn.BCELoss
    'bcelogit_loss' - torch.nn.BCEWithLogitsLoss
    'nllloss' - torch.nn.NLLLoss
    'kldloss' - torch.nn.KLDivLoss
    'huberloss' - torch.nn.SmoothL1Loss
    'cosineembed_loss' - torch.nn.CosineEmbeddingLoss

Activation Keywords
===================
::

    'elu' - torch.nn.ELU
    'hardshrink' - torch.nn.Hardshrink
    'hardtanh' - torch.nn.Hardtanh
    'lrelu' - torch.nn.LeakyReLU
    'logsig' - torch.nn.LogSigmoid
    'multihead_att' - torch.nn.MultiheadAttention
    'prelu' - torch.nn.PReLU
    'relu' - torch.nn.ReLU
    'rrelu' - torch.nn.RReLU
    'selu' - torch.nn.SELU
    'celu' - torch.nn.CELU
    'sigmoid' - torch.nn.Sigmoid
    'softplus' - torch.nn.Softplus
    'softshrink' - torch.nn.Softshrink
    'softsign' - torch.nn.Softsign
    'tanh' - torch.nn.Tanh
    'tanhshrink' - torch.nn.Tanhshrink
    'threshold' - torch.nn.Threshold

Data Type Keywords
==================
::

    'float16' - torch.float16
    'float32' - torch.float32
    'float64' - torch.float64
    'int8' - torch.int8
    'int16' - torch.int16
    'int32' - torch.int32
    'int64' - torch.int64
    'complex32' - torch.complex32
    'complex64' - torch.complex64
    'complex128' - torch.complex128
    'float' - torch.float
    'short' - torch.short
    'long' - torch.long
    'half' - torch.half
    'uint8' - torch.uint8
    'int' - torch.int
