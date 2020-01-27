.. _layers:
.. currentmodule:: neuralnet_pytorch

=============================
:mod:`layers` -- Basic Layers
=============================

.. module:: neuralnet_pytorch.layers
   :platform: Unix, Windows
   :synopsis: Basics layers in Deep Neural Networks
.. moduleauthor:: Duc Nguyen

This section describes all the backbone modules of Neuralnet-pytorch.

.. contents:: Contents
   :depth: 4

Abstract Layers
===============

Attributes
----------
The following classes equip the plain ``torch`` modules with more bells and whistles.
Also, some features are deeply integrated into Neuralnet-pytorch,
which enables faster and more convenient training and testing of your neural networks.

.. autoclass:: neuralnet_pytorch.layers.abstract._LayerMethod
    :members:
.. autoclass:: neuralnet_pytorch.layers.Net
    :members:
.. autoclass:: neuralnet_pytorch.layers.MultiSingleInputModule
.. autoclass:: neuralnet_pytorch.layers.MultiMultiInputModule


Extended Pytorch Abstract Layers
--------------------------------

.. autoclass:: neuralnet_pytorch.layers.Module
.. autoclass:: neuralnet_pytorch.layers.Sequential

Quick-and-dirty Layers
----------------------

.. autodecorator:: neuralnet_pytorch.layers.wrapper
.. autoclass:: neuralnet_pytorch.layers.Lambda

Common Layers
=============

Extended Pytorch Common Layers
------------------------------

.. autoclass:: neuralnet_pytorch.layers.Conv2d
.. autoclass:: neuralnet_pytorch.layers.ConvTranspose2d
.. autoclass:: neuralnet_pytorch.layers.FC
.. autoclass:: neuralnet_pytorch.layers.Softmax

Extra Layers
------------

.. autoclass:: neuralnet_pytorch.layers.Activation
.. autoclass:: neuralnet_pytorch.layers.ConvNormAct
.. autoclass:: neuralnet_pytorch.layers.DepthwiseSepConv2D
.. autoclass:: neuralnet_pytorch.layers.FCNormAct
.. autoclass:: neuralnet_pytorch.layers.ResNetBasicBlock
.. autoclass:: neuralnet_pytorch.layers.ResNetBottleneckBlock
.. autoclass:: neuralnet_pytorch.layers.StackingConv

Graph Learning Layers
---------------------

.. autoclass:: neuralnet_pytorch.layers.GraphConv
.. autoclass:: neuralnet_pytorch.layers.BatchGraphConv
.. autoclass:: neuralnet_pytorch.layers.GraphXConv

Multi-module Layers
-------------------

.. autoclass:: neuralnet_pytorch.layers.Sum
.. autoclass:: neuralnet_pytorch.layers.ConcurrentSum
.. autoclass:: neuralnet_pytorch.layers.SequentialSum
