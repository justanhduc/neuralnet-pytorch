import torch as T
import numpy as np
from torch import testing
from torchvision.models import resnet

import neuralnet_pytorch as nnt


def assert_list_close(x, y):
    for x_, y_ in zip(x, y):
        testing.assert_allclose(x_, y)


def sanity_check(module1, module2, shape=(64, 3, 32, 32)):
    input = T.from_numpy(np.random.rand(*shape).astype('float32'))

    try:
        module1.load_state_dict(module2.state_dict())
    except RuntimeError:
        params = nnt.utils.batch_get_value(module2.state_dict().values())
        nnt.utils.batch_set_value(module1.state_dict().values(), params)

    if nnt.cuda_available:
        input = input.cuda()
        module1 = module1.cuda()
        module2 = module2.cuda()

    expected = module2(input)
    testing.assert_allclose(module1(input), expected)
    testing.assert_allclose(module1.output_shape, tuple(expected.shape))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return T.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def test_resnet_bottleneck_block():
    shape = (64, 64, 32, 32)
    n_filters = 64

    blk_nnt = nnt.ResNetBottleneckBlock(shape, n_filters)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, downsample=T.nn.Sequential(
        conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 1),
        T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)))
    sanity_check(blk_nnt, blk_pt, shape)

    blk_nnt = nnt.ResNetBottleneckBlock(shape, n_filters, stride=2)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, stride=2, downsample=T.nn.Sequential(
        conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 2),
        T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)))
    sanity_check(blk_nnt, blk_pt, shape)


def test_resnet_basic_block():
    shape = (64, 64, 32, 32)
    n_filters = 64

    blk_nnt = nnt.ResNetBasicBlock(shape, n_filters)
    blk_pt = resnet.BasicBlock(shape[1], n_filters)
    sanity_check(blk_nnt, blk_pt, shape)

    blk_nnt = nnt.ResNetBasicBlock(shape, n_filters, stride=2)
    blk_pt = resnet.BasicBlock(shape[1], n_filters, stride=2, downsample=T.nn.Sequential(
        conv1x1(shape[1], n_filters * blk_pt.expansion, 2),
        T.nn.BatchNorm2d(n_filters * blk_pt.expansion)))
    sanity_check(blk_nnt, blk_pt, shape)


def test_max_avg_pooling_layer():
    shape = (64, 3, 32, 32)
    filter_size = 3
    stride = 1
    padding = filter_size >> 1
    dilation = 2
    ceil_mode = True

    pool_nnt = nnt.MaxPool2d(shape, filter_size, stride=None, padding=0, dilation=1, ceil_mode=False)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(shape, filter_size, stride=stride, padding=0, dilation=1, ceil_mode=False)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=stride, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(shape, filter_size, stride=None, padding=padding, dilation=1, ceil_mode=False)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=padding, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(shape, filter_size, stride=None, padding=0, dilation=dilation, ceil_mode=False)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=dilation, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(shape, filter_size, stride=None, padding=0, dilation=1, ceil_mode=ceil_mode)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=ceil_mode)
    sanity_check(pool_nnt, pool_pt, shape)


def test_avg_pooling_layer():
    shape = (64, 3, 32, 32)
    filter_size = 3
    stride = 1
    padding = filter_size >> 1
    ceil_mode = True
    incl_pad = False

    pool_nnt = nnt.AvgPool2d(shape, filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(shape, filter_size, stride=stride, padding=0, ceil_mode=False, count_include_pad=True)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=stride, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(shape, filter_size, stride=None, padding=padding, ceil_mode=False, count_include_pad=True)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=padding, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(shape, filter_size, stride=None, padding=0, ceil_mode=ceil_mode, count_include_pad=True)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=ceil_mode, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(shape, filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=incl_pad)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=incl_pad)
    sanity_check(pool_nnt, pool_pt, shape)


def test_conv2d_layer():
    shape = (64, 3, 32, 32)
    n_filters = 64
    filter_size = 3

    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size)
    conv_pt = T.nn.Conv2d(shape[1], n_filters, filter_size, padding=filter_size >> 1)
    sanity_check(conv_nnt, conv_pt, shape)


def test_fc_layer():
    shape = (64, 128)
    n_nodes = 256

    fc_nnt = nnt.FC(shape, n_nodes)
    fc_pt = T.nn.Linear(shape[1], n_nodes)
    sanity_check(fc_nnt, fc_pt, shape)


def test_batchnorm2d_layer():
    shape = (64, 3, 32, 32)

    bn_nnt = nnt.BatchNorm2d(shape)
    bn_pt = T.nn.BatchNorm2d(shape[1])
    sanity_check(bn_nnt, bn_pt, shape)
