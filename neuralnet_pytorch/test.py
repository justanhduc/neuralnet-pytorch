import torch as T
import numpy as np
from torch import testing
from torch.nn import functional as F
from torchvision.models import resnet

import neuralnet_pytorch as nnt
from neuralnet_pytorch import cuda_available


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

    if cuda_available:
        input = input.cuda()
        module1 = module1.cuda()
        module2 = module2.cuda()

    expected = module2(input)
    testing.assert_allclose(module1(input), expected)
    testing.assert_allclose(module1.output_shape, tuple(expected.shape))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return T.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def test_softmax():
    shape = (64, 512)
    out_features = 1000
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    sm = nnt.Softmax(shape, out_features)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias)
    testing.assert_allclose(sm(a), expected)
    testing.assert_allclose(sm.output_shape, expected.shape)

    sm = nnt.Softmax(shape, out_features, dim=0)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias, dim=0)
    testing.assert_allclose(sm(a), expected)
    testing.assert_allclose(sm.output_shape, expected.shape)


def test_global_avgpool2d():
    shape = (4, 3, 256, 512)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    pool = nnt.GlobalAvgPool2D(shape)
    expected = T.mean(a, (2, 3))
    testing.assert_allclose(pool(a), expected)
    testing.assert_allclose(pool.output_shape, expected.shape)

    pool = nnt.GlobalAvgPool2D(shape, keepdim=True)
    expected = T.mean(a, (2, 3)).unsqueeze(-1).unsqueeze(-1)
    testing.assert_allclose(pool(a), expected)
    testing.assert_allclose(pool.output_shape, expected.shape)


def test_shape_pad():
    shape = (10, 10)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    padded = nnt.utils.shape_padleft(a)
    expected = a.unsqueeze(0)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = nnt.utils.shape_padleft(a, 2)
    expected = a.unsqueeze(0).unsqueeze(0)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = nnt.utils.shape_padleft(a, 5)
    expected = a.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = nnt.utils.shape_padright(a)
    expected = a.unsqueeze(-1)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = nnt.utils.shape_padright(a, 2)
    expected = a.unsqueeze(-1).unsqueeze(-1)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)

    padded = nnt.utils.shape_padright(a, 5)
    expected = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    testing.assert_allclose(padded, expected)
    testing.assert_allclose(padded.shape, expected.shape)


def test_dimshuffle_layer():
    shape = (64, 512)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    dimshuffle = nnt.DimShuffle(shape, (1, 0))
    expected = a.transpose(1, 0)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle(shape, (0, 1, 'x', 'x'))
    expected = a.unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle(shape, (1, 0, 'x', 'x'))
    expected = a.permute(1, 0).unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle(shape, (0, 'x', 1, 'x'))
    expected = a.unsqueeze(2).permute(0, 2, 1).unsqueeze(3)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle(shape, (1, 'x', 'x', 0))
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(1)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle(shape, (1, 'x', 0, 'x', 'x'))
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(3).unsqueeze(3)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)


def test_dimshuffle():
    shape = (64, 512)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    dimshuffled = nnt.utils.dimshuffle(a, (0, 1, 'x', 'x'))
    expected = a.unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = nnt.utils.dimshuffle(a, (1, 0, 'x', 'x'))
    expected = a.permute(1, 0).unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = nnt.utils.dimshuffle(a, (0, 'x', 1, 'x'))
    expected = a.unsqueeze(2).permute(0, 2, 1).unsqueeze(3)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = nnt.utils.dimshuffle(a, (1, 'x', 'x', 0))
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(1)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)

    dimshuffled = nnt.utils.dimshuffle(a, (1, 'x', 0, 'x', 'x'))
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(3).unsqueeze(3)
    testing.assert_allclose(dimshuffled, expected)
    testing.assert_allclose(dimshuffled.shape, expected.shape)


def test_flatten():
    shape = (10, 4, 2, 3, 6)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    flatten = nnt.Flatten((None,) + shape[1:], 2)
    expected = T.flatten(a, 2)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose((shape[0],) + flatten.output_shape[1:], expected.shape)

    flatten = nnt.Flatten((None,) + shape[1:], 4)
    expected = T.flatten(a, 4)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose((shape[0],) + flatten.output_shape[1:], expected.shape)

    flatten = nnt.Flatten(shape)
    expected = T.flatten(a)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose(flatten.output_shape, expected.shape)

    flatten = nnt.Flatten(shape, 1, 3)
    expected = T.flatten(a, 1, 3)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose(flatten.output_shape, expected.shape)


def test_reshape():
    shape = (10, 3, 9, 9)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    newshape = (-1, 9, 9)
    reshape = nnt.Reshape((None,) + shape[1:], newshape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)
    testing.assert_allclose(reshape.output_shape[1:], expected.shape[1:])

    newshape = (10, -1, 9)
    reshape = nnt.Reshape(shape, newshape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)
    testing.assert_allclose(reshape.output_shape, expected.shape)

    newshape = (9, 9, -1)
    reshape = nnt.Reshape(shape, newshape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)
    testing.assert_allclose(reshape.output_shape[1:], expected.shape[1:])


def test_lambda():
    shape = (3, 10, 5, 5)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    def foo1(x, y):
        return x ** y

    sqr = nnt.Lambda(shape, foo1, y=2.)
    expected = a ** 2.
    testing.assert_allclose(sqr(a), expected)
    testing.assert_allclose(sqr.output_shape, expected.shape)

    def foo2(x, fr, to):
        return x[:, fr:to]

    fr = 3
    to = 7
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    slice = nnt.Lambda(shape, foo2, fr=fr, to=to)
    expected = a[:, fr:to]
    testing.assert_allclose(slice(a), expected)
    testing.assert_allclose(slice.output_shape, expected.shape)


def test_cat():
    shape1 = (3, 2, 4, 4)
    shape2 = (3, 5, 4, 4)
    a = T.rand(*shape1)
    b = T.rand(*shape2)
    if cuda_available:
        a = a.cuda()
        b = b.cuda()

    cat = nnt.Cat((shape1, shape2), 1)
    expected = T.cat((a, b), 1)
    testing.assert_allclose(cat(a, b), expected)
    testing.assert_allclose(cat.output_shape, expected.shape)

    shape1 = (3, 2, 4, 4)
    shape2 = (3, 2, 9, 4)
    a = T.rand(*shape1)
    b = T.rand(*shape2)
    if cuda_available:
        a = a.cuda()
        b = b.cuda()

    cat = nnt.Cat((shape1, shape2), 2)
    expected = T.cat((a, b), 2)
    testing.assert_allclose(cat(a, b), expected)
    testing.assert_allclose(cat.output_shape, expected.shape)


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

    shape = (None, 3, None, 224)
    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, 112)

    shape = (None, 3, None, None)
    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, None)


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

    shape = (None, 3, None, None)
    bn_nnt = nnt.BatchNorm2d(shape)
    assert bn_nnt.output_shape == (None, 3, None, None)
