import torch as T
import numpy as np
from torch import testing
from torch.nn import functional as F
from torchvision.models import resnet
import pytest

import neuralnet_pytorch as nnt
from neuralnet_pytorch import cuda_available

dev = ('cpu', 'cuda') if cuda_available else ('cpu',)


def sanity_check(module1, module2, shape=None, *args, **kwargs):
    device = kwargs.pop('device', 'cpu')
    module1 = module1.to(device)
    module2 = module2.to(device)

    try:
        module1.load_state_dict(module2.state_dict())
    except RuntimeError:
        params = nnt.utils.bulk_to_numpy(module2.state_dict().values())
        nnt.utils.batch_set_value(module1.state_dict().values(), params)

    if shape is not None:
        input = T.from_numpy(np.random.rand(*shape).astype('float32'))
        input = input.to(device)

        expected = module2(input)
        testing.assert_allclose(module1(input), expected)
        testing.assert_allclose(module1.output_shape, tuple(expected.shape))
    else:
        expected = module2(args)
        testing.assert_allclose(module1(*args), expected)
        testing.assert_allclose(module1.output_shape, tuple(expected.shape))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return T.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@pytest.mark.parametrize('device', dev)
def test_resnet_basic_block(device):
    shape = (64, 64, 32, 32)
    n_filters = 64

    # test constructors
    blk_nnt = nnt.ResNetBasicBlock(shape[1], n_filters)
    assert blk_nnt.output_shape == (None, n_filters, None, None)

    blk_nnt = nnt.ResNetBasicBlock((None, shape[1], None, None), n_filters)
    assert blk_nnt.output_shape == (None, n_filters, None, None)

    blk_nnt = nnt.ResNetBasicBlock(shape, n_filters)
    blk_pt = resnet.BasicBlock(shape[1], n_filters)
    sanity_check(blk_nnt, blk_pt, shape, device=device)

    blk_nnt = nnt.ResNetBasicBlock(shape, n_filters * 2, stride=2)
    blk_pt = resnet.BasicBlock(shape[1], n_filters * 2, stride=2,
                               downsample=T.nn.Sequential(
                                   conv1x1(shape[1], n_filters * blk_pt.expansion * 2, 2),
                                   T.nn.BatchNorm2d(n_filters * blk_pt.expansion * 2)
                               ))
    sanity_check(blk_nnt, blk_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
def test_resnet_bottleneck_block(device):
    shape = (64, 64, 32, 32)
    n_filters = 64

    blk_nnt = nnt.ResNetBottleneckBlock(shape, n_filters)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, downsample=T.nn.Sequential(
        conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 1),
        T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)))
    sanity_check(blk_nnt, blk_pt, shape, device=device)

    blk_nnt = nnt.ResNetBottleneckBlock(shape, n_filters, stride=2)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, stride=2,
                               downsample=T.nn.Sequential(
                                   conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 2),
                                   T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)
                               ))
    sanity_check(blk_nnt, blk_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
def test_max_avg_pooling_layer(device):
    shape = (64, 3, 32, 32)
    filter_size = 3
    stride1 = 1
    stride2 = 2
    padding = filter_size >> 1
    dilation = 2
    ceil_mode = True

    # test constructors
    pool_nnt = nnt.MaxPool2d(filter_size)
    assert pool_nnt.output_shape == (None, None, None, None)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=stride1, padding=0, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=stride1, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=stride2, padding=0, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=stride2, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=padding, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=padding, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=0, dilation=dilation, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=dilation, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=ceil_mode, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=ceil_mode)
    sanity_check(pool_nnt, pool_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
def test_avg_pooling_layer(device):
    shape = (64, 3, 32, 32)
    filter_size = 3
    stride1 = 1
    stride2 = 2
    padding = filter_size >> 1
    ceil_mode = True
    incl_pad = False

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                             input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=stride1, padding=0, ceil_mode=False, count_include_pad=True,
                             input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=stride1, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=stride2, padding=0, ceil_mode=False, count_include_pad=True,
                             input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=stride2, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=padding, ceil_mode=False, count_include_pad=True,
                             input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=padding, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=ceil_mode, count_include_pad=True,
                             input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=ceil_mode, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape, device=device)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=incl_pad,
                             input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=incl_pad)
    sanity_check(pool_nnt, pool_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
def test_conv2d_layer(device):
    shape = (64, 3, 32, 32)
    n_filters = 64
    filter_size = 3

    # test constructors
    conv_nnt = nnt.Conv2d(shape[1], n_filters, filter_size)
    assert conv_nnt.output_shape == (None, n_filters, None, None)

    conv_nnt = nnt.Conv2d((None, shape[1], None, None), n_filters, filter_size)
    assert conv_nnt.output_shape == (None, n_filters, None, None)

    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size)
    conv_pt = T.nn.Conv2d(shape[1], n_filters, filter_size, padding=filter_size >> 1)
    sanity_check(conv_nnt, conv_pt, shape, device=device)

    shape = (None, 3, None, 224)
    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, 112)

    shape = (None, 3, None, None)
    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, None)


@pytest.mark.parametrize('device', dev)
def test_convtranspose2d_layer(device):
    shape = (64, 3, 32, 32)
    n_filters = 64
    filter_size = 3

    # test constructors
    conv_nnt = nnt.ConvTranspose2d(shape[1], n_filters, filter_size)
    assert conv_nnt.output_shape == (None, n_filters, None, None)

    conv_nnt = nnt.ConvTranspose2d((None, shape[1], None, None), n_filters, filter_size)
    assert conv_nnt.output_shape == (None, n_filters, None, None)

    conv_nnt = nnt.ConvTranspose2d(shape, n_filters, filter_size)
    conv_pt = T.nn.ConvTranspose2d(shape[1], n_filters, filter_size, padding=filter_size >> 1)
    sanity_check(conv_nnt, conv_pt, shape, device=device)

    stride = 2
    new_size = tuple(i * stride for i in shape[2:])
    conv_nnt = nnt.ConvTranspose2d(shape, n_filters, filter_size, stride=2, output_size=new_size)
    conv_pt = T.nn.ConvTranspose2d(shape[1], n_filters, filter_size, padding=filter_size >> 1, stride=2)
    input = T.rand(*shape)
    if nnt.cuda_available:
        input = input.cuda()
        conv_pt = conv_pt.cuda()

    out_nnt = conv_nnt(input)
    out_pt = conv_pt(input, new_size)
    assert conv_nnt.output_shape[2:] == out_nnt.shape[2:] == out_pt.shape[2:]

    shape = (None, 3, None, 112)
    conv_nnt = nnt.ConvTranspose2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, 223)

    shape = (None, 3, None, None)
    conv_nnt = nnt.ConvTranspose2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, None)


@pytest.mark.parametrize('device', dev)
def test_fc_layer(device):
    shape = (64, 128)
    n_nodes = 256

    # test constructors
    fc_nnt = nnt.FC(shape[1], n_nodes)
    assert fc_nnt.output_shape == (None, n_nodes)

    fc_nnt = nnt.FC((None, shape[1]), n_nodes)
    assert fc_nnt.output_shape == (None, n_nodes)

    fc_nnt = nnt.FC(shape, n_nodes)
    fc_pt = T.nn.Linear(shape[1], n_nodes)
    sanity_check(fc_nnt, fc_pt, shape, device=device)


@pytest.mark.parametrize('device', dev)
def test_batchnorm2d_layer(device):
    shape = (64, 3, 32, 32)

    # test constructors
    bn_nnt = nnt.BatchNorm2d(shape[1])
    assert bn_nnt.output_shape == (None, shape[1], None, None)

    bn_nnt = nnt.BatchNorm2d((None, shape[1], None, None))
    assert bn_nnt.output_shape == (None, shape[1], None, None)

    bn_nnt = nnt.BatchNorm2d(shape)
    bn_pt = T.nn.BatchNorm2d(shape[1])
    sanity_check(bn_nnt, bn_pt, shape, device=device)

    shape = (None, 3, None, None)
    bn_nnt = nnt.BatchNorm2d(shape)
    assert bn_nnt.output_shape == (None, 3, None, None)


@pytest.mark.parametrize('device', dev)
def test_softmax(device):
    shape = (64, 512)
    out_features = 1000
    a = T.rand(*shape).to(device)

    sm = nnt.Softmax(shape, out_features).to(device)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias, dim=1)
    testing.assert_allclose(sm(a), expected)
    testing.assert_allclose(sm.output_shape, expected.shape)

    sm = nnt.Softmax(shape, out_features, dim=0).to(device)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias, dim=0)
    testing.assert_allclose(sm(a), expected)
    testing.assert_allclose(sm.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_global_avgpool2d(device):
    shape = (4, 3, 256, 512)
    a = T.rand(*shape).to(device)

    pool = nnt.GlobalAvgPool2D(input_shape=shape)
    expected = T.mean(a, (2, 3))
    testing.assert_allclose(pool(a), expected)
    testing.assert_allclose(pool.output_shape, expected.shape)

    pool = nnt.GlobalAvgPool2D(keepdim=True, input_shape=shape)
    expected = T.mean(a, (2, 3)).unsqueeze(-1).unsqueeze(-1)
    testing.assert_allclose(pool(a), expected)
    testing.assert_allclose(pool.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_dimshuffle_layer(device):
    shape = (64, 512)
    a = T.rand(*shape).to(device)

    dimshuffle = nnt.DimShuffle((1, 0), input_shape=shape)
    expected = a.transpose(1, 0)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle((0, 1, 'x', 'x'), input_shape=shape)
    expected = a.unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle((1, 0, 'x', 'x'), input_shape=shape)
    expected = a.permute(1, 0).unsqueeze(2).unsqueeze(2)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle((0, 'x', 1, 'x'), input_shape=shape)
    expected = a.unsqueeze(2).permute(0, 2, 1).unsqueeze(3)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle((1, 'x', 'x', 0), input_shape=shape)
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(1)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)

    dimshuffle = nnt.DimShuffle((1, 'x', 0, 'x', 'x'), input_shape=shape)
    expected = a.permute(1, 0).unsqueeze(1).unsqueeze(3).unsqueeze(3)
    testing.assert_allclose(dimshuffle(a), expected)
    testing.assert_allclose(dimshuffle.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_lambda(device):
    shape = (3, 10, 5, 5)
    a = T.rand(*shape).to(device)

    def foo1(x, y):
        return x ** y

    sqr = nnt.Lambda(foo1, y=2., input_shape=shape)
    expected = a ** 2.
    testing.assert_allclose(sqr(a), expected)
    testing.assert_allclose(sqr.output_shape, expected.shape)

    def foo2(x, fr, to):
        return x[:, fr:to]

    fr = 3
    to = 7
    a = T.rand(*shape).to(device)
    if cuda_available:
        a = a.cuda()

    slice = nnt.Lambda(foo2, fr=fr, to=to, input_shape=shape)
    expected = a[:, fr:to]
    testing.assert_allclose(slice(a), expected)
    testing.assert_allclose(slice.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_cat(device):
    shape1 = (3, 2, 4, 4)
    shape2 = (3, 5, 4, 4)
    out_channels = 5

    a = T.rand(*shape1).to(device)
    b = T.rand(*shape2).to(device)

    cat = nnt.Cat(1, nnt.Lambda(lambda x: x + 1., output_shape=shape1, input_shape=shape1),
                  nnt.Lambda(lambda x: 2. * x, output_shape=shape1, input_shape=shape1))
    expected = T.cat((a + 1., a * 2.), 1)
    testing.assert_allclose(cat(a), expected)
    testing.assert_allclose(cat.output_shape, expected.shape)

    cat = nnt.Cat(1, a, b, nnt.Lambda(lambda x: 2. * x, output_shape=shape1, input_shape=shape1))
    expected = T.cat((a, b, a * 2.), 1)
    testing.assert_allclose(cat(a), expected)
    testing.assert_allclose(cat.output_shape, expected.shape)

    con_cat = nnt.ConcurrentCat(1, nnt.Lambda(lambda x: x + 1., output_shape=shape1, input_shape=shape1),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape2, input_shape=shape2))
    expected = T.cat((a + 1., b * 2.), 1)
    testing.assert_allclose(con_cat(a, b), expected)
    testing.assert_allclose(con_cat.output_shape, expected.shape)

    con_cat = nnt.ConcurrentCat(1, a, b,
                                nnt.Lambda(lambda x: x + 1., output_shape=shape1, input_shape=shape1),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape2, input_shape=shape2))
    expected = T.cat((a, b, a + 1., b * 2.), 1)
    testing.assert_allclose(con_cat(a, b), expected)
    testing.assert_allclose(con_cat.output_shape, expected.shape)

    seq_cat = nnt.SequentialCat(2, nnt.Lambda(lambda x: x + 1., output_shape=shape1, input_shape=shape1),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape1, input_shape=shape1))
    expected = T.cat((a + 1., (a + 1.) * 2.), 2)
    testing.assert_allclose(seq_cat(a), expected)
    testing.assert_allclose(seq_cat.output_shape, expected.shape)

    seq_cat = nnt.SequentialCat(2, a,
                                nnt.Lambda(lambda x: x + 1., output_shape=shape1, input_shape=shape1),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape1, input_shape=shape1))
    expected = T.cat((a, a + 1., (a + 1.) * 2.), 2)
    testing.assert_allclose(seq_cat(a), expected)
    testing.assert_allclose(seq_cat.output_shape, expected.shape)

    m1 = nnt.Conv2d(a.shape, out_channels, 3).to(device)
    m2 = nnt.Conv2d(b.shape, out_channels, 3).to(device)
    con_cat = nnt.ConcurrentCat(1, a, m1, b, m2)
    expected = T.cat((a, m1(a), b, m2(b)), 1)
    testing.assert_allclose(con_cat(a, b), expected)
    testing.assert_allclose(con_cat.output_shape, expected.shape)

    m1 = nnt.Conv2d(a.shape, out_channels, 3).to(device)
    m2 = nnt.Conv2d(out_channels, out_channels, 3).to(device)
    seq_cat = nnt.SequentialCat(1, a, m1, m2, b)
    expected = T.cat((a, m1(a), m2(m1(a)), b), 1)
    testing.assert_allclose(seq_cat(a), expected)
    testing.assert_allclose(seq_cat.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_sum(device):
    shape = (3, 2, 4, 4)
    out_channels = 5

    a = T.rand(*shape).to(device)
    b = T.rand(*shape).to(device)

    sum = nnt.Sum(nnt.Lambda(lambda x: x + 1., output_shape=shape, input_shape=shape),
                  nnt.Lambda(lambda x: 2. * x, output_shape=shape, input_shape=shape))
    expected = (a + 1.) + (a * 2.)
    testing.assert_allclose(sum(a), expected)
    testing.assert_allclose(sum.output_shape, expected.shape)

    sum = nnt.Sum(a, b, nnt.Lambda(lambda x: 2. * x, output_shape=shape, input_shape=shape))
    expected = a + b + (a * 2.)
    testing.assert_allclose(sum(a), expected)
    testing.assert_allclose(sum.output_shape, expected.shape)

    con_sum = nnt.ConcurrentSum(nnt.Lambda(lambda x: x + 1., output_shape=shape, input_shape=shape),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape, input_shape=shape))
    expected = (a + 1.) + (b * 2.)
    testing.assert_allclose(con_sum(a, b), expected)
    testing.assert_allclose(con_sum.output_shape, expected.shape)

    con_sum = nnt.ConcurrentSum(a, b,
                                nnt.Lambda(lambda x: x + 1., output_shape=shape, input_shape=shape),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape, input_shape=shape))
    expected = a + b + (a + 1.) + (b * 2.)
    testing.assert_allclose(con_sum(a, b), expected)
    testing.assert_allclose(con_sum.output_shape, expected.shape)

    seq_sum = nnt.SequentialSum(nnt.Lambda(lambda x: x + 1., output_shape=shape, input_shape=shape),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape, input_shape=shape))
    expected = (a + 1.) + (a + 1.) * 2.
    testing.assert_allclose(seq_sum(a), expected)
    testing.assert_allclose(seq_sum.output_shape, expected.shape)

    seq_sum = nnt.SequentialSum(a,
                                nnt.Lambda(lambda x: x + 1., output_shape=shape, input_shape=shape),
                                nnt.Lambda(lambda x: 2. * x, output_shape=shape, input_shape=shape))
    expected = a + (a + 1.) + (a + 1.) * 2.
    testing.assert_allclose(seq_sum(a), expected)
    testing.assert_allclose(seq_sum.output_shape, expected.shape)

    m1 = nnt.Conv2d(a.shape, out_channels, 3).to(device)
    m2 = nnt.Conv2d(b.shape, out_channels, 3).to(device)
    con_sum = nnt.ConcurrentSum(m1, m2)
    expected = m1(a) + m2(b)
    testing.assert_allclose(con_sum(a, b), expected)
    testing.assert_allclose(con_sum.output_shape, expected.shape)

    m1 = nnt.Conv2d(a.shape[1], a.shape[1], 3).to(device)
    m2 = nnt.Conv2d(a.shape[1], a.shape[1], 3).to(device)
    seq_sum = nnt.SequentialSum(a, m1, m2, b)
    expected = a + m1(a) + m2(m1(a)) + b
    testing.assert_allclose(seq_sum(a), expected)
    testing.assert_allclose(seq_sum.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
def test_spectral_norm(device):
    from copy import deepcopy
    import torch.nn as nn

    seed = 48931
    input = T.rand(10, 3, 5, 5).to(device)

    net = nnt.Sequential(
        nnt.Sequential(
            nnt.Conv2d(3, 16, 3),
            nnt.Conv2d(16, 32, 3)
        ),
        nnt.Sequential(
            nnt.Conv2d(32, 64, 3),
            nnt.Conv2d(64, 128, 3),
        ),
        nnt.BatchNorm2d(128),
        nnt.GroupNorm(128, 4),
        nnt.LayerNorm((None, 128, 5, 5)),
        nnt.GlobalAvgPool2D(),
        nnt.FC(128, 1)
    ).to(device)

    net_pt_sn = deepcopy(net)
    T.manual_seed(seed)
    if cuda_available:
        T.cuda.manual_seed_all(seed)

    net_pt_sn[0][0] = nn.utils.spectral_norm(net_pt_sn[0][0])
    net_pt_sn[0][1] = nn.utils.spectral_norm(net_pt_sn[0][1])
    net_pt_sn[1][0] = nn.utils.spectral_norm(net_pt_sn[1][0])
    net_pt_sn[1][1] = nn.utils.spectral_norm(net_pt_sn[1][1])
    net_pt_sn[6] = nn.utils.spectral_norm(net_pt_sn[6])

    T.manual_seed(seed)
    if cuda_available:
        T.cuda.manual_seed_all(seed)

    net_nnt_sn = nnt.spectral_norm(net)

    net_pt_sn(input)
    net_nnt_sn(input)

    assert not hasattr(net_nnt_sn[2], 'weight_u')
    assert not hasattr(net_nnt_sn[3], 'weight_u')
    assert not hasattr(net_nnt_sn[4], 'weight_u')

    testing.assert_allclose(net_pt_sn[0][0].weight, net_nnt_sn[0][0].weight)
    testing.assert_allclose(net_pt_sn[0][1].weight, net_nnt_sn[0][1].weight)
    testing.assert_allclose(net_pt_sn[1][0].weight, net_nnt_sn[1][0].weight)
    testing.assert_allclose(net_pt_sn[1][1].weight, net_nnt_sn[1][1].weight)
    testing.assert_allclose(net_pt_sn[6].weight, net_nnt_sn[6].weight)
