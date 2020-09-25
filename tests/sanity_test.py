import torch as T
import numpy as np
from torch import testing
from torch.nn import functional as F
import pytest
import torchvision
import sympy as sp

import neuralnet_pytorch as nnt
from neuralnet_pytorch import cuda_available
from neuralnet_pytorch import zoo

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
    else:
        expected = module2(args)
        testing.assert_allclose(module1(*args), expected)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return T.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('filter_size', 'stride', 'padding', 'dilation'),
    ((3, 1, 0, 1),
     (3, 1, 1, 1),
     (3, 2, 0, 1),
     (3, 2, 1, 1),
     (3, 1, 0, 2),
     (3, 1, 1, 2),
     (3, 2, 0, 2),
     (3, 2, 1, 2),
     (4, 1, 0, 1),
     (4, 1, 1, 1),
     (4, 2, 0, 1),
     (4, 2, 1, 1),
     (4, 1, 2, 1),
     (4, 2, 2, 1),
     (4, 1, 0, 2),
     (4, 1, 1, 2),
     (4, 2, 0, 2),
     (4, 2, 1, 2),
     (4, 1, 2, 2),
     (4, 2, 2, 2))
)
def test_conv2d_layer(device, filter_size, stride, padding, dilation):
    shape_sym = ('b', 3, 'h', 'w')
    shape = (2, 3, 10, 10)
    n_filters = 5

    conv_nnt = nnt.Conv2d(shape_sym, n_filters, filter_size, stride, padding, dilation).to(device)
    conv_pt = T.nn.Conv2d(shape[1], n_filters, filter_size, stride, padding, dilation).to(device)
    sanity_check(conv_nnt, conv_pt, shape, device=device)

    input = T.arange(np.prod(shape)).view(*shape).float().to(device)
    out_pt = conv_pt(input)

    h = conv_nnt.output_shape[2].subs(conv_nnt.input_shape[2], shape[2])
    w = conv_nnt.output_shape[3].subs(conv_nnt.input_shape[3], shape[3])
    assert h == out_pt.shape[2]
    assert w == out_pt.shape[3]


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('filter_size', 'stride', 'padding', 'output_size', 'output_shape'),
    ((3, 1, 0, None, (2, 5, 12, 12)),
     (3, 1, 1, None, (2, 5, 10, 10)),
     (4, 1, 0, None, (2, 5, 13, 13)),
     (4, 1, 1, None, (2, 5, 11, 11)),
     (4, 1, 2, None, (2, 5, 9, 9)),
     (3, 2, 0, None, (2, 5, 21, 21)),
     (3, 2, 1, None, (2, 5, 19, 19)),
     (3, 2, 1, (20, 20), (2, 5, 20, 20)),
     (3, 2, 2, None, (2, 5, 17, 17)),
     (4, 2, 0, None, (2, 5, 22, 22)),
     (4, 2, 1, (21, 21), (2, 5, 21, 21)),
     (4, 2, 1, None, (2, 5, 20, 20)),
     (4, 2, 2, None, (2, 5, 18, 18)))
)
def test_convtranspose2d_layer(device, filter_size, stride, padding, output_size, output_shape):
    shape_sym = ('b', 3, 'h', 'w')
    shape = (2, 3, 10, 10)
    n_filters = 5

    conv_nnt = nnt.ConvTranspose2d(shape_sym, n_filters, filter_size, stride=stride, padding=padding,
                                   output_size=output_size).to(device)
    conv_pt = T.nn.ConvTranspose2d(shape[1], n_filters, filter_size, padding=padding, stride=stride).to(device)

    input = T.arange(np.prod(shape)).view(*shape).float().to(device)
    out_pt = conv_pt(input, output_size)
    if output_size is None:
        sanity_check(conv_nnt, conv_pt, shape, device=device)
        h = conv_nnt.output_shape[2].subs(conv_nnt.input_shape[2], shape[2])
        w = conv_nnt.output_shape[3].subs(conv_nnt.input_shape[3], shape[3])
        assert h == out_pt.shape[2]
        assert w == out_pt.shape[3]
    else:
        out_nnt = conv_nnt(input)
        assert out_pt.shape == out_nnt.shape


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('depth_mul', (1, 2))
def test_depthwise_sepconv(device, depth_mul):
    shape = (2, 3, 5, 5)
    n_filters = 4
    filter_size = 3
    a = T.arange(np.prod(shape)).view(*shape).float().to(device)

    conv_dw = nnt.DepthwiseSepConv2D(shape, n_filters, 3, depth_mul=depth_mul, bias=False).to(device)
    conv = nnt.Conv2d(shape, n_filters, filter_size, bias=False).to(device)

    weight = T.stack([F.conv2d(
        conv_dw.depthwise.weight[i:i+1].transpose(0, 1), conv_dw.pointwise.weight[:, i:i+1]).squeeze()
                      for i in range(shape[1] * depth_mul)])
    weight = weight.view(shape[1], depth_mul, n_filters, 3, 3)
    weight = weight.sum(1).transpose(0, 1)
    conv.weight.data = weight
    testing.assert_allclose(conv_dw(a), conv(a))


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('shape', ((2, 3), (None, 3), 3))
def test_fc_layer(device, shape):
    out_features = 4

    # test constructors
    fc_nnt = nnt.FC(shape, out_features)
    fc_nnt = nnt.FC((2, 3), out_features)
    fc_pt = T.nn.Linear(shape[1] if isinstance(shape, tuple) else shape, out_features)
    sanity_check(fc_nnt, fc_pt, shape=(2, 3), device=device)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('dim', (0, 1))
def test_softmax(device, dim):
    shape = (2, 3)
    out_features = 4
    a = T.rand(*shape).to(device)

    sm = nnt.Softmax(shape, out_features, dim=dim).to(device)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias, dim=dim)
    testing.assert_allclose(sm(a), expected)
    testing.assert_allclose(sm.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('shape', ((2, 3, 4, 5), 3, ('b', 3, 4, 'w'),
                                   pytest.param((2, 'c', 4, 5), marks=pytest.mark.xfail)))
def test_batchnorm2d_layer(device, shape):
    input_shape = (2, 3, 4, 5)
    bn_nnt = nnt.BatchNorm2d(shape)
    bn_pt = T.nn.BatchNorm2d(input_shape[1])
    sanity_check(bn_nnt, bn_pt, shape=(2, 3, 4, 5), device=device)

    for i in range(len(input_shape)):
        if isinstance(bn_nnt.input_shape[i], sp.Symbol):
            s = bn_nnt.output_shape[i].subs(bn_nnt.input_shape[i], input_shape[i])
            assert s == input_shape[i]


@pytest.mark.parametrize('device', dev)
def test_resnet_basic_block(device):
    from torchvision.models import resnet
    shape = (64, 64, 32, 32)
    n_filters = 64

    # test constructors
    blk_nnt = nnt.ResNetBasicBlock(shape[1], n_filters)
    assert len(blk_nnt.output_shape) == 4
    assert blk_nnt.output_shape[1] == 64

    blk_nnt = nnt.ResNetBasicBlock((None, shape[1], None, None), n_filters)
    assert len(blk_nnt.output_shape) == 4
    assert blk_nnt.output_shape[1] == 64

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
    from torchvision.models import resnet
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
@pytest.mark.parametrize(
    ('filter_size', 'stride', 'padding', 'dilation', 'ceil_mode', 'count_include_pad', 'shape', 'output_shape'),
    ((2, None, 0, 1, False, True, (2, 3, 4, 4), (2, 3, 2, 2)),
     (2, None, 0, 1, False, False, (2, 3, 5, 5), (2, 3, 2, 2)),
     (2, None, 0, 1, True, True, (2, 3, 4, 4), (2, 3, 2, 2)),
     (2, None, 0, 1, True, False, (2, 3, 5, 5), (2, 3, 3, 3)),
     (3, 1, 0, 1, False, True, (2, 3, 4, 4), (2, 3, 2, 2)),
     (3, 1, 1, 1, False, False, (2, 3, 4, 4), (2, 3, 4, 4)),
     (3, 1, 1, 1, False, True, (2, 3, 5, 5), (2, 3, 5, 5)),
     (3, 1, 1, 1, True, False, (2, 3, 5, 5), (2, 3, 5, 5)))
)
def test_max_avg_pooling_layer(device, filter_size, stride, padding, dilation, ceil_mode, count_include_pad,
                               shape, output_shape):
    pool_nnt = nnt.MaxPool2d(filter_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode,
                             input_shape=shape).to(device)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=stride, padding=padding, dilation=dilation,
                             ceil_mode=ceil_mode).to(device)
    sanity_check(pool_nnt, pool_pt, shape, device=device)
    testing.assert_allclose(pool_nnt.output_shape, output_shape)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)
    sanity_check(pool_nnt, pool_pt, shape, device=device)
    testing.assert_allclose(pool_nnt.output_shape, output_shape)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize('keepdim', (True, False))
def test_global_avgpool2d(device, keepdim):
    shape = (2, 3, 4, 5)
    a = T.arange(np.prod(shape)).reshape(*shape).to(device).float()
    expected = T.tensor([[9.5000, 29.5000, 49.5000],
                         [69.5000, 89.5000, 109.5000]]).to(device)
    if keepdim:
        expected = expected.unsqueeze(-1).unsqueeze(-1)

    pool = nnt.GlobalAvgPool2D(keepdim=keepdim, input_shape=shape)
    testing.assert_allclose(pool(a), expected)
    testing.assert_allclose(pool.output_shape, expected.shape)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('pattern', 'output_shape'),
    (((1, 0), (3, 2)),
     ((0, 1, 'x', 'x'), (2, 3, 1, 1)),
     ((1, 0, 'x', 'x'), (3, 2, 1, 1)),
     ((0, 'x', 1, 'x'), (2, 1, 3, 1)),
     ((1, 'x', 'x', 0), (3, 1, 1, 2)),
     ((1, 'x', 0, 'x', 'x'), (3, 1, 2, 1, 1)))
)
def test_dimshuffle_layer(device, pattern, output_shape):
    shape = (2, 3)
    a = T.rand(*shape).to(device)

    dimshuffle = nnt.DimShuffle(pattern, input_shape=shape)
    testing.assert_allclose(dimshuffle.output_shape, output_shape)
    testing.assert_allclose(dimshuffle(a).shape, output_shape)


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


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('size', 'scale_factor', 'mode', 'align_corners', 'input_shape'),
    ((None, .6, 'bilinear', True, (2, 3, 10, 10)),
     (None, 1.3, 'bicubic', True, (2, 3, 10, 10)),
     (None, (.6, 1.3), 'nearest', None, (2, 3, 10, 10)),
     ((8, 15), None, 'bicubic', True, (2, 3, 10, 10)),
     (None, 2.1, 'linear', True, (2, 3, 10)),
     ((15,), None, 'nearest', None, (2, 3, 10)),
     (None, 2.5, 'trilinear', None, (2, 3, 10, 10, 10)),
     (None, (2.5, 1.2, .4), 'area', None, (2, 3, 10, 10, 10)),
     ((15, 10, 9), None, 'nearest', None, (2, 3, 10, 10, 10)))
)
def test_interpolate(device, size, scale_factor, mode, align_corners, input_shape):
    a = T.arange(np.prod(input_shape)).view(*input_shape).to(device).float()
    interp = nnt.Interpolate(size, scale_factor, mode, align_corners, input_shape)

    output = F.interpolate(a, size, scale_factor, mode, align_corners)
    output_nnt = interp(a)

    testing.assert_allclose(output_nnt, output)
    testing.assert_allclose(interp.output_shape, output.shape)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('dim1', 'dim2'),
    ((1, (2, 3)),
     ((2, 3), (1, 2)),
     (2, (1, 3)))
)
def test_adain(device, dim1, dim2):

    def _expected(module1, module2, input1, input2, dim1, dim2):
        output1 = module1(input1)
        output2 = module2(input2)
        mean1, std1 = T.mean(output1, dim1, keepdim=True), T.sqrt(T.var(output1, dim1, keepdim=True) + 1e-8)
        mean2, std2 = T.mean(output2, dim2, keepdim=True), T.sqrt(T.var(output2, dim2, keepdim=True) + 1e-8)
        return std2 * (output1 - mean1) / std1 + mean2

    shape = (2, 3, 4, 5)
    a = T.rand(*shape).to(device)
    b = T.rand(*shape).to(device)

    module1 = nnt.Conv2d(shape, 6, 3).to(device)
    module2 = nnt.Conv2d(shape, 6, 3).to(device)
    adain = nnt.AdaIN(module1, dim1).to(device)
    mi_adain = nnt.MultiInputAdaIN(module1, module2, dim1=dim1, dim2=dim2).to(device)
    mm_adain = nnt.MultiModuleAdaIN(module1, module2, dim1=dim1, dim2=dim2).to(device)

    actual_adain = adain(a, b)
    expected_adain = _expected(module1, module1, a, b, dim1, dim1)
    testing.assert_allclose(actual_adain, expected_adain)
    testing.assert_allclose(adain.output_shape, expected_adain.shape)

    actual_mi_adain = mi_adain(a, b)
    expected_mi_adain = _expected(module1, module2, a, b, dim1, dim2)
    testing.assert_allclose(actual_mi_adain, expected_mi_adain)
    testing.assert_allclose(mi_adain.output_shape, expected_mi_adain.shape)

    actual_mm_adain = mm_adain(a)
    expected_mm_adain = _expected(module1, module2, a, a, dim1, dim2)
    testing.assert_allclose(actual_mm_adain, expected_mm_adain)
    testing.assert_allclose(mm_adain.output_shape, expected_mm_adain.shape)


@pytest.mark.parametrize('device', dev)
@pytest.mark.parametrize(
    ('model_pt', 'model_nnt'),
    ((torchvision.models.resnet18, zoo.resnet18),
     (torchvision.models.resnet34, zoo.resnet34),
     (torchvision.models.resnet50, zoo.resnet50),
     (torchvision.models.resnet101, zoo.resnet101),
     (torchvision.models.resnet152, zoo.resnet152),
     (torchvision.models.resnext50_32x4d, zoo.resnext50_32x4d),
     (torchvision.models.resnext101_32x8d, zoo.resnext101_32x8d),
     (torchvision.models.wide_resnet50_2, zoo.wide_resnet50_2),
     (torchvision.models.wide_resnet101_2, zoo.wide_resnet101_2),
     (torchvision.models.vgg11, zoo.vgg11),
     (torchvision.models.vgg11_bn, zoo.vgg11_bn),
     (torchvision.models.vgg13, zoo.vgg13),
     (torchvision.models.vgg13_bn, zoo.vgg13_bn),
     (torchvision.models.vgg16, zoo.vgg16),
     (torchvision.models.vgg16_bn, zoo.vgg16_bn),
     (torchvision.models.vgg19, zoo.vgg19),
     (torchvision.models.vgg19_bn, zoo.vgg19_bn))
)
def test_pretrained_models(device, model_pt, model_nnt):
    model_pt = model_pt(True).to(device).eval()
    model_nnt = model_nnt(True).to(device).eval()
    input = T.rand(1, 3, 224, 224).to(device)

    expected = model_pt(input)
    actual = model_nnt(input)
    testing.assert_allclose(actual, expected)
