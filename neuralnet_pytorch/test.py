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


def sanity_check(module1, module2, shape=None, *args):
    try:
        module1.load_state_dict(module2.state_dict())
    except RuntimeError:
        params = nnt.utils.bulk_to_numpy(module2.state_dict().values())
        nnt.utils.batch_set_value(module1.state_dict().values(), params)

    if shape is not None:
        input = T.from_numpy(np.random.rand(*shape).astype('float32'))
        if cuda_available:
            input = input.cuda()
            module1 = module1.cuda()
            module2 = module2.cuda()

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


def test_gin_config():
    try:
        from neuralnet_pytorch.gin_nnt import gin
    except ImportError:
        print('Please install Gin-config first and run this test again')
        return

    @gin.configurable('net')
    def run(dtype, activation, loss, optimizer, scheduler):
        print(dtype)
        print(activation)
        print(loss)
        print(optimizer)
        print(scheduler)

    config_file = 'test_files/test.gin'
    gin.parse_config_file(config_file)
    run()


def test_track():
    shape = (2, 3, 5, 5)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    conv1 = nnt.track('op', nnt.Conv2d(shape, 4, 3), 'all')
    conv2 = nnt.Conv2d(conv1.output_shape, 5, 3)
    intermediate = conv1(a)
    output = nnt.track('conv2_output', conv2(intermediate), 'all')
    loss = T.sum(output ** 2)
    loss.backward(retain_graph=True)
    d_inter = T.autograd.grad(loss, intermediate, retain_graph=True)
    d_out = T.autograd.grad(loss, output)
    tracked = nnt.eval_tracked_variables()

    testing.assert_allclose(tracked['conv2_output'], nnt.utils.to_numpy(output))
    testing.assert_allclose(np.stack(tracked['grad_conv2_output']), nnt.utils.to_numpy(d_out[0]))
    testing.assert_allclose(tracked['op'], nnt.utils.to_numpy(intermediate))
    for d_inter_, tracked_d_inter_ in zip(d_inter, tracked['grad_op_output']):
        testing.assert_allclose(tracked_d_inter_, nnt.utils.to_numpy(d_inter_))


def test_ravel_index():
    shape = (2, 4, 5, 3)
    a = T.arange(np.prod(shape)).reshape(*shape)
    if cuda_available:
        a = a.cuda()

    indices = [[1, 0, 1, 1, 0], [1, 3, 3, 2, 1], [1, 1, 4, 0, 3], [1, 2, 2, 2, 0]]
    linear_indices = nnt.utils.ravel_index(indices, shape)
    testing.assert_allclose(linear_indices.type_as(a), a[indices])


def test_monitor():
    shape = (64, 512)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()
    n_loops = 5

    mon = nnt.Monitor()
    mon.dump('foo.pkl', a)
    loaded = mon.load('foo.pkl')
    testing.assert_allclose(a, loaded)
    mon.reset()

    for i in range(n_loops):
        with mon:
            mon.dump('foo.pkl', a + i, keep=3)
    loaded = mon.load('foo.pkl', version=2)
    testing.assert_allclose(a + 2., loaded)
    mon.reset()

    mon.dump('foo.txt', nnt.utils.to_numpy(a), 'txt')
    loaded = mon.load('foo.txt', 'txt', dtype='float32')
    if cuda_available:
        loaded = nnt.utils.to_cuda(loaded)
    testing.assert_allclose(a, loaded)
    mon.reset()

    for i in range(n_loops):
        with mon:
            mon.dump('foo.txt', nnt.utils.to_numpy(a + i), 'txt', keep=3)
    loaded = mon.load('foo.txt', 'txt', version=2, dtype='float32')
    if cuda_available:
        loaded = nnt.utils.to_cuda(loaded)
    testing.assert_allclose(a + 2., loaded)
    mon.reset()

    mon.dump('foo.pt', {'a': a}, 'torch')
    loaded = mon.load('foo.pt', 'torch')['a']
    testing.assert_allclose(a, loaded)
    mon.reset()

    for i in range(n_loops):
        with mon:
            mon.dump('foo.pt', {'a': a + i}, 'torch', keep=4)
    loaded = mon.load('foo.pt', 'torch', version=3)['a']
    testing.assert_allclose(a + 3, loaded)
    mon.reset()


def test_softmax():
    shape = (64, 512)
    out_features = 1000
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    sm = nnt.Softmax(shape, out_features)
    expected = F.softmax(T.mm(a, sm.weight.t()) + sm.bias, dim=1)
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

    pool = nnt.GlobalAvgPool2D(input_shape=shape)
    expected = T.mean(a, (2, 3))
    testing.assert_allclose(pool(a), expected)
    testing.assert_allclose(pool.output_shape, expected.shape)

    pool = nnt.GlobalAvgPool2D(keepdim=True, input_shape=shape)
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

    flatten = nnt.Flatten(2, input_shape=(None,) + shape[1:])
    expected = T.flatten(a, 2)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose((shape[0],) + flatten.output_shape[1:], expected.shape)

    flatten = nnt.Flatten(4, input_shape=(None,) + shape[1:])
    expected = T.flatten(a, 4)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose((shape[0],) + flatten.output_shape[1:], expected.shape)

    flatten = nnt.Flatten(input_shape=shape)
    expected = T.flatten(a)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose(flatten.output_shape, expected.shape)

    flatten = nnt.Flatten(1, 3, input_shape=shape)
    expected = T.flatten(a, 1, 3)
    testing.assert_allclose(flatten(a), expected)
    testing.assert_allclose(flatten.output_shape, expected.shape)


def test_reshape():
    shape = (10, 3, 9, 9)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    newshape = (-1, 9, 9)
    reshape = nnt.Reshape(newshape, input_shape=(None,) + shape[1:])
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)
    testing.assert_allclose(reshape.output_shape[1:], expected.shape[1:])

    newshape = (10, -1, 9)
    reshape = nnt.Reshape(newshape, input_shape=shape)
    expected = T.reshape(a, newshape)
    testing.assert_allclose(reshape(a), expected)
    testing.assert_allclose(reshape.output_shape, expected.shape)

    newshape = (9, 9, -1)
    reshape = nnt.Reshape(newshape, input_shape=shape)
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

    sqr = nnt.Lambda(foo1, y=2., input_shape=shape)
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

    slice = nnt.Lambda(foo2, fr=fr, to=to, input_shape=shape)
    expected = a[:, fr:to]
    testing.assert_allclose(slice(a), expected)
    testing.assert_allclose(slice.output_shape, expected.shape)


def test_cat():
    shape1 = (3, 2, 4, 4)
    shape2 = (3, 5, 4, 4)
    out_channels = 5

    a = T.rand(*shape1)
    b = T.rand(*shape2)
    if cuda_available:
        a = a.cuda()
        b = b.cuda()

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

    m1 = nnt.Conv2d(a.shape, out_channels, 3)
    m2 = nnt.Conv2d(b.shape, out_channels, 3)
    con_cat = nnt.ConcurrentCat(1, a, m1, b, m2)
    expected = T.cat((a, m1(a), b, m2(b)), 1)
    testing.assert_allclose(con_cat(a, b), expected)
    testing.assert_allclose(con_cat.output_shape, expected.shape)

    m1 = nnt.Conv2d(a.shape, out_channels, 3)
    m2 = nnt.Conv2d(out_channels, out_channels, 3)
    seq_cat = nnt.SequentialCat(1, a, m1, m2, b)
    expected = T.cat((a, m1(a), m2(m1(a)), b), 1)
    testing.assert_allclose(seq_cat(a), expected)
    testing.assert_allclose(seq_cat.output_shape, expected.shape)


def test_sum():
    shape = (3, 2, 4, 4)
    out_channels = 5

    a = T.rand(*shape)
    b = T.rand(*shape)
    if cuda_available:
        a = a.cuda()
        b = b.cuda()

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

    m1 = nnt.Conv2d(a.shape, out_channels, 3)
    m2 = nnt.Conv2d(b.shape, out_channels, 3)
    con_sum = nnt.ConcurrentSum(m1, m2)
    expected = m1(a) + m2(b)
    testing.assert_allclose(con_sum(a, b), expected)
    testing.assert_allclose(con_sum.output_shape, expected.shape)

    m1 = nnt.Conv2d(a.shape[1], a.shape[1], 3)
    m2 = nnt.Conv2d(a.shape[1], a.shape[1], 3)
    seq_sum = nnt.SequentialSum(a, m1, m2, b)
    expected = a + m1(a) + m2(m1(a)) + b
    testing.assert_allclose(seq_sum(a), expected)
    testing.assert_allclose(seq_sum.output_shape, expected.shape)


def test_resnet_bottleneck_block():
    shape = (64, 64, 32, 32)
    n_filters = 64

    blk_nnt = nnt.ResNetBottleneckBlock(shape, n_filters)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, downsample=T.nn.Sequential(
        conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 1),
        T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)))
    sanity_check(blk_nnt, blk_pt, shape)

    blk_nnt = nnt.ResNetBottleneckBlock(shape, n_filters, stride=2)
    blk_pt = resnet.Bottleneck(shape[1], n_filters, stride=2,
                               downsample=T.nn.Sequential(
                                   conv1x1(shape[1], n_filters * resnet.Bottleneck.expansion, 2),
                                   T.nn.BatchNorm2d(n_filters * resnet.Bottleneck.expansion)
                               ))
    sanity_check(blk_nnt, blk_pt, shape)


def test_resnet_basic_block():
    shape = (64, 64, 32, 32)
    n_filters = 64

    # test constructors
    blk_nnt = nnt.ResNetBasicBlock(shape[1], n_filters)
    assert blk_nnt.output_shape == (None, n_filters, None, None)

    blk_nnt = nnt.ResNetBasicBlock((None, shape[1], None, None), n_filters)
    assert blk_nnt.output_shape == (None, n_filters, None, None)

    blk_nnt = nnt.ResNetBasicBlock(shape, n_filters)
    blk_pt = resnet.BasicBlock(shape[1], n_filters)
    sanity_check(blk_nnt, blk_pt, shape)

    blk_nnt = nnt.ResNetBasicBlock(shape, n_filters * 2, stride=2)
    blk_pt = resnet.BasicBlock(shape[1], n_filters * 2, stride=2,
                               downsample=T.nn.Sequential(
                                   conv1x1(shape[1], n_filters * blk_pt.expansion * 2, 2),
                                   T.nn.BatchNorm2d(n_filters * blk_pt.expansion * 2)
                               ))
    sanity_check(blk_nnt, blk_pt, shape)


def test_max_avg_pooling_layer():
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
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=stride1, padding=0, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=stride1, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=stride2, padding=0, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=stride2, padding=0, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=padding, dilation=1, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=padding, dilation=1, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=0, dilation=dilation, ceil_mode=False, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=dilation, ceil_mode=False)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=ceil_mode, input_shape=shape)
    pool_pt = T.nn.MaxPool2d(filter_size, stride=None, padding=0, dilation=1, ceil_mode=ceil_mode)
    sanity_check(pool_nnt, pool_pt, shape)


def test_avg_pooling_layer():
    shape = (64, 3, 32, 32)
    filter_size = 3
    stride1 = 1
    stride2 = 2
    padding = filter_size >> 1
    ceil_mode = True
    incl_pad = False

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=stride1, padding=0, ceil_mode=False, count_include_pad=True, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=stride1, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=stride2, padding=0, ceil_mode=False, count_include_pad=True, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=stride2, padding=0, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=padding, ceil_mode=False, count_include_pad=True, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=padding, ceil_mode=False, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=ceil_mode, count_include_pad=True, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=ceil_mode, count_include_pad=True)
    sanity_check(pool_nnt, pool_pt, shape)

    pool_nnt = nnt.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=incl_pad, input_shape=shape)
    pool_pt = T.nn.AvgPool2d(filter_size, stride=None, padding=0, ceil_mode=False, count_include_pad=incl_pad)
    sanity_check(pool_nnt, pool_pt, shape)


def test_conv2d_layer():
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
    sanity_check(conv_nnt, conv_pt, shape)

    shape = (None, 3, None, 224)
    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, 112)

    shape = (None, 3, None, None)
    conv_nnt = nnt.Conv2d(shape, n_filters, filter_size, stride=2)
    assert conv_nnt.output_shape == (None, 64, None, None)


def test_convtranspose2d_layer():
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
    sanity_check(conv_nnt, conv_pt, shape)

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


def test_fc_layer():
    shape = (64, 128)
    n_nodes = 256

    # test constructors
    fc_nnt = nnt.FC(shape[1], n_nodes)
    assert fc_nnt.output_shape == (None, n_nodes)

    fc_nnt = nnt.FC((None, shape[1]), n_nodes)
    assert fc_nnt.output_shape == (None, n_nodes)

    fc_nnt = nnt.FC(shape, n_nodes)
    fc_pt = T.nn.Linear(shape[1], n_nodes)
    sanity_check(fc_nnt, fc_pt, shape)


def test_batchnorm2d_layer():
    shape = (64, 3, 32, 32)

    # test constructors
    bn_nnt = nnt.BatchNorm2d(shape[1])
    assert bn_nnt.output_shape == (None, shape[1], None, None)

    bn_nnt = nnt.BatchNorm2d((None, shape[1], None, None))
    assert bn_nnt.output_shape == (None, shape[1], None, None)

    bn_nnt = nnt.BatchNorm2d(shape)
    bn_pt = T.nn.BatchNorm2d(shape[1])
    sanity_check(bn_nnt, bn_pt, shape)

    shape = (None, 3, None, None)
    bn_nnt = nnt.BatchNorm2d(shape)
    assert bn_nnt.output_shape == (None, 3, None, None)
