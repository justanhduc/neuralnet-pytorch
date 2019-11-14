import torch as T
import numpy as np
from torch import testing

import neuralnet_pytorch as nnt
from neuralnet_pytorch import cuda_available


def test_gin_config():
    try:
        from neuralnet_pytorch import gin_nnt as gin
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

    import os
    config_file = os.path.join(os.path.dirname(__file__), 'test_files/test.gin')
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


def test_monitor():
    shape = (64, 512)
    a = T.rand(*shape)
    if cuda_available:
        a = a.cuda()

    n_epochs = 5
    n_iters = 10
    print_freq = 5

    mon = nnt.Monitor(use_tensorboard=True, print_freq=print_freq)
    mon.dump('foo.pkl', a)
    loaded = mon.load('foo.pkl')
    testing.assert_allclose(a, loaded)
    mon.reset()

    for epoch in mon.iter_epoch(range(n_epochs)):
        for it in mon.iter_batch(range(n_iters)):
            mon.dump('foo.pkl', a + it, keep=3)
            mon.plot('parabol1', (it + epoch) ** 2)
            mon.hist('histogram1', a + (it * epoch))
            mon.imwrite('image', a[None, None])

    loaded = mon.load('foo.pkl', version=48)
    testing.assert_allclose(a + 8., loaded)
    mon.reset()

    mon.dump('foo.txt', nnt.utils.to_numpy(a), 'txt')
    loaded = mon.load('foo.txt', 'txt', dtype='float32')
    if cuda_available:
        loaded = nnt.utils.to_cuda(loaded)
    testing.assert_allclose(a, loaded)
    mon.reset()

    for epoch in mon.iter_epoch(range(n_epochs)):
        for it in mon.iter_batch(range(n_iters)):
            mon.plot('parabol2', (it + epoch) ** 2)
            mon.hist('histogram2', a + (it * epoch))
            mon.dump('foo.txt', nnt.utils.to_numpy(a + it), 'txt', keep=3)
    loaded = mon.load('foo.txt', 'txt', version=48, dtype='float32')
    if cuda_available:
        loaded = nnt.utils.to_cuda(loaded)
    testing.assert_allclose(a + 8., loaded)
    mon.reset()

    mon.dump('foo.pt', {'a': a}, 'torch')
    loaded = mon.load('foo.pt', 'torch')['a']
    testing.assert_allclose(a, loaded)
    mon.reset()

    for epoch in mon.iter_epoch(range(n_epochs)):
        for it in mon.iter_batch(range(n_iters)):
            mon.plot('parabol3', (it + epoch) ** 2)
            mon.hist('histogram3', a + (it * epoch))
            mon.dump('foo.pt', {'a': a + it}, 'torch', keep=4)
    loaded = mon.load('foo.pt', 'torch', version=49)['a']
    testing.assert_allclose(a + 9, loaded)
    mon.reset()
