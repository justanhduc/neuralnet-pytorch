import torch as T
import numpy as np
from torch import testing
import pytest

import neuralnet_pytorch as nnt
from neuralnet_pytorch import cuda_available

dev = ('cpu', 'cuda') if cuda_available else ('cpu',)


def test_gin_config():
    try:
        from neuralnet_pytorch import gin_nnt as gin
    except ImportError:
        print('Please install Gin-config first and run this test again')
        return

    @gin.configurable('net')
    def assert_same(dtype, activation, loss, optimizer, scheduler):
        assert dtype is T.float32
        assert isinstance(activation(), T.nn.Tanh)
        assert isinstance(loss(), T.nn.L1Loss)

    import os
    config_file = os.path.join(os.path.dirname(__file__), 'test_files/test.gin')
    gin.parse_config_file(config_file)
    assert_same()


@pytest.mark.parametrize('device', dev)
def test_track(device):
    shape = (2, 3, 5, 5)
    a = T.rand(*shape).to(device)

    conv1 = nnt.track('op', nnt.Conv2d(shape, 4, 3), 'all').to(device)
    conv2 = nnt.Conv2d(conv1.output_shape, 5, 3).to(device)
    intermediate = conv1(a)
    output = nnt.track('conv2_output', conv2(intermediate), 'all').to(device)
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


@pytest.mark.parametrize('device', dev)
def test_monitor(device):
    shape = (64, 512)
    a = T.rand(*shape).to(device)

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
    loaded = T.from_numpy(mon.load('foo.txt', 'txt', dtype='float32')).to(device)
    testing.assert_allclose(a, loaded)
    mon.reset()

    for epoch in mon.iter_epoch(range(n_epochs)):
        for it in mon.iter_batch(range(n_iters)):
            mon.plot('parabol2', (it + epoch) ** 2)
            mon.hist('histogram2', a + (it * epoch))
            mon.dump('foo.txt', nnt.utils.to_numpy(a + it), 'txt', keep=3)
    loaded = T.from_numpy(mon.load('foo.txt', 'txt', version=48, dtype='float32')).to(device)
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


@pytest.mark.parametrize(
    'idx',
    (slice(None, None), slice(None, 2), slice(1, None), slice(1, 3))
)
def test_slicing_sequential(idx):
    input_shape = (None, 3, 256, 256)

    a = nnt.Sequential(input_shape=input_shape)
    a.conv1 = nnt.Conv2d(a.output_shape, 64, 3)
    a.conv2 = nnt.Conv2d(a.output_shape, 128, 3)
    a.conv3 = nnt.Conv2d(a.output_shape, 256, 3)
    a.conv4 = nnt.Conv2d(a.output_shape, 512, 3)

    b = a[idx]
    start = 0 if idx.start is None else idx.start
    assert b.input_shape == a[start].input_shape

    class Foo(nnt.Sequential):
        def __init__(self, input_shape):
            super().__init__(input_shape=input_shape)

            self.conv1 = nnt.Conv2d(self.output_shape, 64, 3)
            self.conv2 = nnt.Conv2d(self.output_shape, 128, 3)
            self.conv3 = nnt.Conv2d(self.output_shape, 256, 3)
            self.conv4 = nnt.Conv2d(self.output_shape, 512, 3)

    foo = Foo(input_shape)
    b = foo[idx]
    start = 0 if idx.start is None else idx.start
    assert isinstance(b, nnt.Sequential)
    assert b.input_shape == a[start].input_shape


@pytest.mark.parametrize('bs', (pytest.param(None, marks=pytest.mark.xfail), 1, 2, 3, 4, 5))
@pytest.mark.parametrize('shuffle', (True, False))
@pytest.mark.parametrize('drop_last', (True, False))
@pytest.mark.parametrize('pin_memory', (True, False))
def test_dataloader(bs, shuffle, drop_last, pin_memory):
    from torch.utils.data import TensorDataset
    data, label = T.arange(10), T.arange(10) + 10
    dataset = TensorDataset(data, label)
    loader = nnt.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory,
                            num_workers=10)
    loader = nnt.DataPrefetcher(loader)

    for epoch in range(2):
        data_, label_ = [], []
        for batch in loader:
            data_.append(batch[0])
            label_.append(batch[1])

        data_ = sorted(T.cat(data_))
        label_ = sorted(T.cat(label_))
        if len(data_) != len(data):
            assert len(data_) == len(data) // bs * bs
            assert len(label_) == len(label) // bs * bs
        else:
            testing.assert_allclose(data_, data)
            testing.assert_allclose(label_, label)
