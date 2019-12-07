import torch as T
import numpy as np
from torch import testing
import pytest

import neuralnet_pytorch as nnt
from neuralnet_pytorch import cuda_available

dev = ('cpu', 'cuda') if cuda_available else ('cpu',)


@pytest.mark.parametrize('device', dev)
def test_ravel_index(device):
    shape = (2, 4, 5, 3)
    a = T.arange(np.prod(shape)).reshape(*shape).to(device)

    indices = [[1, 0, 1, 1, 0], [1, 3, 3, 2, 1], [1, 1, 4, 0, 3], [1, 2, 2, 2, 0]]
    linear_indices = nnt.utils.ravel_index(T.tensor(indices), shape)
    testing.assert_allclose(linear_indices.type_as(a), a[indices])


@pytest.mark.parametrize('device', dev)
def test_shape_pad(device):
    shape = (10, 10)
    a = T.rand(*shape).to(device)

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


@pytest.mark.parametrize('device', dev)
def test_dimshuffle(device):
    shape = (64, 512)
    a = T.rand(*shape).to(device)

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


@pytest.mark.parametrize('device', dev)
def test_flatten(device):
    shape = (10, 4, 2, 3, 6)
    a = T.rand(*shape).to(device)

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


@pytest.mark.parametrize('device', dev)
def test_reshape(device):
    shape = (10, 3, 9, 9)
    a = T.rand(*shape).to(device)

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
