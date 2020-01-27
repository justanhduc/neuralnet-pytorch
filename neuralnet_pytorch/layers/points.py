from functools import partial
import numpy as np
import torch as T
import torch.nn as nn

from .abstract import wrapper, Module, Sequential
from .convolution import Conv2d, DepthwiseSepConv2D, FC
from .. import utils
from ..utils import _pointset_shape

__all__ = ['XConv', 'GraphConv', 'BatchGraphConv', 'GraphXConv']


@utils.add_custom_repr
class XConv(Module):
    """
    Performs X-Convolution on unordered set as in `this paper`_.

    .. _this paper: https://papers.nips.cc/paper/7362-pointcnn-convolution-on-x-transformed-points.pdf

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    feature_dim : int
        dimension of the input features.
    out_channels : int
        number of channels produced by the convolution.
    out_features : int
        size of each output sample.
    num_neighbors : int
        size of the convolving kernel.
    depth_mul
        depth multiplier for intermediate result of depthwise convolution
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    dropout : bool
        whether to use dropout.
    bn : bool
        whether to use batch normalization.
    kwargs
        extra keyword arguments to pass to activation.
    """
    def __init__(self, input_shape, feature_dim, out_channels, out_features, num_neighbors, depth_mul,
                 activation='relu', dropout=None, bn=True, **kwargs):
        input_shape = _pointset_shape(input_shape)

        super().__init__(input_shape)
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.out_features = out_features
        self.depth_mul = depth_mul
        self.activation = utils.function(activation, **kwargs)
        self.dropout = dropout
        self.bn = bn

        self.fcs = Sequential(input_shape=input_shape)
        self.fcs.add_module('fc1', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout1', wrapper(self.output_shape, nn.Dropout2d, p=dropout))
        self.fcs.add_module('fc2', FC(self.fcs.output_shape, out_features, activation=activation))
        if dropout:
            self.fcs.add_module('dropout2', wrapper(self.output_shape, nn.Dropout2d, p=dropout))

        from neuralnet_pytorch.layers.resizing import DimShuffle
        from neuralnet_pytorch.layers.normalization import BatchNorm2d

        self.x_trans = Sequential(input_shape=input_shape[:2] + (num_neighbors, input_shape[-1]))
        self.x_trans.add_module('dimshuffle1', DimShuffle(self.x_trans.output_shape, (0, 3, 1, 2)))
        self.x_trans.add_module('conv', Conv2d(self.x_trans.output_shape, num_neighbors ** 2, (1, num_neighbors),
                                               activation=activation, padding='valid', **kwargs))
        self.x_trans.add_module('dimshuffle2', DimShuffle(self.x_trans.output_shape, (0, 2, 3, 1)))
        self.x_trans.add_module('fc1', FC(self.x_trans.output_shape, num_neighbors ** 2, activation='relu', **kwargs))
        self.x_trans.add_module('fc2', FC(self.x_trans.output_shape, num_neighbors ** 2, **kwargs))

        self.end_conv = Sequential(input_shape=input_shape[:2] + (num_neighbors, feature_dim + out_features))
        self.end_conv.add_module('dimshuffle1', DimShuffle(self.end_conv.output_shape, (0, 3, 1, 2)))
        self.end_conv.add_module('conv',
                                 DepthwiseSepConv2D(self.end_conv.output_shape, out_channels, (1, num_neighbors),
                                                    depth_mul=depth_mul, activation=None if bn else activation,
                                                    padding='valid', **kwargs))
        if bn:
            self.end_conv.add_module('bn', BatchNorm2d(self.end_conv.output_shape, momentum=.9, activation=activation))
        self.end_conv.add_module('dimshuffle2', DimShuffle(self.end_conv.output_shape, (0, 2, 3, 1)))

    def forward(self, *input, **kwargs):
        rep_pt, pts, fts = input

        if fts is not None:
            assert rep_pt.size()[0] == pts.size()[0] == fts.size()[0]  # Check N is equal.
            assert rep_pt.size()[1] == pts.size()[1] == fts.size()[1]  # Check P is equal.
            assert pts.size()[2] == fts.size()[2] == self.num_neighbors  # Check K is equal.
            assert fts.size()[3] == self.feature_dim  # Check C_in is equal.
        else:
            assert rep_pt.size()[0] == pts.size()[0]  # Check N is equal.
            assert rep_pt.size()[1] == pts.size()[1]  # Check P is equal.
            assert pts.size()[2] == self.num_neighbors  # Check K is equal.
        assert rep_pt.size()[2] == pts.size()[3] == self.input_shape[-1]  # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = T.unsqueeze(rep_pt, dim=2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center  # (N, P, K, dims)

        # Individually lift each point into C_mid space.
        fts_lifted = self.fcs(pts_local)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = T.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.num_neighbors, self.num_neighbors)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        # Weight and permute fts_cat with the learned X.
        fts_X = T.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim=2)
        return fts_p

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[:2] + (self.out_channels,)

    def extra_repr(self):
        s = ('{input_shape}, feature_dim={feature_dim}, out_channels={out_channels}, out_features={out_features}, '
             'num_neighbors={num_neighbors}, depth_mul={depth_mul}, dropout={dropout}, bn={bn}')

        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


class GraphConv(FC):
    """
    Performs graph convolution as described in https://arxiv.org/abs/1609.02907.
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, **kwargs):
        super().__init__(input_shape, out_features, bias, activation=activation, **kwargs)

    def reset_parameters(self):
        if self.weights_init is None:
            stdv = 1. / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weights_init(self.weight)

        if self.bias is not None:
            if self.bias_init is None:
                stdv = 1. / np.sqrt(self.weight.size(1))
                self.bias.data.uniform_(-stdv, stdv)
            else:
                self.bias_init(self.bias)

    def forward(self, input, adj, *args, **kwargs):
        support = T.mm(input, self.weight.t())
        output = T.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output


class BatchGraphConv(GraphConv):
    """
    Performs graph convolution as described in https://arxiv.org/abs/1609.02907 on a batch of graphs.
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py.

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        The first dim should be batch size.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, bias=True, activation=None, **kwargs):
        input_shape = _pointset_shape(input_shape)
        super().__init__(input_shape, out_features, bias, activation=activation, **kwargs)

    def forward(self, input, adj, *args, **kwargs):
        """
        Performs graphs convolution.

        :param input:
            a ``list``/``tuple`` of 2D matrices.
        :param adj:
            a block diagonal matrix whose diagonal consists of
            adjacency matrices of the input graphs.
        :return:
            a batch of embedded graphs.
        """

        shapes = [input_.shape[0] for input_ in input]
        X = T.cat(input, 0)
        output = super().forward(X, adj)
        output = T.split(output, shapes)
        return output


@utils.add_custom_repr
class GraphXConv(Module):
    """
    Performs GraphX Convolution as described here_.
    **Disclaimer:** I am the first author of the paper.

    .. _here:
        http://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_GraphX-Convolution_for_Point_Cloud_Deformation_in_2D-to-3D_Conversion_ICCV_2019_paper.html

    Parameters
    ----------
    input_shape
        shape of the input tensor.
        The first dim should be batch size.
        If an integer is passed, it is treated as the size of each input sample.
    out_features : int
        size of each output sample.
    out_instances : int
        resolution of the output point clouds.
        If not specified, output will have the same resolution as input.
        Default: ``None``.
    rank
        if specified and smaller than `num_out_points`, the mixing matrix will
        be broken into a multiplication of two matrices of sizes
        ``(num_out_points, rank)`` and ``(rank, input_shape[1])``.
    bias
        whether to use bias.
        Default: ``True``.
    activation
        non-linear function to activate the linear result.
        It accepts any callable function
        as well as a recognizable ``str``.
        A list of possible ``str`` is in :func:`~neuralnet_pytorch.utils.function`.
    weights_init
        a kernel initialization method from :mod:`torch.nn.init`.
    bias_init
        a bias initialization method from :mod:`torch.nn.init`.
    kwargs
        extra keyword arguments to pass to activation.
    """

    def __init__(self, input_shape, out_features, out_instances=None, rank=None, bias=True, activation=None,
                 weights_init=None, bias_init=None, **kwargs):
        input_shape = _pointset_shape(input_shape)
        super().__init__(input_shape=input_shape)

        self.out_features = out_features
        self.out_instances = out_instances if out_instances else input_shape[-2]
        if rank:
            assert rank <= self.out_instances // 2, 'rank should be smaller than half of num_out_points'

        self.rank = rank
        self.activation = utils.function(activation, **kwargs)
        pattern = list(range(len(input_shape)))
        pattern[-1], pattern[-2] = pattern[-2], pattern[-1]
        self.pattern = pattern
        self.weights_init = weights_init
        self.bias_init = bias_init

        self.weight = nn.Parameter(T.Tensor(out_features, input_shape[-1]))
        if self.rank is None:
            self.mixing = nn.Parameter(T.Tensor(self.out_instances, input_shape[-2]))
        else:
            self.mixing_u = nn.Parameter(T.Tensor(self.rank, input_shape[-2]))
            self.mixing_v = nn.Parameter(T.Tensor(self.out_instances, self.rank))

        if bias:
            self.bias = nn.Parameter(T.Tensor(out_features))
            self.mixing_bias = nn.Parameter(T.Tensor(self.out_instances))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mixing_bias', None)

        self.reset_parameters()

    def forward(self, input):
        output = utils.dimshuffle(input, self.pattern)
        mixing = T.mm(self.mixing_v, self.mixing_u) if self.rank else self.mixing
        output = T.matmul(output, mixing.t())
        if self.mixing_bias is not None:
            output = output + self.mixing_bias

        output = utils.dimshuffle(output, self.pattern)
        output = T.matmul(output, self.weight.t())
        if self.bias is not None:
            output = output + self.bias

        return self.activation(output)

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[:-2] + (self.out_instances, self.out_features)

    def reset_parameters(self):
        super().reset_parameters()
        weights_init = partial(nn.init.kaiming_uniform_, a=np.sqrt(5)) if self.weights_init is None \
            else self.weights_init

        weights_init(self.weight)
        if self.rank:
            weights_init(self.mixing_u)
            weights_init(self.mixing_v)
        else:
            weights_init(self.mixing)

        if self.bias is not None:
            if self.bias_init is None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                bias_init = partial(nn.init.uniform_, a=-bound, b=bound)
            else:
                bias_init = self.bias_init

            bias_init(self.bias)
            nn.init.zeros_(self.mixing_bias)

    def extra_repr(self):
        s = '{input_shape}, out_features={out_features}, out_instances={out_instances}, rank={rank}'
        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s
