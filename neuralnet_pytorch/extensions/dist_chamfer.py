from torch import nn
from torch.autograd import Function
import neuralnet_pytorch.ext as ext

__all__ = ['chamfer_distance']


class ChamferFunction(Function):
    """
    Chamfer's distance module @thibaultgroueix
    GPU tensors only
    """

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1, dist2, idx1, idx2 = ext.chamfer_forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        grad_xyz1, grad_xyz2 = ext.chamfer_backward(xyz1, xyz2, graddist1, graddist2, idx1, idx2)
        return grad_xyz1, grad_xyz2


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, input1, input2):
        return ChamferFunction.apply(input1, input2)


chamfer_distance = ChamferDistance()
