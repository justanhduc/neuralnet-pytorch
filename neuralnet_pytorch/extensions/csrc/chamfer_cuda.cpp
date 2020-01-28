#include "chamfer_cuda.h"

#include "utils.h"

std::vector<torch::Tensor> chamfer_cuda_forward(at::Tensor xyz1,
                                                at::Tensor xyz2);
std::vector<torch::Tensor> chamfer_cuda_backward(
  at::Tensor xyz1, at::Tensor xyz2, at::Tensor graddist1, at::Tensor graddist2,
  at::Tensor idx1, at::Tensor idx2);

std::vector<torch::Tensor>
chamfer_forward(at::Tensor xyz1, at::Tensor xyz2)
{
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), xyz2.size(2));
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  return chamfer_cuda_forward(xyz1, xyz2);
}

std::vector<torch::Tensor>
chamfer_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor graddist1,
                 at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2)
{
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(1), idx1.size(1));
  CHECK_EQ(xyz2.size(1), idx2.size(1));
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  return chamfer_cuda_backward(xyz1, xyz2, graddist1, graddist2, idx1, idx2);
}
