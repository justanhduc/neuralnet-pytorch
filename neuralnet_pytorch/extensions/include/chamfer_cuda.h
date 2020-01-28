#pragma once
#include <torch/torch.h>
#include <vector>

std::vector<torch::Tensor> chamfer_forward(at::Tensor xyz1, at::Tensor xyz2);
std::vector<torch::Tensor> chamfer_backward(at::Tensor xyz1, at::Tensor xyz2,
                                            at::Tensor graddist1,
                                            at::Tensor graddist2,
                                            at::Tensor idx1, at::Tensor idx2);
