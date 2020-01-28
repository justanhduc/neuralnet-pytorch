#pragma once
#include <torch/torch.h>
#include <vector>

at::Tensor approx_match_forward(const at::Tensor xyz1, const at::Tensor xyz2);
at::Tensor match_cost_forward(const at::Tensor xyz1, const at::Tensor xyz2,
                              const at::Tensor match);
std::vector<at::Tensor> match_cost_backward(const at::Tensor grad_cost,
                                            const at::Tensor xyz1,
                                            const at::Tensor xyz2,
                                            const at::Tensor match);
