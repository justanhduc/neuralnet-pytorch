#pragma once
#include <torch/torch.h>

torch::Tensor batch_pairwise_distance_forward(torch::Tensor x, torch::Tensor y);
