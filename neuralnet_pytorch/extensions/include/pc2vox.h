#pragma once
#include <torch/torch.h>

torch::Tensor pointcloud_to_voxel_forward(torch::Tensor pc, int voxel_size,
                                          float grid_size, bool filter_outlier);
