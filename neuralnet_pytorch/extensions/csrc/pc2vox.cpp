#include <vector>

#include "pc2vox.h"
#include "utils.h"

torch::Tensor
pointcloud_to_voxel_forward(torch::Tensor pc, int voxel_size, float grid_size,
                            bool filter_outlier)
{
  auto b = pc.size(0), n = pc.size(1);
  float half_size = grid_size / 2.;
  auto valid = torch::ones({ b * n }).to(pc.device().type()).to(torch::kLong);

  auto pc_grid = (pc + half_size) * (voxel_size - 1.);
  auto indices_floor = at::floor(pc_grid);
  auto indices = indices_floor.to(torch::kLong);

  auto batch_indices =
    torch::arange(b).to(pc.device().type()).to(indices.dtype());
  batch_indices = batch_indices.unsqueeze(1).unsqueeze(2);
  batch_indices = batch_indices.expand({ b, n, 1 });
  indices = at::cat({ batch_indices, indices }, 2);
  indices = at::reshape(indices, { -1, 4 });
  auto r = pc_grid - indices_floor;
  std::vector<torch::Tensor> rr{ 1. - r, r };

  if (filter_outlier) {
    valid =
      at::all(at::__and__(at::ge(pc, -half_size), at::le(pc, half_size)), 2);
    valid = valid.flatten();
    indices = indices.index(valid);
  }

  std::vector<int64_t> output_shape{ b, voxel_size, voxel_size, voxel_size };
  at::Tensor output_shape_tensor =
    torch::tensor(output_shape).to(pc.device().type()).to(torch::kLong);
  auto voxel = torch::zeros(output_shape, pc.type()).flatten();
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        auto updates =
          rr[k].slice(2, 0, 1) * rr[j].slice(2, 1, 2) * rr[i].slice(2, 2, 3);
        updates = updates.flatten();
        if (filter_outlier)
          updates = updates.index(valid);

        std::vector<int64_t> shift{ 0, k, j, i };
        at::Tensor indices_shift = torch::tensor(shift)
                                     .to(pc.device().type())
                                     .to(torch::kLong)
                                     .unsqueeze(0);
        auto indices_tmp = indices + indices_shift;

        auto linear_indices = ravel_index(indices_tmp, output_shape_tensor);
        voxel = scatter_add(voxel, 0, linear_indices, updates);
      }
    }
  }
  voxel = voxel.reshape(output_shape);
  voxel = at::clamp(voxel, 0., 1.);
  return voxel;
}
