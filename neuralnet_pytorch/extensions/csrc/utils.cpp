#include <torch/torch.h>

#include "utils.h"

torch::Tensor
ravel_index(at::Tensor indices, at::Tensor shape)
{
  torch::Tensor linear = torch::zeros({ indices.size(0), 1 })
                           .to(indices.device().type())
                           .to(torch::kLong);
  for (int i = 0; i < indices.size(1); ++i)
    linear = linear +
             indices.slice(1, i, i + 1) *
               at::prod(shape.slice(0, i + 1, shape.size(0)));

  return linear.flatten();
}
