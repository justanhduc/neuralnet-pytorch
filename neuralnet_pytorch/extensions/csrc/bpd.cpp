#include <vector>

#include "bpd.h"

torch::Tensor
batch_pairwise_distance_forward(torch::Tensor x, torch::Tensor y)
{
  std::vector<int64_t> p{ 0, 2, 1 };
  auto xx = at::sum(at::pow(x, 2), 2);
  auto yy = at::sum(at::pow(y, 2), 2);
  auto xy = at::bmm(x, y.permute(p));

  auto rx = xx.unsqueeze(1).expand_as(xy.permute(p));
  auto ry = yy.unsqueeze(1).expand_as(xy);
  auto P = rx.permute(p) + ry - 2. * xy;
  return P;
}
