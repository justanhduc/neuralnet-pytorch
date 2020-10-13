#include <vector>

#include "bpd.h"

torch::Tensor
batch_pairwise_distance_forward(torch::Tensor x, torch::Tensor y) {
    auto xx = at::sum(at::pow(x, 2), -1);
    auto yy = at::sum(at::pow(y, 2), -1);
    auto xy = at::matmul(x, at::transpose(y, -1, -2));

    auto rx = xx.unsqueeze(-2).expand_as(at::transpose(xy, -1, -2));
    auto ry = yy.unsqueeze(-2).expand_as(xy);
    auto P = at::transpose(rx, -1, -2) + ry - 2. * xy;
    return P;
}
