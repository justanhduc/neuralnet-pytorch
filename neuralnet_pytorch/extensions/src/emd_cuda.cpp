#include "emd_cuda.h"
#include "utils.h"


at::Tensor approx_match_cuda_forward(const at::Tensor xyz1, const at::Tensor xyz2);
at::Tensor match_cost_cuda_forward(const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match);
std::vector<at::Tensor> match_cost_cuda_backward(const at::Tensor grad_cost, const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match);

/* ApproxMatch forward interface
Input:
  xyz1: (B, N1, 3)  # dataset_points
  xyz2: (B, N2, 3)  # query_points
Output:
  match: (B, N2, N1)
*/
at::Tensor approx_match_forward(const at::Tensor xyz1, const at::Tensor xyz2) {
    CHECK_EQ(xyz1.size(0), xyz2.size(0));
    CHECK_EQ(xyz1.size(2), 3);
    CHECK_EQ(xyz2.size(2), 3);
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    return approx_match_cuda_forward(xyz1, xyz2);
}


/* MatchCost forward interface
Input:
  xyz1: (B, N1, 3)  # dataset_points
  xyz2: (B, N2, 3)  # query_points
  match: (B, N2, N1)
Output:
  cost: (B)
*/
at::Tensor match_cost_forward(const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match) {
    CHECK_EQ(xyz1.size(0), xyz2.size(0));
    CHECK_EQ(xyz1.size(2), 3);
    CHECK_EQ(xyz2.size(2), 3);
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    return match_cost_cuda_forward(xyz1, xyz2, match);
}


/* MatchCost backward interface
Input:
  grad_cost: (B)    # gradients on cost
  xyz1: (B, N1, 3)  # dataset_points
  xyz2: (B, N2, 3)  # query_points
  match: (B, N2, N1)
Output:
  grad1: (B, N1, 3)
  grad2: (B, N2, 3)
*/
std::vector<at::Tensor> match_cost_backward(const at::Tensor grad_cost, const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match) {
    CHECK_EQ(xyz1.size(0), xyz2.size(0));
    CHECK_EQ(xyz1.size(2), 3);
    CHECK_EQ(xyz2.size(2), 3);
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    return match_cost_cuda_backward(grad_cost, xyz1, xyz2, match);
}

