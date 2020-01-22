#include "bpd.h"
#include "chamfer_cuda.h"
#include "emd_cuda.h"
#include "pc2vox.h"
#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("chamfer_forward", &chamfer_forward, "chamfer forward (CUDA)");
  m.def("chamfer_backward", &chamfer_backward, "chamfer backward (CUDA)");
  m.def("approx_match_forward", &approx_match_forward,
        "ApproxMatch forward (CUDA)");
  m.def("match_cost_forward", &match_cost_forward, "MatchCost forward (CUDA)");
  m.def("match_cost_backward", &match_cost_backward,
        "MatchCost backward (CUDA)");
  m.def("bpd_forward", &batch_pairwise_distance_forward,
        "batch pairwise distance forward");
  m.def("pc2vox_forward", &pointcloud_to_voxel_forward,
        "pointcloud to voxel forward");
  m.def("ravel_index_forward", &ravel_index, "ravel index forward");
}
