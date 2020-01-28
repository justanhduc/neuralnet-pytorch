#include <ATen/ATen.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__global__ void
NmDistanceKernel(int b, int n, const scalar_t* __restrict__ xyz, int m,
                 const scalar_t* __restrict__ xyz2,
                 scalar_t* __restrict__ result, scalar_t* __restrict__ result_i)
{
  const int batch = 512;
  __shared__ scalar_t buf[batch * 3];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += batch) {
      int end_k = min(m, k2 + batch) - k2;
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 3 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * 3 + 0];
        scalar_t y1 = xyz[(i * n + j) * 3 + 1];
        scalar_t z1 = xyz[(i * n + j) * 3 + 2];
        int best_i = 0;
        scalar_t best = 0;
        int end_ka = end_k - (end_k & 3);
        if (end_ka == batch) {
          for (int k = 0; k < batch; k += 4) {
            {
              scalar_t x2 = buf[k * 3 + 0] - x1;
              scalar_t y2 = buf[k * 3 + 1] - y1;
              scalar_t z2 = buf[k * 3 + 2] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              scalar_t x2 = buf[k * 3 + 3] - x1;
              scalar_t y2 = buf[k * 3 + 4] - y1;
              scalar_t z2 = buf[k * 3 + 5] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              scalar_t x2 = buf[k * 3 + 6] - x1;
              scalar_t y2 = buf[k * 3 + 7] - y1;
              scalar_t z2 = buf[k * 3 + 8] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              scalar_t x2 = buf[k * 3 + 9] - x1;
              scalar_t y2 = buf[k * 3 + 10] - y1;
              scalar_t z2 = buf[k * 3 + 11] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              scalar_t x2 = buf[k * 3 + 0] - x1;
              scalar_t y2 = buf[k * 3 + 1] - y1;
              scalar_t z2 = buf[k * 3 + 2] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              scalar_t x2 = buf[k * 3 + 3] - x1;
              scalar_t y2 = buf[k * 3 + 4] - y1;
              scalar_t z2 = buf[k * 3 + 5] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              scalar_t x2 = buf[k * 3 + 6] - x1;
              scalar_t y2 = buf[k * 3 + 7] - y1;
              scalar_t z2 = buf[k * 3 + 8] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              scalar_t x2 = buf[k * 3 + 9] - x1;
              scalar_t y2 = buf[k * 3 + 10] - y1;
              scalar_t z2 = buf[k * 3 + 11] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 3 + 0] - x1;
          scalar_t y2 = buf[k * 3 + 1] - y1;
          scalar_t z2 = buf[k * 3 + 2] - z1;
          scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = scalar_t(best_i);
        }
      }
      __syncthreads();
    }
  }
}

std::vector<torch::Tensor>
chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2)
{
  cudaSetDevice((int)xyz1.device().index());
  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1); // num_points point cloud A
  const auto m = xyz2.size(1); // num_points point cloud B

  auto dist1 = torch::zeros({ batch_size, n }, xyz1.type());
  auto dist2 = torch::zeros({ batch_size, m }, xyz1.type());
  auto idx1 = torch::zeros({ batch_size, n }, xyz1.type());
  auto idx2 = torch::zeros({ batch_size, m }, xyz1.type());

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half, xyz1.scalar_type(), "chamfer_cuda_forward", ([&] {
      NmDistanceKernel<scalar_t><<<dim3(32, 16, 1), 512>>>(
        batch_size, n, xyz1.data<scalar_t>(), m, xyz2.data<scalar_t>(),
        dist1.data<scalar_t>(), idx1.data<scalar_t>());
      NmDistanceKernel<scalar_t><<<dim3(32, 16, 1), 512>>>(
        batch_size, m, xyz2.data<scalar_t>(), n, xyz1.data<scalar_t>(),
        dist2.data<scalar_t>(), idx2.data<scalar_t>());
    }));
  THCudaCheck(cudaGetLastError());

  return { dist1, dist2, idx1, idx2 };
}

template <typename scalar_t>
__global__ void
NmDistanceGradKernel(int b, int n, const scalar_t* __restrict__ xyz1, int m,
                     const scalar_t* __restrict__ xyz2,
                     const scalar_t* __restrict__ grad_dist1,
                     const scalar_t* __restrict__ idx,
                     scalar_t* __restrict__ grad_xyz1,
                     scalar_t* __restrict__ grad_xyz2)
{
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      scalar_t x1 = xyz1[(i * n + j) * 3 + 0];
      scalar_t y1 = xyz1[(i * n + j) * 3 + 1];
      scalar_t z1 = xyz1[(i * n + j) * 3 + 2];
      int j2 = (int)idx[i * n + j];
      scalar_t x2 = xyz2[(i * m + j2) * 3 + 0];
      scalar_t y2 = xyz2[(i * m + j2) * 3 + 1];
      scalar_t z2 = xyz2[(i * m + j2) * 3 + 2];
      scalar_t g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 3 + 2]), g * (z1 - z2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 1]), -(g * (y1 - y2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 3 + 2]), -(g * (z1 - z2)));
    }
  }
}

std::vector<torch::Tensor>
chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor graddist1,
                      at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2)
{
  const auto batch_size = xyz1.size(0);
  const auto n = xyz1.size(1); // num_points point cloud A
  const auto m = xyz2.size(1); // num_points point cloud B

  auto gradxyz1 = torch::zeros_like(xyz1);
  auto gradxyz2 = torch::zeros_like(xyz2);

  AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half, xyz1.scalar_type(), "chamfer_backward_cuda", ([&] {
      NmDistanceGradKernel<scalar_t><<<dim3(1, 16, 1), 256>>>(
        batch_size, n, xyz1.data<scalar_t>(), m, xyz2.data<scalar_t>(),
        graddist1.data<scalar_t>(), idx1.data<scalar_t>(),
        gradxyz1.data<scalar_t>(), gradxyz2.data<scalar_t>());
      NmDistanceGradKernel<scalar_t><<<dim3(1, 16, 1), 256>>>(
        batch_size, m, xyz2.data<scalar_t>(), n, xyz1.data<scalar_t>(),
        graddist2.data<scalar_t>(), idx2.data<scalar_t>(),
        gradxyz2.data<scalar_t>(), gradxyz1.data<scalar_t>());
    }));
  THCudaCheck(cudaGetLastError());

  return { gradxyz1, gradxyz2 };
}
