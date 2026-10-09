#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_euler_kernel(scalar_t* x, const scalar_t* v, double dt, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += static_cast<scalar_t>(dt) * v[i];
}

void fused_euler_cuda(torch::Tensor x, torch::Tensor v, double dt) {
  const int threads = 256;
  const int blocks = (x.numel() + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_euler_cuda", ([&] {
    fused_euler_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), dt, x.numel());
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
