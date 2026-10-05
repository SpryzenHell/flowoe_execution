#include <torch/extension.h>
void fused_euler_cuda(torch::Tensor x, torch::Tensor v, double dt);
void fused_euler(torch::Tensor x, torch::Tensor v, double dt) {
  TORCH_CHECK(x.is_cuda() && v.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.is_contiguous() && v.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(x.numel() == v.numel(), "shape mismatch");
  fused_euler_cuda(x, v, dt);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_euler_step", &fused_euler); }
