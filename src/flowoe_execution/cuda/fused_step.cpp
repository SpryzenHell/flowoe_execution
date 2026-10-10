#include <torch/extension.h>
#include <cmath>

void fused_euler_cuda(torch::Tensor x, torch::Tensor v, double dt);

void fused_euler(torch::Tensor x, torch::Tensor v, double dt) {
  TORCH_CHECK(x.is_cuda() && v.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.device() == v.device(), "tensors must be on the same device");
  TORCH_CHECK(x.sizes() == v.sizes(), "x and v must have identical shapes");
  TORCH_CHECK(x.is_contiguous() && v.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(x.scalar_type() == v.scalar_type(), "x and v must have the same dtype");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
              "only float32 and float64 tensors are supported");
  TORCH_CHECK(std::isfinite(dt), "dt must be finite");
  if (x.numel() == 0) return;
  fused_euler_cuda(x, v, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_euler_step", &fused_euler); }
