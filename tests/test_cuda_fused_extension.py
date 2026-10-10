from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fused_ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA extension tests require a visible CUDA device")
    from torch.utils.cpp_extension import load

    return load(
        name="flowoe_fused_bench",
        sources=[
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def test_fused_euler_matches_torch_and_mutates_x(fused_ext):
    torch.manual_seed(9)
    x = torch.randn(16, 8, device="cuda", dtype=torch.float32)
    v = torch.randn_like(x)
    expected = x + 0.125 * v
    result = fused_ext.fused_euler_step(x, v, 0.125)
    torch.cuda.synchronize()
    assert result is None
    torch.testing.assert_close(x, expected, rtol=1e-6, atol=1e-6)


def test_fused_euler_rejects_equal_numel_but_different_shapes(fused_ext):
    x = torch.zeros(2, 4, device="cuda")
    v = torch.zeros(4, 2, device="cuda")
    with pytest.raises(RuntimeError, match="identical shapes"):
        fused_ext.fused_euler_step(x, v, 0.1)


def test_fused_euler_rejects_dtype_mismatch(fused_ext):
    x = torch.zeros(8, device="cuda", dtype=torch.float32)
    v = torch.zeros(8, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="same dtype"):
        fused_ext.fused_euler_step(x, v, 0.1)


def test_fused_euler_rejects_noncontiguous_inputs(fused_ext):
    x = torch.zeros(8, 4, device="cuda").t()
    v = torch.ones(8, 4, device="cuda").t()
    assert not x.is_contiguous() and not v.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        fused_ext.fused_euler_step(x, v, 0.1)


@pytest.mark.parametrize("dt", [math.inf, -math.inf, math.nan])
def test_fused_euler_rejects_nonfinite_step(fused_ext, dt):
    x = torch.zeros(8, device="cuda")
    v = torch.ones_like(x)
    with pytest.raises(RuntimeError, match="finite"):
        fused_ext.fused_euler_step(x, v, dt)


def test_fused_euler_accepts_empty_tensors_without_launch(fused_ext):
    x = torch.empty((0, 8), device="cuda")
    v = torch.empty_like(x)
    result = fused_ext.fused_euler_step(x, v, 0.1)
    assert result is None
    assert x.numel() == 0


def test_fused_euler_rejects_cpu_tensors(fused_ext):
    x = torch.zeros(8)
    v = torch.ones_like(x)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        fused_ext.fused_euler_step(x, v, 0.1)


def test_fused_euler_rejects_different_cuda_devices_when_available(fused_ext):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible CUDA devices")
    x = torch.zeros(8, device="cuda:0")
    v = torch.ones(8, device="cuda:1")
    with pytest.raises(RuntimeError, match="same device"):
        fused_ext.fused_euler_step(x, v, 0.1)
