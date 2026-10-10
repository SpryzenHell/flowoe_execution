from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.fixture(scope="module")
def fused_extension():
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


def test_fused_euler_uses_current_pytorch_stream(fused_extension):
    x = torch.zeros((257, 8), device="cuda", dtype=torch.float32)
    v = torch.linspace(-1, 1, x.numel(), device="cuda", dtype=torch.float32).view_as(x)
    dt = 0.125
    expected = x + dt * v
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_extension.fused_euler_step(x, v, dt)
        observed = x.clone()
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", ["cpu", "shape", "noncontiguous", "dtype", "nonfinite_dt"])
def test_fused_euler_rejects_invalid_inputs(fused_extension, case):
    x = torch.zeros((2, 3), device="cuda", dtype=torch.float32)
    v = torch.ones_like(x)
    dt = 0.25
    if case == "cpu":
        x, v = torch.zeros((2, 3)), torch.ones((2, 3))
    elif case == "shape":
        v = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    elif case == "noncontiguous":
        x = torch.zeros((3, 2), device="cuda", dtype=torch.float32).t()
    elif case == "dtype":
        x = x.half()
        v = v.half()
    elif case == "nonfinite_dt":
        dt = float("inf")
    with pytest.raises(RuntimeError):
        fused_extension.fused_euler_step(x, v, dt)
