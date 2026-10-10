from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_fused_euler_uses_current_pytorch_stream():
    ext = load(
        name="flowoe_fused_bench",
        sources=[
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    x = torch.zeros((257, 8), device="cuda", dtype=torch.float32)
    v = torch.linspace(-1, 1, x.numel(), device="cuda", dtype=torch.float32).view_as(x)
    dt = 0.125
    expected = x + dt * v
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ext.fused_euler_step(x, v, dt)
        observed = x.clone()
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
