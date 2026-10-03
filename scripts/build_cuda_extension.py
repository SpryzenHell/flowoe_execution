from pathlib import Path
import torch
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parents[1]
if not torch.cuda.is_available(): raise SystemExit('CUDA not available')
ext=load(name='flowoe_fused',sources=[str(ROOT/'src/flowoe_execution/cuda/fused_step.cpp'),str(ROOT/'src/flowoe_execution/cuda/fused_step.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=True)
print('built', ext)
