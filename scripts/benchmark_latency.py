from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from flowoe_execution.model import CFMPolicy

ROOT = Path(__file__).resolve().parents[1]

def bench(model, ctx, source, steps, runs, warmup):
    times = []
    for _ in range(warmup): model.sample(ctx, source, steps=steps)
    if ctx.is_cuda: torch.cuda.synchronize()
    for _ in range(runs):
        t = time.perf_counter(); model.sample(ctx, source, steps=steps)
        if ctx.is_cuda: torch.cuda.synchronize()
        times.append((time.perf_counter()-t)*1e3)
    return {'mean_ms': float(np.mean(times)), 'p50_ms': float(np.percentile(times,50)), 'p99_ms': float(np.percentile(times,99)), 'device': str(ctx.device)}

def main():
    torch.manual_seed(7); np.random.seed(7)
    ap = argparse.ArgumentParser(); ap.add_argument('--batch',type=int,default=1); ap.add_argument('--steps',type=int,default=16); ap.add_argument('--runs',type=int,default=300); args=ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CFMPolicy(8).to(device).eval(); ctx = torch.randn(args.batch,32,45,device=device)
    result = {'pytorch_fp32': bench(model,ctx,'crypto',args.steps,args.runs,20), 'cuda_available': torch.cuda.is_available(), 'tensorrt_available': False, 'int8_status': 'NOT_RUN'}
    try:
        import tensorrt  # noqa: F401
        result['tensorrt_available'] = True
        result['int8_status'] = 'AVAILABLE_BUT_REQUIRES_GPU_ENGINE_BUILD'
    except Exception:
        pass
    out=ROOT/'results'; out.mkdir(exist_ok=True); (out/'latency.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
