from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler
ROOT=Path(__file__).resolve().parents[1]

def stats(x): return {'mean_ms':float(np.mean(x)),'p50_ms':float(np.percentile(x,50)),'p95_ms':float(np.percentile(x,95)),'p99_ms':float(np.percentile(x,99))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--engine',default='results/flowoe_int8.pth'); ap.add_argument('--checkpoint',default='results/cfm_policy_smoke.pt'); ap.add_argument('--runs',type=int,default=500); args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    from torch2trt_dynamic import TRTModule
    state=torch.load(ROOT/args.checkpoint,map_location='cuda',weights_only=True)
    model=CFMPolicy(8).cuda().eval(); model.load_state_dict(state); fp=FixedStepCryptoSampler(model,16).cuda().eval()
    trt_model=TRTModule(); trt_model.load_state_dict(torch.load(ROOT/args.engine,map_location='cuda',weights_only=False))
    context=torch.randn(1,32,45,device='cuda'); x=torch.randn(1,8,device='cuda')
    for _ in range(50): fp(context,x); trt_model(context,x)
    torch.cuda.synchronize(); a=[]
    for _ in range(args.runs):
        t=time.perf_counter(); fp(context,x); torch.cuda.synchronize(); a.append((time.perf_counter()-t)*1e3)
    b=[]
    for _ in range(args.runs):
        t=time.perf_counter(); trt_model(context,x); torch.cuda.synchronize(); b.append((time.perf_counter()-t)*1e3)
    with torch.no_grad(): err=float((fp(context,x)-trt_model(context,x)).abs().max().item())
    out={'pytorch_fixed_fp32':stats(a),'tensorrt_int8':stats(b),'speedup_mean':float(np.mean(a)/np.mean(b)),'max_abs_error':err,
         'gpu':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
         'tensorrt_version':__import__('tensorrt').__version__}
    (ROOT/'results/tensorrt_latency.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
