"""Build the optional fixed-step INT8 TensorRT engine on a CUDA host."""
from pathlib import Path
import argparse, json
import torch
from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='results/cfm_policy_smoke.pt'); ap.add_argument('--steps',type=int,default=16); ap.add_argument('--output',default='results/flowoe_int8.pth'); args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA not available: refusing to claim TensorRT benchmark')
    try:
        import tensorrt as trt
        from torch2trt_dynamic import module2trt, BuildEngineConfig, SequenceDataset
    except Exception as e:
        raise SystemExit(f'TensorRT/torch2trt_dynamic unavailable: {e}')
    policy=CFMPolicy(8).cuda().eval(); policy.load_state_dict(torch.load(ROOT/args.checkpoint,map_location='cuda',weights_only=True))
    wrapper=FixedStepCryptoSampler(policy,args.steps).cuda().eval()
    context=torch.randn(1,32,45,device='cuda'); x=torch.randn(1,8,device='cuda')
    calib=[{'context':torch.randn_like(context),'x':torch.randn_like(x)} for _ in range(32)]
    cfg=BuildEngineConfig(
        shape_ranges={'context':{'min':context.shape,'opt':context.shape,'max':(4,32,45)},'x':{'min':x.shape,'opt':x.shape,'max':(4,8)}},
        int8=True,int8_calib_dataset=SequenceDataset(calib),int8_batch_size=1,int8_cache_file=str(ROOT/'results/flowoe_int8.cache'))
    engine=module2trt(wrapper,args=[context,x],config=cfg)
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True); torch.save(engine.state_dict(),out)
    (ROOT/'results/tensorrt_build.json').write_text(json.dumps({'tensorrt_version':trt.__version__,'steps':args.steps,'engine_path':str(out),'precision':'INT8','status':'built'},indent=2))
    print(out)

if __name__=='__main__': main()
