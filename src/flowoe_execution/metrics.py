from __future__ import annotations
import numpy as np

def bootstrap_mean_ci(values,seed=7,n_boot=2000,alpha=0.05):
    x=np.asarray(values,dtype=np.float64)
    if len(x)==0: raise ValueError('values must be non-empty')
    rng=np.random.default_rng(seed); means=x[rng.integers(0,len(x),size=(n_boot,len(x)))].mean(1)
    return float(x.mean()),(float(np.quantile(means,alpha/2)),float(np.quantile(means,1-alpha/2)))
def improvement_bps(baseline_slippage_bps,policy_slippage_bps): return float(baseline_slippage_bps-policy_slippage_bps)
def percentile_ms(values,q=99): return float(np.percentile(np.asarray(values,dtype=np.float64),q))
