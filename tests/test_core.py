import numpy as np
import pandas as pd
import torch
from flowoe_execution.features import l2_features
from flowoe_execution.model import CFMPolicy, ProbabilityFlowODEPolicy
from flowoe_execution.execution import ExecutionSimulator


def sample_l2(n=16, levels=10):
    rows=[]
    for i in range(n):
        r={'timestamp':pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(milliseconds=i)}
        for l in range(levels):
            r[f'bid{l}']=100-l*.01; r[f'ask{l}']=100.02+l*.01; r[f'bid_size{l}']=1.; r[f'ask_size{l}']=1.
        rows.append(r)
    return pd.DataFrame(rows)


def test_feature_dim():
    x=l2_features(sample_l2()); assert x.shape==(16,45); assert np.isfinite(x).all()


def test_cfm_loss_and_sample():
    m=CFMPolicy(8); c=torch.randn(4,16,45); y=torch.softmax(torch.randn(4,8),-1)
    assert torch.isfinite(m.cfm_loss(c,y,'crypto')); assert m.sample(c,'crypto',steps=4).shape==(4,8)


def test_probability_flow_ode():
    m=ProbabilityFlowODEPolicy(8); c=torch.randn(2,16,45); z=m.sample(c,'crypto',steps=4)
    assert z.shape==(2,8) and torch.isfinite(z).all()


def test_execution():
    r=ExecutionSimulator(1.0).run(sample_l2(8),np.ones(8)/8)
    assert r.executed_qty>0 and np.isfinite(r.slippage_bps)
