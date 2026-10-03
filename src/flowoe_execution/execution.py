from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def softmax(x,temp=1.0):
    z=(np.asarray(x,dtype=np.float64)-np.max(x))/max(temp,1e-6); e=np.exp(z); return e/np.maximum(e.sum(),1e-12)
def make_schedule_from_trajectory(x): return softmax(x)
def market_vwap(trades,start,end):
    t=trades[(trades.timestamp>=start)&(trades.timestamp<=end)]
    if t.empty or float(t.qty.sum())<=0: raise ValueError('No trades in requested execution window')
    return float((t.price*t.qty).sum()/t.qty.sum())
def book_vwap(snapshots,start,end):
    s=snapshots[(snapshots.timestamp>=start)&(snapshots.timestamp<=end)]
    if s.empty: raise ValueError('No L2 snapshots in execution window')
    mid=(s.bid0+s.ask0)/2; w=s.bid_size0+s.ask_size0; return float((mid*w).sum()/max(w.sum(),1e-12))
@dataclass
class ExecutionResult:
    strategy:str; executed_qty:float; avg_price:float; benchmark_vwap:float; slippage_bps:float; completion:float
class ExecutionSimulator:
    def __init__(self,quantity=1.0): self.quantity=quantity
    def run(self,snapshots,fractions,benchmark='book_vwap',trades=None,strategy='flowoe'):
        if len(snapshots)<len(fractions): raise ValueError('Not enough snapshots for requested horizon')
        f=np.asarray(fractions,dtype=np.float64); f=f/max(f.sum(),1e-12); target=self.quantity*f; remaining=self.quantity; cost=done=0.0
        for i,q in enumerate(target):
            row=snapshots.iloc[i]; take=min(float(q),remaining,max(float(row.bid_size0+row.ask_size0),1e-9)); cost+=take*float(row.ask0); done+=take; remaining-=take
            if remaining<=1e-12: break
        avg=cost/max(done,1e-12); start=snapshots.iloc[0].timestamp; end=snapshots.iloc[min(len(f)-1,len(snapshots)-1)].timestamp
        vwap=market_vwap(trades,start,end) if benchmark=='trade_vwap' else book_vwap(snapshots,start,end)
        slip=(avg-vwap)/max(vwap,1e-12)*1e4
        return ExecutionResult(strategy,done,avg,vwap,float(slip),done/max(self.quantity,1e-12))
