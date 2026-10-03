from __future__ import annotations
import numpy as np
import pandas as pd

def l2_features(snapshots: pd.DataFrame, levels: int = 10) -> np.ndarray:
    a=snapshots; bid0=a['bid0'].to_numpy(np.float32); ask0=a['ask0'].to_numpy(np.float32); mid=(bid0+ask0)/2
    spread=(ask0-bid0)/np.maximum(mid,1e-8)
    bv=a[[f'bid_size{i}' for i in range(levels)]].to_numpy(np.float32); av=a[[f'ask_size{i}' for i in range(levels)]].to_numpy(np.float32)
    imbalance=(bv.sum(1)-av.sum(1))/np.maximum(bv.sum(1)+av.sum(1),1e-8)
    micro=(ask0*bv[:,0]+bid0*av[:,0])/np.maximum(bv[:,0]+av[:,0],1e-8); micro_dev=(micro-mid)/np.maximum(mid,1e-8)
    ret=np.r_[0.0,np.diff(np.log(np.maximum(mid,1e-12)))]; vol=pd.Series(ret).rolling(32,min_periods=2).std().fillna(0).to_numpy(np.float32)
    db=(a[[f'bid{i}' for i in range(levels)]].to_numpy(np.float32)-mid[:,None])/np.maximum(mid[:,None],1e-8)
    da=(a[[f'ask{i}' for i in range(levels)]].to_numpy(np.float32)-mid[:,None])/np.maximum(mid[:,None],1e-8)
    x=np.concatenate([db,da,np.log1p(bv),np.log1p(av),spread[:,None],micro_dev[:,None],imbalance[:,None],vol[:,None]],1)
    assert x.shape[1]==44
    ofi=np.r_[0.0,np.diff(bv[:,0]-av[:,0])]; ofi=np.tanh(ofi/(np.std(ofi)+1e-6))[:,None]
    return np.concatenate([x,ofi],1).astype(np.float32)

def features_from_fi2010(features: np.ndarray) -> np.ndarray:
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=144: raise ValueError(f'expected (N,144), got {x.shape}')
    return np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
