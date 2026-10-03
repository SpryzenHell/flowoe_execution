from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

@dataclass
class FI2010Data:
    features: np.ndarray
    labels: np.ndarray

@dataclass
class L2Data:
    snapshots: pd.DataFrame
    trades: pd.DataFrame|None=None

def load_fi2010(path,max_rows=None):
    kw={'dtype':np.float32};
    if max_rows is not None: kw['max_rows']=int(max_rows)
    arr=np.loadtxt(path,**kw)
    if arr.ndim!=2 or arr.shape[1]<149: raise ValueError(f'Expected >=149 columns, got {arr.shape}')
    return FI2010Data(arr[:,:144],arr[:,144:149].astype(np.int16))

def _book(df,levels=10):
    if 'timestamp' not in df.columns:
        if {'tsec','tnsec'}.issubset(df.columns): df['timestamp']=df.tsec.astype('int64')+df.tnsec.astype('int64')/1e9
        elif 'ts' in df.columns: df['timestamp']=df['ts']
        else: raise ValueError('L2 file must contain timestamp or tsec/tnsec')
    missing=[c for i in range(levels) for c in (f'bid{i}',f'bid_size{i}',f'ask{i}',f'ask_size{i}') if c not in df.columns]
    if missing: raise ValueError(f'Missing L2 columns: {missing[:8]}')
    cols=['timestamp']+[c for i in range(levels) for c in (f'bid{i}',f'bid_size{i}',f'ask{i}',f'ask_size{i}')]; out=df[cols].copy()
    if np.issubdtype(out.timestamp.dtype,np.number): out['timestamp']=pd.to_datetime(out.timestamp,unit='s',errors='coerce')
    else: out['timestamp']=pd.to_datetime(out.timestamp,utc=True,errors='coerce',format='mixed')
    return out.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

def load_l2_csv(path,levels=10,max_rows=None): return L2Data(_book(pd.read_csv(path,nrows=max_rows),levels))

def load_l2_jsonl(path,levels=10):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        o=json.loads(line); r={'timestamp':o.get('timestamp',o.get('ts'))}
        for i in range(levels): r[f'bid{i}'],r[f'bid_size{i}']=o['bids'][i]; r[f'ask{i}'],r[f'ask_size{i}']=o['asks'][i]
        rows.append(r)
    return L2Data(_book(pd.DataFrame(rows),levels))

def load_trades_csv(path):
    df=pd.read_csv(path)
    if 'timestamp' not in df.columns:
        if {'tsec','tnsec'}.issubset(df.columns): df['timestamp']=df.tsec.astype('int64')+df.tnsec.astype('int64')/1e9
        elif 'ts' in df.columns: df['timestamp']=df.ts
        else: raise ValueError('trade file must contain timestamp')
    qty='qty' if 'qty' in df.columns else 'quantity' if 'quantity' in df.columns else None
    if 'price' not in df.columns or qty is None: raise ValueError('trade file must contain price and qty/quantity')
    if np.issubdtype(df.timestamp.dtype,np.number): df['timestamp']=pd.to_datetime(df.timestamp,unit='s',errors='coerce')
    else: df['timestamp']=pd.to_datetime(df.timestamp,utc=True,errors='coerce',format='mixed')
    return df.dropna(subset=['timestamp','price',qty]).rename(columns={qty:'qty'})[['timestamp','price','qty']].sort_values('timestamp').reset_index(drop=True)
