from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/smoke'; OUT.mkdir(parents=True, exist_ok=True)
r = np.random.default_rng(42); n = 6000; levels = 10
mid = 100 * np.exp(np.cumsum(r.normal(0, 0.00035, n)))
rows = []
for t in range(n):
    spr = 0.02 * (1 + 0.2 * abs(r.normal()))
    row = {'timestamp': pd.Timestamp('2024-01-01', tz='UTC') + pd.Timedelta(milliseconds=t)}
    for i in range(levels):
        row[f'bid{i}'] = mid[t] - spr/2 - i*0.01
        row[f'ask{i}'] = mid[t] + spr/2 + i*0.01
        row[f'bid_size{i}'] = max(0.1, r.lognormal(0, 0.35))
        row[f'ask_size{i}'] = max(0.1, r.lognormal(0, 0.35))
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv(OUT/'crypto_l2.csv', index=False)
trades = pd.DataFrame({'timestamp': df.timestamp, 'price': mid+r.normal(0,0.005,n), 'qty': r.lognormal(-0.1,0.5,n)})
trades.to_csv(OUT/'crypto_trades.csv', index=False)
fi = r.normal(size=(n,144)).astype('float32')
labels = np.column_stack([(r.random(n)>0.33).astype('int16') + (r.random(n)>0.80).astype('int16') for _ in range(5)])
np.savetxt(OUT/'fi2010_smoke.txt', np.column_stack([fi, labels]), fmt='%.6f')
print(f'wrote smoke dataset to {OUT}')
