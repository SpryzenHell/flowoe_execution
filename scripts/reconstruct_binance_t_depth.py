from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
LEVELS=10

def rows(path):
    with open(path,newline='',encoding='utf-8') as f: yield from csv.DictReader(f)

def apply(book,row):
    side='ask' if row['side'].lower() in {'a','ask'} else 'bid'
    price=float(row['price']); qty=float(row['qty']); typ=row.get('update_type','set').lower()
    book[side][price] = book[side].get(price,0.0)+qty if typ=='delta' else qty
    if book[side].get(price,0.0)<=0: book[side].pop(price,None)

def top(side,book):
    vals=sorted(((p,q) for p,q in book[side].items() if q>0),reverse=(side=='bid'))[:LEVELS]
    return sorted(vals) if side=='ask' else vals

def emit(writer,ts,book):
    bids,asks=top('bid',book),top('ask',book)
    if len(bids)<LEVELS or len(asks)<LEVELS: return
    out={'timestamp':ts}
    for i in range(LEVELS):
        out[f'bid{i}'],out[f'bid_size{i}']=bids[i]; out[f'ask{i}'],out[f'ask_size{i}']=asks[i]
    writer.writerow(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--snapshot',required=True); ap.add_argument('--updates',required=True); ap.add_argument('--out',required=True); ap.add_argument('--interval-ms',type=int,default=100); a=ap.parse_args()
    book=defaultdict(dict); Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    fields=['timestamp']+[c for i in range(LEVELS) for c in (f'bid{i}',f'bid_size{i}',f'ask{i}',f'ask_size{i}')]
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows(a.snapshot): apply(book,row)
        next_emit=None
        for row in rows(a.updates):
            raw=row.get('timestamp') or row.get('time') or row.get('ts')
            if raw is None: raise ValueError('No timestamp/time/ts in update row')
            ts_ms=int(float(raw)); apply(book,row)
            if next_emit is None: next_emit=ts_ms
            if ts_ms>=next_emit:
                emit(writer,ts_ms/1000.0,book); next_emit=ts_ms+a.interval_ms
    print(a.out)
if __name__=='__main__': main()
