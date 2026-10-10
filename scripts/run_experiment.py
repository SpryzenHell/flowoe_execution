from pathlib import Path
import argparse, hashlib, json, platform, time, sys
import numpy as np
import torch
from flowoe_execution.data import load_fi2010, load_l2_csv, load_trades_csv
from flowoe_execution.features import l2_features, features_from_fi2010
from flowoe_execution.model import CFMPolicy
from flowoe_execution.execution import ExecutionSimulator, make_schedule_from_trajectory
from flowoe_execution.metrics import improvement_bps, paired_block_bootstrap_ci, summary_stats
from flowoe_execution.labels import normalize_fi2010_labels

ROOT = Path(__file__).resolve().parents[1]


def fi_training_windows(features, labels, train_ratio, context_len=32, stride=8, max_label_horizon=10):
    """Build chronological FI windows without labels crossing the train split.

    FI-2010 labels encode future price movement up to the 10-step horizon.
    The final observation in every context must have its label horizon fully
    contained within the training partition.
    """
    if features.ndim != 2 or labels.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must be aligned 2-D arrays")
    if not 0 < train_ratio < 1 or context_len < 1 or stride < 1 or max_label_horizon < 1:
        raise ValueError("invalid FI training window parameters")
    split = int(len(features) * train_ratio)
    stop = max(0, split - max_label_horizon - context_len + 1)
    contexts, targets = [], []
    for start in range(0, stop, stride):
        last = start + context_len - 1
        contexts.append(features[start:start + context_len])
        targets.append(torch.as_tensor(labels[last], dtype=torch.long))
        assert last + max_label_horizon < split
    return contexts, targets


def expert(mid_future, horizon, side='buy'):
    future = np.asarray(mid_future[:horizon], dtype=np.float64)
    centered = (future - future.mean()) / (future.std() + 1e-6)
    # Buy more when future prices are lower; sell more when they are higher.
    z = -centered if side == 'buy' else centered
    return make_schedule_from_trajectory(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--fi')
    ap.add_argument('--l2')
    ap.add_argument('--trades')
    ap.add_argument('--fi-max-rows', type=int, default=50000)
    ap.add_argument('--fi-label-encoding', choices=['auto', 'zero_one_two', 'minus1_0_1', 'one_two_three'], default='auto',
                    help='FI-2010 label encoding; auto rejects ambiguous observed subsets')
    ap.add_argument('--l2-max-rows', type=int, default=200000)
    ap.add_argument('--train-ratio', type=float, default=0.60)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                    help='Training/inference device. auto selects CUDA when available.')
    ap.add_argument('--allow-book-vwap', action='store_true', help='Allow the depth-weighted proxy for a real run (not a trade-VWAP evidence run).')
    ap.add_argument('--side', choices=['buy', 'sell'], default='buy')
    ap.add_argument('--quantity', type=float, default=1.0)
    ap.add_argument('--instrument', default='BTCUSDT')
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error('--epochs must be positive')
    if not 0.1 <= args.train_ratio <= 0.9:
        ap.error('--train-ratio must be between 0.1 and 0.9')
    if args.quantity <= 0:
        ap.error('--quantity must be positive')
    if args.fi_max_rows < 1 or args.l2_max_rows < 1:
        ap.error('row limits must be positive')
    torch.manual_seed(7); np.random.seed(7)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda was requested, but CUDA is not available')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(7)

    if args.smoke or (args.fi is None and args.l2 is None):
        import subprocess
        # Use this same environment, so GPU runs do not fall back to the
        # runner's system Python where numpy/pandas may not be installed.
        subprocess.run([sys.executable, str(ROOT / 'scripts/generate_smoke_data.py')], check=True)
        fi_path = ROOT / 'data/smoke/fi2010_smoke.txt'
        l2_path = ROOT / 'data/smoke/crypto_l2.csv'
        trade_path = ROOT / 'data/smoke/crypto_trades.csv'
        dataset_name = 'synthetic smoke'
    else:
        if not (args.fi and args.l2):
            raise SystemExit('--fi and --l2 are required together for a real-data run')
        fi_path, l2_path = Path(args.fi), Path(args.l2)
        if not args.trades and not args.allow_book_vwap:
            raise SystemExit('--trades is required for the real-data evidence run; use --allow-book-vwap only for a proxy run.')
        trade_path = Path(args.trades) if args.trades else None
        dataset_name = 'user-supplied real data'

    l2 = load_l2_csv(l2_path, max_rows=args.l2_max_rows)
    trades = load_trades_csv(trade_path) if trade_path else None
    fi = load_fi2010(fi_path, max_rows=args.fi_max_rows)
    fi_labels, fi_label_encoding = normalize_fi2010_labels(fi.labels, args.fi_label_encoding)
    x_crypto = torch.from_numpy(l2_features(l2.snapshots))
    ntrain = int(len(l2.snapshots) * args.train_ratio)
    ctx_n, horizon, stride = 32, 8, 8
    if len(l2.snapshots) <= ctx_n + horizon + 1:
        raise SystemExit(f'Need more than {ctx_n + horizon + 1} L2 rows; found {len(l2.snapshots)}')
    if not torch.isfinite(x_crypto).all():
        raise SystemExit('L2 feature builder returned non-finite values')
    ntrain = max(ntrain, ctx_n + horizon + 1)
    if ntrain >= len(l2.snapshots) - horizon:
        raise SystemExit('Chronological split leaves no evaluation windows')
    train_ctx, train_y = [], []
    mids = ((l2.snapshots.bid0 + l2.snapshots.ask0) / 2).to_numpy(np.float32)
    for i in range(0, ntrain - ctx_n - horizon, stride):
        train_ctx.append(x_crypto[i:i+ctx_n])
        train_y.append(torch.tensor(expert(mids[i+ctx_n:i+ctx_n+horizon], horizon, args.side), dtype=torch.float32))

    if not len(train_ctx):
        raise SystemExit('Chronological training split produced no crypto training windows')
    fi_x = torch.from_numpy(features_from_fi2010(fi.features))
    if not torch.isfinite(fi_x).all():
        raise SystemExit('FI-2010 feature conversion returned non-finite values')
    fi_ctx, fi_targets = fi_training_windows(
        fi_x, fi_labels, args.train_ratio, context_len=ctx_n, stride=stride,
        max_label_horizon=10,
    )

    torch.set_num_threads(2)
    model = CFMPolicy(horizon=horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train(); t0 = time.perf_counter()
    for ep in range(args.epochs):
        order = np.random.default_rng(ep).permutation(len(train_ctx))
        for j in range(0, len(order), 64):
            b = order[j:j+64]
            c = torch.stack([train_ctx[k] for k in b]).to(device)
            y = torch.stack([train_y[k] for k in b]).to(device)
            loss = model.cfm_loss(c, y, 'crypto')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if fi_ctx:
            idx = np.random.default_rng(ep + 1000).choice(len(fi_ctx), min(32, len(fi_ctx)), replace=False)
            c = torch.stack([fi_ctx[k] for k in idx]).to(device)
            y = torch.stack([fi_targets[k] for k in idx]).to(device)
            loss = model.fi_aux_loss(c, y, label_encoding='zero_one_two')
            opt.zero_grad(); loss.backward(); opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_s = time.perf_counter() - t0

    sim = ExecutionSimulator(quantity=args.quantity, side=args.side); baseline, policy = [], []
    baseline_completion, policy_completion = [], []
    benchmark_name = 'trade_vwap' if trades is not None else 'book_vwap'
    for i in range(ntrain, len(l2.snapshots)-horizon, horizon*3):
        ctx = x_crypto[i-ctx_n:i].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.sample(ctx, 'crypto', steps=24)[0].detach().cpu().numpy()
        r_flow = sim.run(l2.snapshots.iloc[i:i+horizon], make_schedule_from_trajectory(pred), benchmark=benchmark_name, trades=trades, strategy='flowoe')
        r_twap = sim.run(l2.snapshots.iloc[i:i+horizon], np.ones(horizon)/horizon, benchmark=benchmark_name, trades=trades, strategy='twap')
        baseline.append(r_twap.slippage_bps); policy.append(r_flow.slippage_bps)
        baseline_completion.append(r_twap.completion); policy_completion.append(r_flow.completion)

    if not baseline or not policy:
        raise SystemExit('Evaluation split produced no execution episodes; use a longer L2 window')
    twap_stats = summary_stats(baseline)
    flow_stats = summary_stats(policy)
    paired_improvement, paired_ci95 = paired_block_bootstrap_ci(
        baseline, policy, seed=7, n_boot=2000, alpha=0.05, block_size=5
    )

    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    report = {
        'dataset': dataset_name,
        'instrument': args.instrument,
        'side': args.side,
        'quantity': args.quantity,
        'device': str(device),
        'context_length': ctx_n,
        'horizon': horizon,
        'sampling_steps': 24,
        'train_ratio': args.train_ratio,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'fi_path': str(fi_path), 'fi_sha256': file_sha256(fi_path),
        'l2_path': str(l2_path), 'l2_sha256': file_sha256(l2_path),
        'trades_path': str(trade_path) if trade_path else None,
        'trades_sha256': file_sha256(trade_path) if trade_path else None,
        'train_windows': len(train_ctx), 'fi_windows': len(fi_ctx),
        'fi_label_encoding_detected': fi_label_encoding,
        'epochs': args.epochs, 'training_seconds': train_s, 'episodes': len(policy),
        'twap_slippage_bps': twap_stats,
        'flowoe_slippage_bps': flow_stats,
        'twap_completion_mean': float(np.mean(baseline_completion)) if baseline_completion else None,
        'flowoe_completion_mean': float(np.mean(policy_completion)) if policy_completion else None,
        'benchmark': benchmark_name,
        'improvement_vs_twap_bps': improvement_bps(twap_stats['mean'], flow_stats['mean']),
        'paired_improvement_vs_twap_bps': paired_improvement,
        'paired_improvement_ci95_bps': paired_ci95,
        'paired_ci_method': 'paired moving-block bootstrap on consecutive execution windows (block size 5)',
        'note': 'Synthetic smoke data only; not FI-2010 or real crypto L2.' if dataset_name.startswith('synthetic') else ('User-supplied real data with trade VWAP.' if trades is not None else 'User-supplied real data with book-VWAP proxy; not the resume evidence benchmark.')
    }
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / ('cfm_policy_smoke.pt' if dataset_name.startswith('synthetic') else 'cfm_policy.pt'))
    (out / ('smoke_experiment.json' if dataset_name.startswith('synthetic') else 'real_experiment.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
