from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from flowoe_execution.execution import ExecutionSimulator, make_schedule_from_trajectory
from flowoe_execution.features import l2_features
from flowoe_execution.model import CFMPolicy, ProbabilityFlowODEPolicy

SCENARIOS = ["trend", "volatile", "mean_revert", "jump", "wide_spread", "thin_book", "imbalanced"]
SEEDS = [7, 17, 27, 37, 47]


def bootstrap_mean_ci(values, seed=7, n_boot=2000):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def make_market(scenario, n, seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="ms", tz="UTC")
    logmid = np.empty(n)
    logmid[0] = math.log(100.0)
    for i in range(1, n):
        if scenario == "trend":
            step = 0.00012 + rng.normal(0, 0.00022)
        elif scenario == "volatile":
            step = rng.normal(0, 0.0010)
        elif scenario == "mean_revert":
            step = -0.08 * (logmid[i - 1] - math.log(100.0)) + rng.normal(0, 0.00035)
        elif scenario == "jump":
            step = rng.normal(0, 0.00028) + (rng.normal(0, 0.006) if rng.random() < 0.01 else 0.0)
        elif scenario == "wide_spread":
            step = rng.normal(0, 0.00028)
        elif scenario == "thin_book":
            step = rng.normal(0, 0.00032)
        else:
            signal = math.tanh(8.0 * math.sin(i / 35.0) + 0.8 * rng.normal())
            step = rng.normal(0, 0.00025) + 0.00005 * signal
        logmid[i] = logmid[i - 1] + step
    mid = np.exp(logmid)

    rows = []
    for i, t in enumerate(ts):
        if scenario == "wide_spread":
            spread = 0.0009 * mid[i] * (1 + 0.2 * abs(rng.normal()))
        elif scenario == "volatile":
            spread = 0.00035 * mid[i] * (1 + 0.4 * abs(rng.normal()))
        else:
            spread = 0.00022 * mid[i] * (1 + 0.25 * abs(rng.normal()))
        size_scale = 0.45 if scenario == "thin_book" else 1.0
        if scenario == "imbalanced":
            signal = math.tanh(8.0 * math.sin(i / 35.0) + 0.8 * rng.normal())
            bid_scale, ask_scale = ((1.35, 0.70) if signal > 0 else (0.70, 1.35))
        else:
            bid_scale = ask_scale = 1.0
        row = {"timestamp": t}
        for level in range(10):
            depth = (level + 1) * 0.00005 * mid[i]
            row[f"bid{level}"] = mid[i] - spread / 2 - depth
            row[f"ask{level}"] = mid[i] + spread / 2 + depth
            row[f"bid_size{level}"] = max(0.03, size_scale * bid_scale * rng.lognormal(0, 0.32) / (1 + 0.08 * level))
            row[f"ask_size{level}"] = max(0.03, size_scale * ask_scale * rng.lognormal(0, 0.32) / (1 + 0.08 * level))
        rows.append(row)
    book = pd.DataFrame(rows)
    trades = pd.DataFrame({
        "timestamp": ts,
        "price": mid * (1 + rng.normal(0, 5e-5, n)),
        "qty": rng.lognormal(-0.25, 0.45, n),
    })
    return book, trades


def expert(mid_future):
    z = -(mid_future - mid_future.min()) / (mid_future.std() + 1e-6)
    return make_schedule_from_trajectory(z)


def build_windows(book, context=32):
    x = torch.from_numpy(l2_features(book))
    mids = ((book.bid0 + book.ask0) / 2).to_numpy(np.float32)
    ntrain = int(len(book) * 0.60)
    contexts, targets = [], []
    for i in range(0, ntrain - context - 8, 8):
        contexts.append(x[i:i + context])
        targets.append(torch.tensor(expert(mids[i + context:i + context + 8]), dtype=torch.float32))
    return x, contexts, targets, ntrain


def train_cfm(book, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, contexts, targets, ntrain = build_windows(book)
    model = CFMPolicy(8)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    torch.set_num_threads(2)
    losses, aux_history = [], []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(contexts))
        total = 0.0
        batches = 0
        model.train()
        for j in range(0, len(order), 64):
            batch = order[j:j + 64]
            c = torch.stack([contexts[k] for k in batch])
            y = torch.stack([targets[k] for k in batch])
            loss = model.cfm_loss(c, y, "crypto")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            batches += 1
        fi_x = torch.randn(96, 16, 144)
        fi_y = torch.randint(0, 3, (96, 5))
        aux_loss = model.fi_aux_loss(fi_x, fi_y)
        opt.zero_grad()
        aux_loss.backward()
        opt.step()
        losses.append(total / max(batches, 1))
        aux_history.append(float(aux_loss.detach()))
    return model, x, ntrain, losses, aux_history


def train_pf(book, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, contexts, targets, ntrain = build_windows(book)
    model = ProbabilityFlowODEPolicy(8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    torch.set_num_threads(2)
    losses = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(contexts))
        total = 0.0
        batches = 0
        model.train()
        for j in range(0, len(order), 64):
            batch = order[j:j + 64]
            c = torch.stack([contexts[k] for k in batch])
            y = torch.stack([targets[k] for k in batch])
            loss = model.score_loss(c, y, "crypto")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / max(batches, 1))
    return model, x, ntrain, losses


def evaluate(model, x, book, trades, ntrain, steps=16, quantity=1.0):
    sim = ExecutionSimulator(quantity=quantity)
    rows, schedules = [], []
    for i in range(ntrain, len(book) - 8, 24):
        context = x[i - 32:i].unsqueeze(0)
        torch.manual_seed(100000 + i + steps)
        with torch.no_grad():
            pred = model.sample(context, "crypto", steps=steps)[0].numpy()
        schedule = make_schedule_from_trajectory(pred)
        flow = sim.run(book.iloc[i:i + 8], schedule, benchmark="trade_vwap", trades=trades, strategy="flowoe")
        twap = sim.run(book.iloc[i:i + 8], np.ones(8) / 8, benchmark="trade_vwap", trades=trades, strategy="twap")
        schedules.append(schedule)
        rows.append({
            "flow": flow.slippage_bps,
            "twap": twap.slippage_bps,
            "improvement": twap.slippage_bps - flow.slippage_bps,
            "completion": flow.completion,
            "levels": flow.levels_consumed,
            "entropy": -float(np.sum(schedule * np.log(np.maximum(schedule, 1e-12)))),
        })
    return pd.DataFrame(rows), np.stack(schedules)


def evaluate_family(book, trades, family, quantity=1.0):
    sim = ExecutionSimulator(quantity=quantity)
    start, horizon = int(len(book) * 0.60), 8
    rows = []
    for i in range(start, len(book) - horizon, 24):
        if family == "twap":
            f = np.ones(horizon) / horizon
        elif family == "front":
            f = np.exp(-np.linspace(0, 2, horizon)); f /= f.sum()
        elif family == "back":
            f = np.exp(np.linspace(-2, 0, horizon)); f /= f.sum()
        elif family == "liquidity":
            z = book.bid_size0.iloc[i:i + horizon].to_numpy() + book.ask_size0.iloc[i:i + horizon].to_numpy()
            f = z / z.sum()
        else:
            z = 1.0 / np.maximum(book.ask0.iloc[i:i + horizon].to_numpy() - book.bid0.iloc[i:i + horizon].to_numpy(), 1e-9)
            f = z / z.sum()
        r = sim.run(book.iloc[i:i + horizon], f, benchmark="trade_vwap", trades=trades, strategy=family)
        rows.append((r.slippage_bps, r.completion, r.levels_consumed))
    return pd.DataFrame(rows, columns=["slippage_bps", "completion", "levels"])


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Run the FlowOE synthetic analysis and sensitivity suite.")
    ap.add_argument("--output", default="docs/results/analysis")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    scenarios = SCENARIOS[:4] if args.quick else SCENARIOS
    seeds = SEEDS[:2] if args.quick else SEEDS
    n = min(args.n, 1600) if args.quick else args.n
    epochs = min(args.epochs, 3) if args.quick else args.epochs

    out = Path(args.output)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    markets = {s: make_market(s, n, 100 + i) for i, s in enumerate(scenarios)}

    regime_rows = []
    for s, (book, trades) in markets.items():
        mid = (book.bid0 + book.ask0) / 2
        spread = (book.ask0 - book.bid0) / mid * 1e4
        bid_depth = book[[f"bid_size{i}" for i in range(10)]].sum(axis=1)
        ask_depth = book[[f"ask_size{i}" for i in range(10)]].sum(axis=1)
        imbalance = (bid_depth - ask_depth) / np.maximum(bid_depth + ask_depth, 1e-9)
        ret = np.diff(np.log(mid), prepend=np.log(mid.iloc[0]))
        regime_rows.append({
            "scenario": s, "rows": len(book),
            "median_spread_bps": float(spread.median()),
            "p95_spread_bps": float(spread.quantile(.95)),
            "imbalance_mean": float(imbalance.mean()),
            "imbalance_std": float(imbalance.std()),
            "one_step_vol_bps": float(ret.std() * 1e4),
            "median_bid_depth": float(bid_depth.median()),
            "median_ask_depth": float(ask_depth.median()),
            "trade_rows": len(trades),
        })
    regime = pd.DataFrame(regime_rows)
    regime.to_csv(out / "regime_statistics.csv", index=False)

    robustness = []
    progress = None
    for scenario in ["trend", "volatile", "mean_revert", "thin_book"]:
        if scenario not in markets:
            continue
        book, trades = markets[scenario]
        for seed in seeds:
            model, x, ntrain, loss, aux = train_cfm(book, seed, epochs)
            ev, schedules = evaluate(model, x, book, trades, ntrain, steps=16)
            robustness.append({
                "scenario": scenario, "seed": seed,
                "mean_twap_bps": ev.twap.mean(),
                "mean_flowoe_bps": ev.flow.mean(),
                "mean_improvement_bps": ev.improvement.mean(),
                "median_improvement_bps": ev.improvement.median(),
                "p95_flowoe_bps": ev.flow.quantile(.95),
                "completion": ev.completion.mean(),
                "mean_levels": ev.levels.mean(),
                "mean_schedule_entropy": ev.entropy.mean(),
            })
            if scenario == "trend" and seed == seeds[0]:
                progress = (loss, aux, schedules.mean(axis=0))
    robustness = pd.DataFrame(robustness)
    robustness.to_csv(out / "seed_robustness.csv", index=False)

    robust_summary = robustness.groupby("scenario").agg(
        mean_improvement_bps=("mean_improvement_bps", "mean"),
        min_improvement_bps=("mean_improvement_bps", "min"),
        max_improvement_bps=("mean_improvement_bps", "max"),
        positive_seed_rate=("mean_improvement_bps", lambda x: float((x > 0).mean())),
        completion=("completion", "mean"),
    ).reset_index()
    robust_summary.to_csv(out / "seed_robustness_summary.csv", index=False)

    family_rows = []
    for s, (book, trades) in markets.items():
        for family in ["twap", "front", "back", "liquidity", "spread_aware"]:
            ev = evaluate_family(book, trades, family)
            family_rows.append({
                "scenario": s, "strategy": family,
                "mean_slippage_bps": ev.slippage_bps.mean(),
                "median_slippage_bps": ev.slippage_bps.median(),
                "p95_slippage_bps": ev.slippage_bps.quantile(.95),
                "completion": ev.completion.mean(),
                "mean_levels": ev.levels.mean(),
            })
    families = pd.DataFrame(family_rows)
    families.to_csv(out / "schedule_family_comparison.csv", index=False)

    book, trades = markets["trend"]
    model, x, ntrain, loss, aux = train_cfm(book, seeds[0], epochs + 2)
    step_rows = []
    for step_count in [4, 8, 12, 16, 24, 32]:
        ev, _ = evaluate(model, x, book, trades, ntrain, steps=step_count)
        step_rows.append({
            "steps": step_count,
            "mean_flowoe_bps": ev.flow.mean(),
            "mean_twap_bps": ev.twap.mean(),
            "improvement_bps": ev.improvement.mean(),
            "median_flowoe_bps": ev.flow.median(),
            "p95_flowoe_bps": ev.flow.quantile(.95),
        })
    steps = pd.DataFrame(step_rows)
    steps.to_csv(out / "sampling_steps_sensitivity.csv", index=False)

    book, trades = markets["thin_book"]
    model, x, ntrain, _, _ = train_cfm(book, seeds[0], epochs + 2)
    qty_rows = []
    for q in [0.25, 0.5, 1.0, 2.0, 4.0]:
        ev, _ = evaluate(model, x, book, trades, ntrain, steps=16, quantity=q)
        qty_rows.append({
            "quantity": q,
            "mean_flowoe_bps": ev.flow.mean(),
            "mean_twap_bps": ev.twap.mean(),
            "improvement_bps": ev.improvement.mean(),
            "completion": ev.completion.mean(),
            "mean_levels": ev.levels.mean(),
            "partial_fill_rate": float((ev.completion < .999999).mean()),
        })
    quantity = pd.DataFrame(qty_rows)
    quantity.to_csv(out / "quantity_sensitivity.csv", index=False)

    volatile_book, volatile_trades = markets["volatile"]
    pf, px, pntrain, pfloss = train_pf(volatile_book, seeds[0], epochs)
    cfm, cx, cntrain, _, _ = train_cfm(volatile_book, seeds[0], epochs)
    pf_ev, _ = evaluate(pf, px, volatile_book, volatile_trades, pntrain, steps=24)
    cfm_ev, _ = evaluate(cfm, cx, volatile_book, volatile_trades, cntrain, steps=24)
    comparison = pd.DataFrame([
        {
            "model": "CFM",
            "mean_slippage_bps": cfm_ev.flow.mean(),
            "median_slippage_bps": cfm_ev.flow.median(),
            "p95_slippage_bps": cfm_ev.flow.quantile(.95),
            "improvement_bps": cfm_ev.improvement.mean(),
            "completion": cfm_ev.completion.mean(),
        },
        {
            "model": "PF-ODE",
            "mean_slippage_bps": pf_ev.flow.mean(),
            "median_slippage_bps": pf_ev.flow.median(),
            "p95_slippage_bps": pf_ev.flow.quantile(.95),
            "improvement_bps": pf_ev.improvement.mean(),
            "completion": pf_ev.completion.mean(),
        },
    ])
    comparison.to_csv(out / "cfm_vs_pf_ode.csv", index=False)

    fig = plt.figure(figsize=(9, 5))
    for s, (b, _) in markets.items():
        mid = ((b.bid0 + b.ask0) / 2).to_numpy()
        plt.plot(np.arange(len(mid)) / 1000, mid / mid[0] - 1, label=s)
    plt.xlabel("Time (s)"); plt.ylabel("Mid-price change"); plt.title("Synthetic market regimes"); plt.legend(fontsize=9, ncol=2)
    save(fig, figdir / "mid_price_regimes.svg")

    fig = plt.figure(figsize=(8, 5)); plt.bar(regime.scenario, regime.median_spread_bps)
    plt.ylabel("Median spread (bps)"); plt.title("Spread by synthetic regime"); plt.xticks(rotation=30, ha="right")
    save(fig, figdir / "spread_by_regime.svg")

    pivot = families.pivot(index="scenario", columns="strategy", values="mean_slippage_bps")
    fig = plt.figure(figsize=(10, 5)); idx = np.arange(len(pivot)); width = .15
    for j, col in enumerate(pivot.columns): plt.bar(idx + (j - 2) * width, pivot[col], width, label=col)
    plt.xticks(idx, pivot.index, rotation=30, ha="right"); plt.ylabel("Mean slippage (bps)"); plt.title("Schedule families across regimes"); plt.legend(fontsize=8, ncol=3)
    save(fig, figdir / "schedule_families.svg")

    fig = plt.figure(figsize=(9, 5))
    for s in robustness.scenario.unique():
        z = robustness[robustness.scenario == s]; plt.plot(z.seed, z.mean_improvement_bps, "o-", label=s)
    plt.axhline(0, linewidth=1); plt.xlabel("Seed"); plt.ylabel("Mean improvement vs TWAP (bps)"); plt.title("Seed robustness"); plt.legend()
    save(fig, figdir / "seed_robustness.svg")

    fig = plt.figure(figsize=(8, 5)); plt.plot(steps.steps, steps.mean_flowoe_bps, "o-"); plt.plot(steps.steps, steps.mean_twap_bps, "o--")
    plt.xlabel("Sampling steps"); plt.ylabel("Mean slippage (bps)"); plt.title("Sampling-step sensitivity"); plt.legend(["FlowOE", "TWAP"])
    save(fig, figdir / "sampling_step_sensitivity.svg")

    fig = plt.figure(figsize=(8, 5)); plt.plot(quantity.quantity, quantity.mean_flowoe_bps, "o-"); plt.plot(quantity.quantity, quantity.mean_twap_bps, "o--")
    plt.xlabel("Order quantity"); plt.ylabel("Mean slippage (bps)"); plt.title("Order-quantity sensitivity"); plt.legend(["FlowOE", "TWAP"])
    save(fig, figdir / "quantity_sensitivity.svg")

    fig = plt.figure(figsize=(8, 5)); plt.bar(comparison.model, comparison.improvement_bps)
    plt.ylabel("Mean improvement vs TWAP (bps)"); plt.title("CFM vs probability-flow ODE")
    save(fig, figdir / "cfm_vs_pf_ode.svg")

    if progress:
        fig = plt.figure(figsize=(8, 5)); plt.plot(np.arange(1, len(progress[0]) + 1), progress[0], "o-")
        plt.xlabel("Epoch"); plt.ylabel("CFM loss"); plt.title("Training progress"); save(fig, figdir / "training_progress.svg")
        fig = plt.figure(figsize=(8, 5)); plt.plot(np.arange(1, 9), progress[2], "o-")
        plt.xlabel("Execution interval"); plt.ylabel("Mean schedule fraction"); plt.title("Mean FlowOE schedule"); save(fig, figdir / "schedule_profile.svg")

    fig = plt.figure(figsize=(8, 5)); plt.bar(robust_summary.scenario, robust_summary.positive_seed_rate)
    plt.ylim(0, 1.05); plt.ylabel("Positive-seed rate"); plt.title("Fraction of seeds improving on TWAP"); plt.xticks(rotation=30, ha="right")
    save(fig, figdir / "positive_seed_rate.svg")

    fig = plt.figure(figsize=(9, 5)); plt.scatter(regime.median_spread_bps, regime.one_step_vol_bps)
    for _, row in regime.iterrows():
        plt.annotate(row.scenario, (row.median_spread_bps, row.one_step_vol_bps), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Median spread (bps)"); plt.ylabel("One-step return volatility (bps)"); plt.title("Synthetic regime map")
    save(fig, figdir / "regime_map.svg")

    trend = robustness[robustness.scenario == "trend"]
    fig = plt.figure(figsize=(8, 5)); plt.bar(["FlowOE", "TWAP"], [trend.mean_flowoe_bps.mean(), trend.mean_twap_bps.mean()])
    plt.ylabel("Mean slippage (bps)"); plt.title("Trend regime: FlowOE vs TWAP")
    save(fig, figdir / "trend_flowoe_vs_twap.svg")

    elapsed = time.perf_counter() - started
    report = {
        "coverage": {
            "scenarios": len(scenarios),
            "cfm_seed_fits": len(robustness),
            "schedule_evaluations": len(families),
            "sampling_step_settings": len(steps),
            "quantity_settings": len(quantity),
            "model_comparisons": 2,
        },
        "runtime_seconds": elapsed,
        "trend_seed_bootstrap_ci": bootstrap_mean_ci(
            robustness.loc[robustness.scenario == "trend", "mean_improvement_bps"].to_numpy()
        ),
    }
    (out / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), **report["coverage"], "runtime_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
