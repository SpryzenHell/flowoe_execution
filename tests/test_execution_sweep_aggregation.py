import json

import pytest

from scripts.run_execution_sweep import aggregate_reports


def _report(path, seed, improvement, n=6):
    episodes = []
    policies = []
    for i in range(n):
        twap = 0.1 * i
        slip = twap - improvement
        policies.append(slip)
        episodes.append({
            "episode_index": i,
            "l2_start": f"2026-01-01T00:{i:02d}:00+00:00",
            "l2_end": f"2026-01-01T00:{i:02d}:35+00:00",
            "twap_slippage_bps": twap,
            "methods": {"fixed_euler_2": {"slippage_bps": slip}},
        })
    data = {
        "status": "passed",
        "side": "buy",
        "seed": seed,
        "methods": {
            "fixed_euler_2": {
                "slippage_bps": {"mean": sum(policies) / len(policies)},
                "paired_improvement_vs_twap_bps": improvement,
                "paired_improvement_ci95_bps": [improvement - 0.01, improvement + 0.01],
                "completion_mean": 0.99,
                "episodes": n,
                "schedule_error_vs_rk4": {
                    "mean_abs_error": 0.01,
                    "mean_kl_reference_to_candidate": 0.001,
                },
            }
        },
        "episode_results": episodes,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"evaluation_report": str(path)}


def test_aggregate_reports_pools_paired_seed_averaged_windows(tmp_path):
    records = [
        _report(tmp_path / "s7.json", 7, 0.10),
        _report(tmp_path / "s17.json", 17, 0.20),
        _report(tmp_path / "s27.json", 27, 0.30),
    ]
    value = aggregate_reports(records)["buy"]["fixed_euler_2"]["pooled_by_window"]
    assert value["seed_count"] == 3
    assert value["shared_episodes"] == 6
    assert value["mean_paired_improvement_vs_twap_bps"] == pytest.approx(0.20)
    assert value["paired_improvement_ci95_bps"] == pytest.approx([0.20, 0.20])
    assert value["ci_crosses_zero"] is False
    assert value["block_size_episodes"] == 5
    assert value["bootstrap_resamples"] == 5000


def test_aggregate_reports_aligns_only_shared_timestamps(tmp_path):
    records = [
        _report(tmp_path / "s7.json", 7, 0.1, n=6),
        _report(tmp_path / "s17.json", 17, 0.2, n=5),
        _report(tmp_path / "s27.json", 27, 0.3, n=6),
    ]
    value = aggregate_reports(records)["buy"]["fixed_euler_2"]["pooled_by_window"]
    assert value["shared_episodes"] == 5
    assert value["seed_count"] == 3
