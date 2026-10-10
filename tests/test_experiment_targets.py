import numpy as np
import torch
import pytest

from scripts.run_experiment import expert, fi_training_windows
from scripts.evaluate_execution_grid import compare_schedules, summarize_paired_method
from scripts.benchmark_trt import accuracy_gate


def test_fi_training_windows_keep_all_label_horizons_inside_training_split():
    features = torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4)
    labels = np.arange(100 * 5, dtype=np.int16).reshape(100, 5)
    contexts, targets = fi_training_windows(
        features, labels, train_ratio=0.60, context_len=8, stride=4, max_label_horizon=10
    )
    assert len(contexts) == len(targets) > 0
    split = 60
    for ctx, target in zip(contexts, targets):
        final_row = int(ctx[-1, 0].item() // 4)
        assert final_row + 10 < split
        torch.testing.assert_close(target, torch.as_tensor(labels[final_row], dtype=torch.long))


@pytest.mark.parametrize("args", [
    (np.zeros((5, 4)), np.zeros((4, 5)), 0.6, 2, 1, 1),
    (np.zeros((5, 4)), np.zeros((5, 5)), 1.0, 2, 1, 1),
    (np.zeros((5, 4)), np.zeros((5, 5)), 0.6, 0, 1, 1),
])
def test_fi_training_windows_reject_invalid_inputs(args):
    with pytest.raises(ValueError):
        fi_training_windows(*args)



def test_schedule_comparison_is_zero_for_identical_normalized_outputs():
    p = torch.softmax(torch.tensor([[1.0, 2.0, -1.0], [0.4, 0.1, 0.2]]), dim=1)
    result = compare_schedules(p, p.clone())
    assert result["mean_abs_error"] == pytest.approx(0.0)
    assert result["max_abs_error"] == pytest.approx(0.0)
    assert result["mean_kl_reference_to_candidate"] == pytest.approx(0.0, abs=1e-7)


def test_schedule_comparison_rejects_shape_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError, match="identical"):
        compare_schedules(torch.ones(1, 3), torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        compare_schedules(torch.tensor([[1.0, float("nan")]]), torch.ones(1, 2))



def test_paired_method_summary_accepts_only_completed_episode_pairs():
    baseline = [2.0, 2.5, 2.2]
    slippage = [1.8, 2.4, 2.0]
    completion = [1.0, 0.9, 1.0]
    errors = [
        {"mean_abs_error": 0.0, "max_abs_error": 0.0, "mean_kl_reference_to_candidate": 0.0}
        for _ in baseline
    ]
    result = summarize_paired_method(baseline, slippage, completion, errors, seed=11)
    assert result["episodes"] == 3
    assert result["baseline_episodes"] == 3
    assert result["improvement_vs_twap_bps"] == pytest.approx(0.1666666667)
    assert result["paired_improvement_ci95_bps"][0] <= result["paired_improvement_vs_twap_bps"]


def test_paired_method_summary_rejects_partial_baseline_alignment():
    errors = [{"mean_abs_error": 0.0, "max_abs_error": 0.0, "mean_kl_reference_to_candidate": 0.0}]
    with pytest.raises(ValueError, match="same episodes"):
        summarize_paired_method([2.0, 2.1], [1.8], [1.0], errors)



def test_tensor_rt_accuracy_gate_blocks_latency_claim_when_error_is_too_high():
    passed = accuracy_gate(0.01, 0.05)
    failed = accuracy_gate(0.08, 0.05)
    assert passed["status"] == "passed"
    assert passed["latency_claim_allowed"] is True
    assert failed["status"] == "failed_accuracy"
    assert failed["accuracy_gate_passed"] is False
    assert failed["latency_claim_allowed"] is False


def test_tensor_rt_accuracy_gate_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="finite"):
        accuracy_gate(float("nan"), 0.05)
    with pytest.raises(ValueError, match="tolerance positive"):
        accuracy_gate(0.01, 0.0)
