import pytest
import torch

from scripts.benchmark_fixed_steps import schedule_quality, schedule_stats


def test_reference_schedule_stats_do_not_claim_zero_self_error():
    reference = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]], dtype=torch.float32)
    stats = schedule_stats(reference)
    assert stats["mean_schedule_sum"] == pytest.approx(1.0)
    assert stats["minimum_schedule_weight"] == pytest.approx(0.1)
    assert "mean_abs_schedule_error_vs_rk4" not in stats
    assert "mean_kl_reference_to_euler" not in stats


def test_schedule_quality_is_zero_for_matching_reference():
    reference = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]], dtype=torch.float32)
    stats = schedule_quality(reference, reference)
    assert stats["mean_abs_schedule_error_vs_rk4"] == pytest.approx(0.0)
    assert stats["max_abs_schedule_error_vs_rk4"] == pytest.approx(0.0)
    assert stats["mean_kl_reference_to_euler"] == pytest.approx(0.0, abs=1e-7)
    assert stats["mean_schedule_sum"] == pytest.approx(1.0)


def test_schedule_quality_detects_nonzero_error_and_kl():
    reference = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
    candidate = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    stats = schedule_quality(candidate, reference)
    assert stats["mean_abs_schedule_error_vs_rk4"] > 0
    assert stats["max_abs_schedule_error_vs_rk4"] > 0
    assert stats["mean_kl_reference_to_euler"] > 0


def test_schedule_quality_rejects_shape_mismatch_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="shape mismatch"):
        schedule_quality(torch.ones(1, 3), torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        schedule_quality(torch.tensor([[float("nan"), 0.5]]), torch.tensor([[0.5, 0.5]]))
