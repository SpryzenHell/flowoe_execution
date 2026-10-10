import pytest
import torch

from scripts.benchmark_trt import accuracy_gate, latency_target_gate, output_error_metrics


def test_output_error_metrics_are_zero_for_matching_schedules():
    ref = torch.softmax(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), dim=1)
    result = output_error_metrics(ref, ref.clone())
    assert result["max_abs_output_error"] == pytest.approx(0.0)
    assert result["mean_abs_output_error"] == pytest.approx(0.0)
    assert result["mean_kl_pytorch_to_tensorrt"] == pytest.approx(0.0, abs=1e-7)
    assert result["pytorch_mean_schedule_sum"] == pytest.approx(1.0)
    assert result["tensorrt_mean_schedule_sum"] == pytest.approx(1.0)


def test_output_error_metrics_measure_drift_and_kl():
    ref = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    candidate = torch.tensor([[0.12, 0.18, 0.31, 0.39]])
    result = output_error_metrics(ref, candidate)
    assert result["max_abs_output_error"] == pytest.approx(0.02)
    assert result["mean_abs_output_error"] > 0
    assert result["mean_kl_pytorch_to_tensorrt"] > 0


def test_output_error_metrics_reject_shape_mismatch_and_nonfinite():
    with pytest.raises(ValueError, match="identical"):
        output_error_metrics(torch.ones(1, 3), torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        output_error_metrics(torch.tensor([[float("nan"), 0.5]]), torch.tensor([[0.5, 0.5]]))


def test_accuracy_gate_uses_inclusive_threshold():
    assert accuracy_gate(0.02, 0.02)["accuracy_gate_passed"] is True
    assert accuracy_gate(0.02001, 0.02)["accuracy_gate_passed"] is False
    with pytest.raises(ValueError):
        accuracy_gate(float("nan"), 0.02)


def test_latency_gate_does_not_accept_exactly_two_milliseconds():
    assert latency_target_gate(True, 1.99)["latency_claim_allowed"] is True
    assert latency_target_gate(True, 2.0)["latency_claim_allowed"] is False
    assert latency_target_gate(False, 1.0)["latency_claim_allowed"] is False
