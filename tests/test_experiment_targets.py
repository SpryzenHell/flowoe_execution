import numpy as np
import torch
import pytest

from scripts.run_experiment import expert, fi_training_windows
from scripts.evaluate_execution_grid import compare_schedules


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
