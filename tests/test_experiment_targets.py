import numpy as np
import torch
import pytest

from scripts.run_experiment import expert, fi_training_windows


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
