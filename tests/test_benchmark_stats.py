import numpy as np
import pytest

from scripts.benchmark_cuda_fused import stats


def test_fused_benchmark_uses_interpolated_percentiles():
    result = stats([0.0, 10.0])
    assert result["mean_ms"] == pytest.approx(5.0)
    assert result["p50_ms"] == pytest.approx(5.0)
    assert result["p95_ms"] == pytest.approx(9.5)
    assert result["p99_ms"] == pytest.approx(9.9)
    assert result["min_ms"] == 0.0
    assert result["max_ms"] == 10.0


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1.0, 2.0]]])
def test_fused_benchmark_rejects_invalid_timing_samples(values):
    with pytest.raises(ValueError):
        stats(values)
