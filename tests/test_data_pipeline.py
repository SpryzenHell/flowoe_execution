# CI regression surface: exercise data-integrity gates on the reconciled branch.
import csv
import pandas as pd

from flowoe_execution.data import _timestamp_series
from scripts.reconstruct_binance_t_depth import _check_event_sequence, _timestamp_seconds


def test_timestamp_units_cover_ms_and_microseconds():
    ms = _timestamp_series(pd.Series([1667347199939]))
    us = _timestamp_series(pd.Series([1735689600010866]))
    assert str(ms.iloc[0]).startswith("2022-11-01 23:59:59")
    assert str(us.iloc[0]).startswith("2025-01-01 00:00:00")


def test_binance_sequence_gap_is_detected():
    previous_last = 100
    contiguous = {
        "first_update_id": "101",
        "last_update_id": "110",
        "pu": "100",
    }
    gap = {
        "first_update_id": "120",
        "last_update_id": "125",
        "pu": "119",
    }
    assert not _check_event_sequence(contiguous, previous_last)
    assert _check_event_sequence(gap, previous_last)


def test_depth_timestamp_units_cover_ms_and_microseconds():
    assert abs(_timestamp_seconds(1667347199939) - 1667347199.939) < 1e-6
    assert abs(_timestamp_seconds(1735689600010866) - 1735689600.010866) < 1e-6
