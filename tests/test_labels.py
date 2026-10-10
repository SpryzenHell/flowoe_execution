import pytest
import torch

from flowoe_execution.labels import normalize_fi2010_labels


@pytest.mark.parametrize(
    ("raw", "encoding", "expected"),
    [
        ([-1, 0, 1], "minus1_0_1", [0, 1, 2]),
        ([0, 1, 2], "zero_one_two", [0, 1, 2]),
        ([1, 2, 3], "one_two_three", [0, 1, 2]),
    ],
)
def test_normalize_fi2010_labels_explicit_encodings(raw, encoding, expected):
    labels, detected = normalize_fi2010_labels(torch.tensor(raw), encoding=encoding)
    assert labels.tolist() == expected
    assert detected == encoding
    assert labels.dtype == torch.long


@pytest.mark.parametrize(
    ("raw", "expected_encoding", "expected"),
    [
        ([-1, 0, 1], "minus1_0_1", [0, 1, 2]),
        ([0, 1, 2], "zero_one_two", [0, 1, 2]),
        ([1, 2, 3], "one_two_three", [0, 1, 2]),
    ],
)
def test_normalize_fi2010_labels_auto_when_observed_range_is_unambiguous(raw, expected_encoding, expected):
    labels, detected = normalize_fi2010_labels(torch.tensor(raw), encoding="auto")
    assert labels.tolist() == expected
    assert detected == expected_encoding


@pytest.mark.parametrize("raw", [
    [[1, 2], [2, 1]],
    [[0, 1], [1, 0]],
])
def test_normalize_fi2010_labels_auto_rejects_ambiguous_subset(raw):
    with pytest.raises(ValueError, match="Cannot infer"):
        normalize_fi2010_labels(torch.tensor(raw), encoding="auto")


@pytest.mark.parametrize(
    ("raw", "encoding"),
    [
        ([0, 3], "zero_one_two"),
        ([-1, 2], "minus1_0_1"),
        ([0, 3], "one_two_three"),
        ([0.0, 1.5, 2.0], "zero_one_two"),
        ([0.0, float("nan"), 1.0], "auto"),
        ([], "auto"),
    ],
)
def test_normalize_fi2010_labels_rejects_invalid_values(raw, encoding):
    with pytest.raises(ValueError):
        normalize_fi2010_labels(torch.tensor(raw), encoding=encoding)
