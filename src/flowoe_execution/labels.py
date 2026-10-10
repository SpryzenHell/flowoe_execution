from __future__ import annotations

from typing import Literal

import torch

FI2010LabelEncoding = Literal["auto", "zero_one_two", "minus1_0_1", "one_two_three"]


def normalize_fi2010_labels(
    labels: torch.Tensor,
    encoding: FI2010LabelEncoding = "auto",
) -> tuple[torch.Tensor, str]:
    """Convert FI-2010 labels to canonical integer classes 0, 1, 2.

    Auto-detection is accepted only when the observed label range identifies
    the source encoding: negative values identify {-1, 0, 1}; a value of 3
    identifies {1, 2, 3}; and a value of 0 with no negatives identifies
    {0, 1, 2}. A subset such as {1, 2} is ambiguous and is rejected rather
    than silently shifting labels in a training minibatch.
    """
    value = torch.as_tensor(labels)
    if value.numel() == 0:
        raise ValueError("FI-2010 labels must be non-empty")
    if value.is_floating_point():
        if not torch.isfinite(value).all():
            raise ValueError("FI-2010 labels must be finite")
        rounded = value.round()
        if not torch.equal(value, rounded):
            raise ValueError("FI-2010 labels must be integral class IDs")
        value = rounded
    value = value.to(dtype=torch.long)
    low, high = int(value.min().item()), int(value.max().item())

    selected = encoding
    if selected == "auto":
        if low < 0:
            selected = "minus1_0_1"
        elif high == 3:
            selected = "one_two_three"
        elif low == 0 and high == 2:
            selected = "zero_one_two"
        else:
            raise ValueError(
                f"Cannot infer FI-2010 label encoding from observed range [{low}, {high}]; "
                "pass encoding='zero_one_two', 'minus1_0_1', or 'one_two_three' explicitly"
            )

    if selected == "zero_one_two":
        if low < 0 or high > 2:
            raise ValueError(f"Expected FI-2010 labels in {{0,1,2}}, observed [{low}, {high}]")
        normalized = value
    elif selected == "minus1_0_1":
        if low < -1 or high > 1:
            raise ValueError(f"Expected FI-2010 labels in {{-1,0,1}}, observed [{low}, {high}]")
        normalized = value + 1
    elif selected == "one_two_three":
        if low < 1 or high > 3:
            raise ValueError(f"Expected FI-2010 labels in {{1,2,3}}, observed [{low}, {high}]")
        normalized = value - 1
    else:
        raise ValueError(f"Unsupported FI-2010 label encoding: {selected}")

    if int(normalized.min().item()) < 0 or int(normalized.max().item()) > 2:
        raise ValueError("Normalized FI-2010 labels fell outside canonical classes {0,1,2}")
    return normalized.contiguous(), selected
