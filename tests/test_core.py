import numpy as np
import pandas as pd
import torch

from flowoe_execution.data import load_fi2010
from flowoe_execution.execution import ExecutionSimulator
from flowoe_execution.features import l2_features
from flowoe_execution.model import CFMPolicy, ProbabilityFlowODEPolicy

def sample_l2(n=16, levels=10):
    rows = []
    for i in range(n):
        r = {"timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(milliseconds=i)}
        for l in range(levels):
            r[f"bid{l}"] = 100 - l * 0.01
            r[f"ask{l}"] = 100.02 + l * 0.01
            r[f"bid_size{l}"] = 1.0
            r[f"ask_size{l}"] = 1.0
        rows.append(r)
    return pd.DataFrame(rows)

def test_feature_dim():
    x = l2_features(sample_l2())
    assert x.shape == (16, 45)
    assert np.isfinite(x).all()

def test_cfm_loss_and_sample():
    m = CFMPolicy(8)
    c = torch.randn(4, 16, 45)
    y = torch.softmax(torch.randn(4, 8), -1)
    assert torch.isfinite(m.cfm_loss(c, y, "crypto"))
    assert m.sample(c, "crypto", steps=4).shape == (4, 8)

def test_fi_auxiliary_loss():
    m = CFMPolicy(8)
    c = torch.randn(4, 16, 144)
    labels = torch.randint(1, 4, (4, 5))
    assert torch.isfinite(m.fi_aux_loss(c, labels, label_encoding="one_two_three"))

def test_probability_flow_ode():
    m = ProbabilityFlowODEPolicy(8)
    c = torch.randn(2, 16, 45)
    z = m.sample(c, "crypto", steps=4)
    assert z.shape == (2, 8) and torch.isfinite(z).all()

def test_execution_consumes_multiple_levels():
    r = ExecutionSimulator(quantity=2.0, levels=10).run(
        sample_l2(8), np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), benchmark="book_vwap"
    )
    assert r.executed_qty == 2.0
    assert r.levels_consumed == 2
    assert r.avg_price > 100.02

def test_fi2010_loader_transposes_149_by_n(tmp_path):
    raw = np.arange(149 * 4, dtype=np.float32).reshape(149, 4)
    path = tmp_path / "fi.txt"
    np.savetxt(path, raw)
    data = load_fi2010(path)
    assert data.features.shape == (4, 144)
    assert data.labels.shape == (4, 5)

def test_fi2010_row_limit_applies_after_transpose(tmp_path):
    raw = np.arange(149 * 20, dtype=np.float32).reshape(149, 20)
    path = tmp_path / "fi_long.txt"
    np.savetxt(path, raw)
    data = load_fi2010(path, max_rows=3)
    assert data.features.shape == (3, 144)
    assert data.labels.shape == (3, 5)
    np.testing.assert_array_equal(data.features, raw.T[:3, :144])


def test_fi2010_row_limit_on_wide_transposed_file(tmp_path):
    raw = np.arange(149 * 180, dtype=np.float32).reshape(149, 180)
    path = tmp_path / "fi_wide.txt"
    np.savetxt(path, raw)
    data = load_fi2010(path, max_rows=7)
    assert data.features.shape == (7, 144)
    assert data.labels.shape == (7, 5)
    np.testing.assert_array_equal(data.features, raw.T[:7, :144])


def test_fi2010_row_major_file_is_capped(tmp_path):
    raw = np.arange(31 * 149, dtype=np.float32).reshape(31, 149)
    path = tmp_path / "fi_rows.txt"
    np.savetxt(path, raw)
    data = load_fi2010(path, max_rows=5)
    assert data.features.shape == (5, 144)
    assert data.labels.shape == (5, 5)
    np.testing.assert_array_equal(data.features, raw[:5, :144])


def test_fi2010_limit_larger_than_transposed_sample_count(tmp_path):
    raw = np.arange(149 * 12, dtype=np.float32).reshape(149, 12)
    path = tmp_path / "fi_short_wide.txt"
    np.savetxt(path, raw)
    data = load_fi2010(path, max_rows=50000)
    assert data.features.shape == (12, 144)
    np.testing.assert_array_equal(data.features, raw.T[:, :144])


def test_fi2010_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    try:
        load_fi2010(path)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("empty FI-2010 file must be rejected")

def test_fixed_step_fused_path_matches_reference():
    from flowoe_execution.model import FixedStepCryptoSampler

    m = CFMPolicy(8)
    ref = FixedStepCryptoSampler(m, steps=4)
    fused = FixedStepCryptoSampler(m, steps=4, fused_step=lambda x, v, dt: x.add_(dt * v))
    c = torch.randn(2, 16, 45)
    x = torch.randn(2, 8)
    out_ref = ref(c, x.clone())
    out_fused = fused(c, x.clone())
    assert torch.allclose(out_ref, out_fused, atol=1e-6, rtol=1e-5)


def test_fi_auxiliary_loss_accepts_signed_labels():
    m = CFMPolicy(8)
    c = torch.randn(4, 16, 144)
    labels = torch.randint(-1, 2, (4, 5))
    assert torch.isfinite(m.fi_aux_loss(c, labels))


def test_fi_auxiliary_updates_shared_temporal_encoder():
    m = CFMPolicy(8)
    c = torch.randn(4, 16, 144)
    labels = torch.randint(0, 3, (4, 5))
    m.zero_grad()
    loss = m.fi_aux_loss(c, labels)
    loss.backward()
    grads = [p.grad for p in m.context.temporal.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
