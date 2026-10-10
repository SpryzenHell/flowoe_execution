import numpy as np
import pandas as pd
import torch

from flowoe_execution.execution import ExecutionSimulator, book_vwap, market_vwap, make_schedule_from_trajectory
from flowoe_execution.features import l2_features
from flowoe_execution.model import CFMPolicy, ProbabilityFlowODEPolicy, FixedStepCryptoSampler
from flowoe_execution.ode import integrate_ode


def sample_l2(n=12, levels=10):
    rows = []
    for i in range(n):
        row = {"timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(milliseconds=i)}
        for level in range(levels):
            row[f"bid{level}"] = 100 - 0.01 * (level + 1)
            row[f"ask{level}"] = 100 + 0.02 + 0.01 * level
            row[f"bid_size{level}"] = 1.0
            row[f"ask_size{level}"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_schedule_is_normalized():
    schedule = make_schedule_from_trajectory(np.array([0.0, 1.0, 2.0]))
    assert np.isclose(schedule.sum(), 1.0)
    assert np.all(schedule > 0)


def test_sell_execution_consumes_multiple_levels():
    result = ExecutionSimulator(quantity=2.0, side="sell").run(
        sample_l2(), np.array([1.0] + [0.0] * 7), benchmark="book_vwap"
    )
    assert result.executed_qty == 2.0
    assert result.levels_consumed == 2
    assert result.avg_price < 100.0


def test_partial_fill_is_reported():
    book = sample_l2()
    for level in range(10):
        book[f"ask_size{level}"] = 0.02
    result = ExecutionSimulator(quantity=1.0).run(
        book, np.array([1.0] + [0.0] * 7), benchmark="book_vwap"
    )
    assert 0.0 < result.executed_qty < result.requested_qty
    assert 0.0 < result.completion < 1.0


def test_invalid_schedule_is_rejected():
    try:
        ExecutionSimulator().run(sample_l2(), np.zeros(8), benchmark="book_vwap")
    except ValueError:
        pass
    else:
        raise AssertionError("zero-mass schedule should raise ValueError")


def test_zero_top_level_depth_keeps_features_finite():
    book = sample_l2()
    book["bid_size0"] = 0.0
    book["ask_size0"] = 0.0
    assert np.isfinite(l2_features(book)).all()


def test_rk4_integrates_exponential():
    result = integrate_ode(lambda _t, x: x, torch.tensor([[1.0]]), 0.0, 1.0, 32)
    assert torch.allclose(result, torch.tensor([[np.e]]), atol=1e-5)


def test_rk4_handles_reverse_interval():
    result = integrate_ode(lambda _t, x: torch.ones_like(x), torch.tensor([[1.0]]), 1.0, 0.0, 8)
    assert torch.allclose(result, torch.tensor([[0.0]]), atol=1e-6)


def test_probability_flow_score_path_is_finite():
    model = ProbabilityFlowODEPolicy(8)
    context = torch.randn(2, 16, 45)
    target = torch.softmax(torch.randn(2, 8), dim=1)
    assert torch.isfinite(model.score_loss(context, target, "crypto"))
    assert torch.isfinite(model.sample(context, "crypto", steps=4)).all()


def test_fixed_step_sampler_is_normalized_and_fused_equivalent():
    model = CFMPolicy(8)
    context = torch.randn(2, 16, 45)
    latent = torch.randn(2, 8)
    reference = FixedStepCryptoSampler(model, steps=4)
    fused = FixedStepCryptoSampler(model, steps=4, fused_step=lambda x, v, dt: x.add_(dt * v))
    out_ref = reference(context, latent.clone())
    out_fused = fused(context, latent.clone())
    assert torch.allclose(out_ref.sum(dim=1), torch.ones(2))
    assert torch.allclose(out_ref, out_fused, atol=1e-6, rtol=1e-5)


def test_fixed_step_dynamic_time_vector_matches_constant_time_reference():
    torch.manual_seed(41)
    model = CFMPolicy(8).eval()
    steps = 6
    sampler = FixedStepCryptoSampler(model, steps=steps).eval()
    for batch in (1, 4, 8):
        context = torch.randn(batch, 32, 45)
        latent = torch.randn(batch, 8)
        actual = sampler(context, latent.clone())
        ctx = model.context.crypto_encode(context, "crypto")
        x = latent.clone()
        dt = 1.0 / (steps - 1)
        for i in range(steps - 1):
            t = torch.full((batch,), i * dt, dtype=x.dtype, device=x.device)
            x = x + dt * model.vf(t, x, ctx)
        expected = torch.softmax(x, dim=1)
        assert actual.shape == (batch, 8)
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual.sum(1), torch.ones(batch), atol=1e-6)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_vwap_helpers_return_finite_values():
    book = sample_l2()
    trades = pd.DataFrame(
        {"timestamp": book.timestamp, "price": np.linspace(99.9, 100.1, len(book)), "qty": 1.0}
    )
    assert np.isfinite(book_vwap(book, book.timestamp.iloc[0], book.timestamp.iloc[-1]))
    assert np.isfinite(market_vwap(trades, trades.timestamp.iloc[0], trades.timestamp.iloc[-1]))
