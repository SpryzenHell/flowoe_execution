import numpy as np

from scripts.run_analysis_suite import SCENARIOS, bootstrap_mean_ci, make_market


def test_all_synthetic_regimes_have_valid_market_structure():
    for idx, scenario in enumerate(SCENARIOS):
        book, trades = make_market(scenario, 120, 500 + idx)
        assert len(book) == 120
        assert len(trades) == 120
        assert (book.ask0 > book.bid0).all()
        size_cols = [f"{side}_size{level}" for side in ("bid", "ask") for level in range(10)]
        assert np.isfinite(book[size_cols].to_numpy(float)).all()
        assert (book[size_cols].to_numpy(float) >= 0).all()
        assert (trades.price.to_numpy(float) > 0).all()
        assert (trades.qty.to_numpy(float) > 0).all()


def test_bootstrap_ci_is_finite_and_contains_point_estimate():
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    mean, low, high = bootstrap_mean_ci(values, seed=7, n_boot=2000)
    assert np.isfinite([mean, low, high]).all()
    assert low <= mean <= high
