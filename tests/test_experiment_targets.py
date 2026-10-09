import numpy as np

from scripts.run_experiment import expert


def test_buy_schedule_favors_lower_future_prices_and_sell_favors_higher():
    prices = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 100.5, 101.5, 99.5])
    buy = expert(prices, 8, side="buy")
    sell = expert(prices, 8, side="sell")
    assert buy.shape == sell.shape == (8,)
    assert np.isfinite(buy).all() and np.isfinite(sell).all()
    assert np.isclose(buy.sum(), 1.0)
    assert np.isclose(sell.sum(), 1.0)
    assert int(np.argmax(buy)) == int(np.argmin(prices))
    assert int(np.argmax(sell)) == int(np.argmax(prices))
