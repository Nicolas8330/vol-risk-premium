"""Realised volatility from closing prices.

Realised volatility is how much the index really moved. It is the square root
of the average squared daily log return, scaled to a year, and it is quoted in
"vol points" (a value of 16 means 16% a year), the same unit as the VIX.

Two conventions matter and both follow the variance swap contract:
    * the mean return is NOT subtracted (a variance swap does not do it), and
    * the window is 21 trading days, the trading days in 30 calendar days.

Timing is the point of this file. `back[t]` uses returns up to and including
day t, so it is known at the close of t. `fwd[t]` uses the returns of days
t+1 to t+21, so it is what happens AFTER the VIX is read at the close of t.
"""

import numpy as np

from . import HORIZON, TRADING_DAYS


def log_returns(close):
    """Log returns with a leading NaN, so the array lines up with `close`."""
    close = np.asarray(close, dtype=float)
    if np.isnan(close).any():
        raise ValueError('closing prices must have no gaps: drop the missing days first')
    if (close <= 0).any():
        raise ValueError('closing prices must be positive')
    out = np.full(close.shape, np.nan)
    out[1:] = np.log(close[1:] / close[:-1])
    return out


def back(close, window=HORIZON):
    """Realised vol over the last `window` returns, known at the close of day t."""
    r = log_returns(close)
    sq = r * r
    out = np.full(sq.shape, np.nan)
    for t in range(window, len(sq)):
        out[t] = np.sqrt(TRADING_DAYS / window * sq[t - window + 1: t + 1].sum()) * 100.0
    return out


def fwd(close, window=HORIZON):
    """Realised vol over the NEXT `window` returns (days t+1 to t+window).

    This is what a variance swap struck at the close of day t pays on. It is not
    known at t, and it is NaN for the last `window` days, whose future has not
    happened yet.
    """
    b = back(close, window)
    out = np.full(b.shape, np.nan)
    out[:-window] = b[window:]
    return out
