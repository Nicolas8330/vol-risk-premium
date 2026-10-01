"""The trade: sell a one-month variance swap, using the VIX as the strike.

A variance swap pays the difference between the volatility that really happens
and the volatility you agreed to at the start. The seller receives the strike
and pays the realised. The fair strike of a 30-day variance swap is close to
the square of the VIX (that is what the VIX formula measures), so this study
uses the VIX read at the close of day t as the strike K.

For a "vega notional" N (the dollars made or lost per vol point near the
strike), the seller's profit at expiry is

        N / (2 K) x (K^2 - RV^2)

with RV the realised volatility over the next 21 trading days. Read it as:
if RV ends up equal to K, you break even. If RV is below K you win about N per
point of gap. If RV is far above K you lose a lot more than you would win for
the same gap, because the loss grows with the square of RV. That asymmetry is
the whole story of this project.

Costs: the seller is paid a strike a little below the fair one. A cost of `c`
vol points means the strike received is K - c. This is a stand-in for the
bid-offer of a real swap, a real trade uses VIX futures, SPX options or a
dealer swap, with their own costs and margin, which are NOT modelled here.

Results are per 1 unit of vega notional ("vega points"): multiply by N dollars.
"""

import numpy as np

from . import HORIZON

DEFAULT_COST = 0.5      # vol points off the strike, an assumption, tested from 0 to 2


def pnl(strike, realised, cost=0.0):
    """Profit of the seller of a variance swap, per 1 unit of vega notional."""
    ke = np.asarray(strike, dtype=float) - cost
    if np.any(ke <= 0):
        raise ValueError('the strike net of costs must stay positive')
    return (ke * ke - np.asarray(realised, dtype=float) ** 2) / (2.0 * ke)


class Cycles:
    """Non-overlapping 21-day cycles starting at one offset.

    Cycle j opens at the close of day `offset + 21 j` and expires 21 trading days
    later, the day the next cycle opens. Cycles with no VIX (or an unfinished
    future) are dropped; `j` keeps each surviving cycle's number so gaps are
    visible when counting runs of neighbouring cycles.
    """

    def __init__(self, idx, j, strike, realised, cost):
        self.idx = idx            # position of the start day in the daily table
        self.j = j                # cycle number, so neighbours differ by 1
        self.strike = strike
        self.realised = realised
        self.cost = cost
        self.pnl = pnl(strike, realised, cost)

    def __len__(self):
        return len(self.idx)


def cycles(vix, rv_fwd, offset=0, horizon=HORIZON, cost=0.0):
    """Cut the daily table into non-overlapping cycles and price each one."""
    vix = np.asarray(vix, dtype=float)
    rv_fwd = np.asarray(rv_fwd, dtype=float)
    n = len(vix)
    starts = np.arange(offset, n - horizon, horizon)
    ok = np.isfinite(vix[starts]) & np.isfinite(rv_fwd[starts])
    j = np.arange(len(starts))[ok]
    s = starts[ok]
    return Cycles(s, j, vix[s], rv_fwd[s], cost)


def daily_windows(vix, rv_fwd, cost=0.0):
    """P&L of a swap opened on EVERY day (overlapping windows), NaN where unknown.

    Used for the worst outcome (a real trader is not tied to one start day) and
    for the average, not for the Sharpe ratio, because neighbouring windows share
    20 of their 21 days.
    """
    vix = np.asarray(vix, dtype=float)
    rv_fwd = np.asarray(rv_fwd, dtype=float)
    out = np.full(len(vix), np.nan)
    ok = np.isfinite(vix) & np.isfinite(rv_fwd)
    out[ok] = pnl(vix[ok], rv_fwd[ok], cost)
    return out
