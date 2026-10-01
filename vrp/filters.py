"""One simple filter, tested honestly.

Choosing a filter after seeing which one works is how backtests lie. So:

    * The filter was written down before running anything: do not sell when the
      VIX curve is inverted (the 3-month VIX is below the 1-month VIX). It has
      no number to tune. Its logic is that an inverted curve means the market is
      already in stress, where the short-term insurance price is above the
      longer one.
    * A second family, "do not sell when the VIX is below a floor", has one
      number to tune. It is tested walk-forward: each month the floor is chosen
      using only earlier months, then applied to the new month.
    * The number of variants tried is counted and reported (see VARIANTS).

Honest caveat: I know from history that curves inverted in 2008 and 2020, so the
term-structure filter is not blind. Walk-forward protects the floor's number, not
my knowledge of what happened.
"""

import numpy as np

from . import stats

FLOOR_GRID = [12.0, 14.0, 16.0, 18.0, 20.0]     # floors tried, VIX points
MIN_TRAIN = 120                                  # months of history before the first walk-forward choice
VARIANTS = {'term_structure': 1, 'floor_grid': len(FLOOR_GRID)}


def sell_term(slope):
    """Sell only when the curve is in contango or flat (3-month VIX >= 1-month VIX)."""
    return np.asarray(slope, dtype=float) >= 0.0


def sell_floor(vix, floor):
    """Sell only when the VIX is at or above `floor`."""
    return np.asarray(vix, dtype=float) >= floor


def apply(pnl, sell):
    """P&L when the swap is only sold where `sell` is True (zero when flat)."""
    return np.where(np.asarray(sell, dtype=bool), np.asarray(pnl, dtype=float), 0.0)


def walk_forward_floor(vix, pnl, grid=FLOOR_GRID, min_train=MIN_TRAIN):
    """Choose the floor month by month using only earlier months.

    For month j the candidates (no floor, then each floor in `grid`) are scored by
    the Sharpe ratio they would have had over months 0..j-1, and the best one is
    applied to month j. Ties go to the lower floor. Nothing about month j, or any
    later month, is used to pick its floor.

    Returns the out-of-sample P&L (zero when the filter says stay out), the floor
    chosen each month, and the position of the first out-of-sample month.
    """
    vix = np.asarray(vix, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    cands = [0.0] + list(grid)
    n = len(pnl)
    # cumulative sums let each candidate be scored on any prefix in one step
    sells = [(vix >= f) for f in cands]
    oos, chosen = [], []
    for j in range(min_train, n):
        best, best_score = 0, -np.inf
        for c, sell in enumerate(sells):
            s = stats.sharpe(np.where(sell[:j], pnl[:j], 0.0))
            if np.isfinite(s) and s > best_score + 1e-12:
                best, best_score = c, s
        chosen.append(cands[best])
        oos.append(pnl[j] if sells[best][j] else 0.0)
    return np.array(oos), np.array(chosen), min_train


def walk_forward_adopt(pnl, sell, min_train=60):
    """Walk-forward test of the term-structure filter itself.

    Each month, look only at the months before it: if selling with the filter
    would have had a higher Sharpe ratio than selling every month, use the
    filter this month, otherwise do not. This asks "would I have kept the filter
    on in real time?" rather than "does it look good over the whole sample?".

    Returns the out-of-sample P&L, a True/False per month for "filter adopted",
    and the position of the first out-of-sample month.
    """
    pnl = np.asarray(pnl, dtype=float)
    sell = np.asarray(sell, dtype=bool)
    filt = np.where(sell, pnl, 0.0)
    oos, used = [], []
    for j in range(min_train, len(pnl)):
        a, b = stats.sharpe(filt[:j]), stats.sharpe(pnl[:j])
        adopt = bool(np.isfinite(a) and np.isfinite(b) and a > b)
        used.append(adopt)
        oos.append(filt[j] if adopt else pnl[j])
    return np.array(oos), np.array(used, dtype=bool), min_train
