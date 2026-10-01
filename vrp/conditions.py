"""What was known on the day a swap was sold, and the groups it defines.

Every feature below is computed from data available at the close of the start
day. Two rules keep it honest:

    * a feature never reads a later row (the tests rewrite the end of the data
      and check that nothing earlier moves), and
    * monthly series (fed funds, NBER) are read with a lag, because a month is
      not known until it has ended and NBER dates recessions many months late.

The last point matters for the recession split. NBER announces a recession
long after it has started, so nobody could have used it on the day. It is shown
because it answers "what happened in recessions", not as a rule to trade.

The cut-offs are written here, before any result was looked at, and not moved
afterwards. There are 7 families and 21 groups in all: count them when you
read a "significant" group.
"""

import numpy as np

# family -> (feature name, edges, labels). A value below the first edge is the
# first group, at or above an edge the next one.
FAMILIES = {
    'vix_level': ('vix', [14.0, 20.0, 30.0], ['below_14', '14_to_20', '20_to_30', '30_and_above']),
    'vix_tercile': ('vix', None, ['low_third', 'middle_third', 'high_third']),
    'gap': ('gap', [0.0, 5.0], ['below_0', '0_to_5', '5_and_above']),
    'slope': ('slope', [0.0, 2.0], ['inverted', '0_to_2', '2_and_above']),
    'vix_change': ('vchg', [-0.10, 0.10], ['fell', 'flat', 'rose']),
    'rates': ('ff12', [-0.25, 0.25], ['falling', 'flat', 'rising']),
    'recession': ('rec', [0.5], ['expansion', 'recession']),
}
CHANGE_DAYS = 5          # window of the recent change in the VIX
RATE_LAG = (2, 14)       # fed funds read 2 months back, compared with 14 months back


def features(vix, vxv, rv_back, ff_lag2=None, ff_lag14=None, rec=None):
    """Features for every day of the table, from what was known at that close.

    gap    VIX minus the realised volatility of the last 21 days (vol points)
    slope  3-month VIX minus VIX (positive = contango, negative = inverted)
    vchg   change of the VIX over the last 5 trading days, as a fraction
    ff12   fed funds now (2 months back) minus a year before that, in points
    rec    1 in an NBER recession month (looking back, not knowable on the day)
    """
    vix = np.asarray(vix, dtype=float)
    n = len(vix)
    vchg = np.full(n, np.nan)
    vchg[CHANGE_DAYS:] = vix[CHANGE_DAYS:] / vix[:-CHANGE_DAYS] - 1.0
    out = {
        'vix': vix,
        'gap': vix - np.asarray(rv_back, dtype=float),
        'slope': np.asarray(vxv, dtype=float) - vix,
        'vchg': vchg,
    }
    out['ff12'] = (np.asarray(ff_lag2, dtype=float) - np.asarray(ff_lag14, dtype=float)
                   if ff_lag2 is not None else np.full(n, np.nan))
    out['rec'] = np.asarray(rec, dtype=float) if rec is not None else np.full(n, np.nan)
    return out


def tercile_edges(values):
    """The two cut-offs that split `values` into thirds (full sample: descriptive)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    from .stats import quantile
    return [quantile(v, 1 / 3), quantile(v, 2 / 3)]


def labels(values, edges):
    """Group number for each value (0 for below the first edge), -1 when unknown."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), -1, dtype=int)
    ok = np.isfinite(v)
    out[ok] = np.searchsorted(np.asarray(edges, dtype=float), v[ok], side='right')
    return out


def runs(cycle_numbers):
    """Number of stretches of consecutive cycles (the independent episodes)."""
    j = np.asarray(cycle_numbers, dtype=int)
    if len(j) == 0:
        return 0
    return int(1 + (np.diff(j) != 1).sum())
