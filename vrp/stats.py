"""The statistics, in plain formulas so the JavaScript copy can match them.

Every function here has a twin in public/js/vol-risk-premium-engine.js and
scripts/verify-vol-risk-premium.mjs checks that both give the same numbers. The
bootstrap uses the same seeded generator (mulberry32, 32-bit integer maths) in
both languages, so the "random" ranges are identical too.
"""

import functools
import math

import numpy as np

from . import BLOCK, PERIODS, REPS, SEED


# ------------------------------------------------------------------ basics

def mean(x):
    x = np.asarray(x, dtype=float)
    return float(x.sum() / len(x)) if len(x) else float('nan')


def sd(x):
    """Sample standard deviation (n - 1 in the denominator)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float('nan')
    m = x.sum() / n
    return float(math.sqrt(((x - m) ** 2).sum() / (n - 1)))


def sharpe(x, periods=PERIODS):
    """Average P&L divided by its ups and downs, scaled to a year.

    The P&L is that of an unfunded contract (a swap costs nothing to enter), so
    there is no risk-free rate to subtract. NaN when the P&L never varies.
    """
    x = np.asarray(x, dtype=float)
    s = sd(x)
    if not s > 0:
        return float('nan')
    return float(mean(x) / s * math.sqrt(periods))


def skew(x):
    """Skewness: negative means a long tail of losses."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float('nan')
    m = x.sum() / n
    m2 = ((x - m) ** 2).sum() / n
    m3 = ((x - m) ** 3).sum() / n
    return float(m3 / m2 ** 1.5) if m2 > 0 else float('nan')


def hit_rate(x):
    x = np.asarray(x, dtype=float)
    return float((x > 0).sum() / len(x)) if len(x) else float('nan')


def quantile(x, q):
    """Linear-interpolation quantile (the same as numpy's default)."""
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0:
        return float('nan')
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return float(a[lo] + (a[hi] - a[lo]) * (pos - lo))


def shortfall(x, q=0.05):
    """Average of the worst `q` share of outcomes (expected shortfall)."""
    a = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(math.ceil(len(a) * q)))
    return float(a[:k].sum() / k)


def drawdown(x):
    """Deepest fall of the running total from its previous peak.

    Returns depth (a positive number, in the units of x), the positions of the
    peak and the trough (peak -1 means the starting zero), the number of periods
    the fall took, how many periods it took to get back to the old peak (None if
    it never did) and the longest stretch spent below a previous peak.
    """
    x = np.asarray(x, dtype=float)
    cum = np.cumsum(x)
    peak_val, peak_at = 0.0, -1
    depth, d_peak, d_trough = 0.0, -1, -1
    under, longest = 0, 0
    for i, c in enumerate(cum):
        if c > peak_val:
            peak_val, peak_at = c, i
            under = 0
        else:
            under += 1
            longest = max(longest, under)
            if peak_val - c > depth:
                depth, d_peak, d_trough = peak_val - c, peak_at, i
    recover = None
    if depth > 0:
        level = 0.0 if d_peak < 0 else cum[d_peak]
        for j in range(d_trough + 1, len(cum)):
            if cum[j] >= level:
                recover = j - d_trough
                break
    return {'depth': float(depth), 'peak': d_peak, 'trough': d_trough,
            'fall_periods': (d_trough - d_peak) if depth > 0 else 0,
            'recovery_periods': recover, 'longest_under': longest}


# ------------------------------------------------------- the seeded generator

def mulberry32(seed):
    """The generator from the JavaScript world, in Python, bit for bit.

    A tiny 32-bit generator. The point of using it is that the browser and the
    tests draw exactly the same "random" numbers, so a bootstrap range shown on
    the page can be checked against this repository to the last digit.
    """
    mask = 0xFFFFFFFF
    state = [seed & mask]

    def rnd():
        state[0] = (state[0] + 0x6D2B79F5) & mask
        a = state[0]
        t = ((a ^ (a >> 15)) * (1 | a)) & mask
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & mask)) ^ t) & mask
        return ((t ^ (t >> 14)) & mask) / 4294967296.0

    return rnd


@functools.lru_cache(maxsize=32)
def bootstrap_indices(n, block=BLOCK, reps=REPS, seed=SEED):
    """Circular moving-block resamples of the positions 0..n-1.

    Months are not independent (a turbulent month is followed by another), so
    whole runs of `block` neighbouring months are drawn together. A block that
    runs off the end wraps to the start. Shape: (reps, n). Cached: do not modify
    the returned array.
    """
    rnd = mulberry32(seed)
    nb = -(-n // block)
    out = np.empty((reps, nb * block), dtype=np.int64)
    ramp = np.arange(block)
    for r in range(reps):
        for b in range(nb):
            s = int(rnd() * n)
            out[r, b * block:(b + 1) * block] = (s + ramp) % n
    return out[:, :n]


def _rows_sharpe(mat, periods=PERIODS):
    n = mat.shape[1]
    m = mat.sum(axis=1) / n
    var = ((mat - m[:, None]) ** 2).sum(axis=1) / (n - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = m / np.sqrt(var) * math.sqrt(periods)
    out[~(var > 0)] = np.nan
    return out


def _group_rows_sharpe(p, f, min_members=8, periods=PERIODS):
    cnt = f.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        m = (p * f).sum(axis=1) / cnt
        var = (((p - m[:, None]) ** 2) * f).sum(axis=1) / (cnt - 1)
        out = m / np.sqrt(var) * math.sqrt(periods)
    out[(cnt < min_members) | ~(var > 0)] = np.nan
    return out


def boot_ci(values, level=0.95):
    """Percentile range of a list of bootstrap results (NaNs ignored)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [float('nan'), float('nan')]
    a = (1 - level) / 2
    return [quantile(v, a), quantile(v, 1 - a)]


def sharpe_ci(x, block=BLOCK, reps=REPS, seed=SEED):
    """95% range for the Sharpe ratio of a monthly P&L series."""
    x = np.asarray(x, dtype=float)
    idx = bootstrap_indices(len(x), block, reps, seed)
    return boot_ci(_rows_sharpe(x[idx]))


def mean_ci(x, block=BLOCK, reps=REPS, seed=SEED):
    """95% range for the average of a series."""
    x = np.asarray(x, dtype=float)
    idx = bootstrap_indices(len(x), block, reps, seed)
    return boot_ci(x[idx].sum(axis=1) / len(x))


def group_sharpe_ci(pnl, flag, block=BLOCK, reps=REPS, seed=SEED, min_members=8):
    """95% range for the Sharpe of the months where `flag` is True.

    The whole time-ordered series is resampled in blocks and the group is picked
    out of each resample, so the range reflects both the luck of which months
    were drawn and how many of them fall in the group.
    """
    pnl = np.asarray(pnl, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    idx = bootstrap_indices(len(pnl), block, reps, seed)
    return boot_ci(_group_rows_sharpe(pnl[idx], flag[idx], min_members))


def paired_diff_ci(a, b, block=BLOCK, reps=REPS, seed=SEED):
    """95% range for Sharpe(a) - Sharpe(b), resampling the same months in both."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    idx = bootstrap_indices(len(a), block, reps, seed)
    return boot_ci(_rows_sharpe(a[idx]) - _rows_sharpe(b[idx]))
