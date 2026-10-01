"""The reading for the latest data, and the file the page loads it from.

The page computes this in the browser from `/live/vix.json`. This module is the
same calculation in Python, so a server job can write the file, the tests can
check it, and the JavaScript can be compared against it.

Live file schema (parallel arrays over one calendar, oldest first):

    {"asOf": "2026-09-28",           last date with a VIX value
     "generated": "2026-09-30T06:00:00Z",
     "source": "FRED",
     "dates": ["2016-09-30", ...],
     "vix": [13.29, ..., null],      VIXCLS, null where not published
     "vxv": [null, ..., 18.23],      VXVCLS, or the whole field null if unavailable
     "spx": [2168.27, ..., null]}    SP500 close, null where not published

What "today" means here: the VIX at its last date, the realised volatility of the
last 21 S&P 500 returns up to that date, the gap between the two, where each sits
in the 1990 to now history, and how selling a swap turned out from days that
looked alike. It is a description of conditions, not a forecast.
"""

import math

import numpy as np

from . import HORIZON, rv, swap
from .conditions import labels

LEVEL_EDGES = [14.0, 20.0, 30.0]
LEVEL_NAMES = ['below_14', '14_to_20', '20_to_30', '30_and_above']


def live_payload(spx, vix, vxv, generated):
    """Build the live file from FRED series (each a pandas Series with dates)."""
    import pandas as pd
    start = spx.dropna().index[0]
    idx = spx.index.union(vix.index).union(vxv.index)
    idx = idx[idx >= start]
    s, v, x = spx.reindex(idx), vix.reindex(idx), vxv.reindex(idx)
    keep = s.notna() | v.notna()
    idx, s, v, x = idx[keep], s[keep], v[keep], x[keep]
    clean = lambda a: [None if not math.isfinite(z) else float(z) for z in a]
    last = v.dropna().index[-1]
    return {
        'asOf': str(last.date()), 'generated': generated, 'source': 'FRED (VIXCLS, VXVCLS, SP500)',
        'dates': [str(d.date()) for d in idx],
        'vix': clean(v.values), 'vxv': clean(x.values) if x.notna().any() else None,
        'spx': clean(s.values),
    }


def _share_le(values, x):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float((v <= x).sum() / len(v))


def reading(live, hist_vix, hist_vxv, hist_rv_back, hist_rv_fwd, cost=0.5):
    """The latest reading and how similar past days turned out.

    `hist_*` are the daily arrays of the study table (1990 to the last complete
    day). "Similar" means the same VIX band (below 14, 14 to 20, 20 to 30, 30 and
    above) and, when the 3-month VIX is available, the same curve sign.
    """
    dates = live['dates']
    vix = [np.nan if z is None else z for z in live['vix']]
    vxv = [np.nan] * len(dates) if live.get('vxv') is None else [np.nan if z is None else z for z in live['vxv']]
    spx = [np.nan if z is None else z for z in live['spx']]

    iv = max(i for i, z in enumerate(vix) if np.isfinite(z))
    as_of = dates[iv]
    # closes up to the VIX date (ISO dates compare correctly as text)
    pairs = [(d, c) for d, c in zip(dates, spx) if np.isfinite(c) and d <= as_of]
    closes = np.array([c for _, c in pairs])
    if len(closes) < HORIZON + 1:
        raise ValueError('not enough S&P 500 closes for a 21-day realised volatility')
    rv_now = float(rv.back(closes)[-1])
    rv_date = pairs[-1][0]
    v_now = float(vix[iv])
    x_now = float(vxv[iv]) if np.isfinite(vxv[iv]) else None
    gap = v_now - rv_now
    slope = None if x_now is None else x_now - v_now

    hv, hx, hb, hf = (np.asarray(a, dtype=float) for a in (hist_vix, hist_vxv, hist_rv_back, hist_rv_fwd))
    band = int(labels([v_now], LEVEL_EDGES)[0])
    day_band = labels(hv, LEVEL_EDGES)
    same = (day_band == band)
    if slope is not None:
        with np.errstate(invalid='ignore'):
            same &= np.isfinite(hx) & ((hx - hv >= 0) == (slope >= 0))
    p = swap.daily_windows(hv, hf, cost)
    out = p[same & np.isfinite(p)]
    similar = {'band': LEVEL_NAMES[band], 'curve': None if slope is None else ('contango' if slope >= 0 else 'inverted'),
               'days': int(len(out))}
    if len(out):
        from . import stats
        similar.update({'months': len(out) / HORIZON, 'share_positive': stats.hit_rate(out),
                        'mean': stats.mean(out), 'p05': stats.quantile(out, 0.05), 'worst': float(out.min())})
    return {
        'as_of': as_of, 'rv_date': rv_date, 'vix': v_now, 'vxv': x_now,
        'rv_back': rv_now, 'gap': gap, 'slope': slope,
        'pct_vix': _share_le(hv, v_now),
        'pct_gap': _share_le(hv - hb, gap),
        'pct_slope': None if slope is None else _share_le(hx - hv, slope),
        'similar': similar,
    }
