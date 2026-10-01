"""Files for the site: the derived history, the live-file fallback, the reference values.

The site never receives S&P 500 closes for 1990 to 2016 (that table is not ours
to redistribute). It receives the VIX and 3-month VIX from FRED, and two
numbers derived from the closes for each day: the realised volatility of the
last 21 returns and of the next 21. That is all the exhibit needs.

`ref_check.json` holds what Python computes on exactly those files. The
JavaScript in the browser must reproduce every value to 1e-9
(scripts/verify-vol-risk-premium.mjs in the site).
"""

import json
import math

import numpy as np

from . import SEED, conditions, filters, stats, swap, study, today

TOL = 1e-9


def clean(o):
    """Make a structure JSON-safe: NaN and infinity become null, numpy becomes plain."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def history_payload(table, generated):
    """The derived daily table the exhibit loads (no S&P 500 prices)."""
    def col(name):
        return [None if not math.isfinite(v) else float(v) for v in table[name].values]
    return {
        'schema': 1, 'generated': generated, 'source': 'FRED VIXCLS and VXVCLS; realised volatility derived from S&P 500 closes',
        'asOf': str(table['vix'].dropna().index[-1].date()),
        'lastComplete': str(table['rv_fwd'].dropna().index[-1].date()),
        'dates': [str(d.date()) for d in table.index],
        'vix': col('vix'), 'vxv': col('vxv'), 'rvb': col('rv_back'), 'rvf': col('rv_fwd'),
    }


def as_arrays(hist):
    """Arrays with NaN for null, the shape every function above expects."""
    nan = lambda a: np.array([np.nan if v is None else v for v in a], dtype=float)
    return {'dates': np.array(hist['dates']), 'vix': nan(hist['vix']), 'vxv': nan(hist['vxv']),
            'rvb': nan(hist['rvb']), 'rvf': nan(hist['rvf'])}


def _cycle_ref(vix, rvf, offset, cost, dates):
    c = swap.cycles(vix, rvf, offset, cost=cost)
    s = study.summ(c.pnl)
    return {'offset': offset, 'cost': cost, 'n': s['n'], 'sum': float(c.pnl.sum()), 'sharpe': s['sharpe'],
            'mean': s['mean'], 'median': s['median'], 'worst': s['worst'], 'skew': s['skew'], 'es5': s['es5'],
            'hit': s['hit'], 'dd_depth': s['dd']['depth'], 'dd_recovery': s['dd']['recovery_periods'],
            'dd_longest': s['dd']['longest_under'], 'dd_peak': s['dd']['peak'], 'dd_trough': s['dd']['trough']}


SELECTIONS = [
    {'year_from': 1990, 'year_to': 2026, 'offset': 0, 'cost': 0.5, 'mode': 'all', 'level': 16, 'notional': 100},
    {'year_from': 2005, 'year_to': 2015, 'offset': 3, 'cost': 1.0, 'mode': 'floor', 'level': 16, 'notional': 150},
    {'year_from': 1995, 'year_to': 2026, 'offset': 10, 'cost': 0.5, 'mode': 'cap', 'level': 30, 'notional': 60},
    {'year_from': 2008, 'year_to': 2026, 'offset': 17, 'cost': 0.0, 'mode': 'curve', 'level': 16, 'notional': 40},
]


def ref_checks(hist, live, cost=study.COST):
    """Reference values for the JavaScript port, computed from the shipped files."""
    from . import rv
    a = as_arrays(hist)
    vix, vxv, rvb, rvf, dates = a['vix'], a['vxv'], a['rvb'], a['rvf'], a['dates']
    out = {'tolerance': TOL}

    # realised volatility straight from closes (FRED's S&P 500 in the live file)
    closes = np.array([v for v in live['spx'] if v is not None])
    b, f = rv.back(closes), rv.fwd(closes)
    out['rv'] = {'n': int(len(closes)), 'back_sum': float(np.nansum(b)), 'back_count': int(np.isfinite(b).sum()),
                 'back_last': float(b[-1]), 'back_100': float(b[100]), 'fwd_sum': float(np.nansum(f)),
                 'fwd_count': int(np.isfinite(f).sum()), 'fwd_100': float(f[100]),
                 'fwd_21': float(rv.fwd(closes, 21)[500]), 'fwd_10': float(rv.fwd(closes, 10)[500])}

    # the mulberry32 stream
    rnd = stats.mulberry32(SEED)
    out['rng'] = [rnd() for _ in range(6)]

    # the daily premium
    ok = np.isfinite(vix) & np.isfinite(rvf)
    prem = (vix - rvf)[ok]
    out['premium'] = {'n': int(len(prem)), 'mean': stats.mean(prem), 'median': stats.quantile(prem, 0.5),
                      'q05': stats.quantile(prem, 0.05), 'q99': stats.quantile(prem, 0.99),
                      'pos': stats.hit_rate(prem), 'es5': stats.shortfall(prem, 0.05)}

    out['cycles'] = [_cycle_ref(vix, rvf, o, c, dates) for o in (0, 7, 20) for c in (0.0, cost)]
    out['pnl_point'] = {'strike': 19.43, 'realised': 51.3659, 'cost': 0.5, 'value': float(swap.pnl(19.43, 51.3659, 0.5))}

    # bootstrap ranges on the offset-0 path
    c0 = swap.cycles(vix, rvf, 0, cost=cost)
    out['bootstrap'] = {'sharpe_ci': stats.sharpe_ci(c0.pnl), 'mean_ci': stats.mean_ci(c0.pnl),
                        'first_idx': stats.bootstrap_indices(len(c0.pnl))[0][:12].tolist()}

    # groups
    feats = conditions.features(vix, vxv, rvb)
    groups = []
    for fam in ('vix_level', 'gap', 'slope', 'vix_change'):
        feat, edges, names = conditions.FAMILIES[fam]
        lab = conditions.labels(feats[feat], edges)[c0.idx]
        for g, name in enumerate(names):
            m = lab == g
            if m.sum() < 8:
                continue
            rest = (lab >= 0) & ~m
            known = lab >= 0
            row = {'family': fam, 'group': name, 'n': int(m.sum()), 'episodes': conditions.runs(c0.j[m]),
                   'sharpe': stats.sharpe(c0.pnl[m]), 'mean': stats.mean(c0.pnl[m]),
                   'worst': float(c0.pnl[m].min()),
                   'sharpe_ci': stats.group_sharpe_ci(c0.pnl[known], m[known]),
                   'diff_ci': study._diff_ci(c0.pnl[known], m[known], rest[known])}
            groups.append(row)
    out['groups'] = groups

    # filters
    has = np.isfinite(feats['slope'][c0.idx])
    p = c0.pnl[has]
    sell = filters.sell_term(feats['slope'][c0.idx][has])
    f_ = filters.apply(p, sell)
    oos, chosen, st = filters.walk_forward_floor(c0.strike, c0.pnl)
    oos2, used, st2 = filters.walk_forward_adopt(p, sell)
    out['filters'] = {
        'term': {'n': int(len(p)), 'sold': int(sell.sum()), 'sharpe': stats.sharpe(f_), 'base_sharpe': stats.sharpe(p),
                 'worst': float(f_.min()), 'diff_ci': stats.paired_diff_ci(f_, p)},
        'floor16': {'sharpe': stats.sharpe(filters.apply(c0.pnl, c0.strike >= 16.0)), 'sold': int((c0.strike >= 16.0).sum())},
        'floor_walk': {'start': st, 'n': int(len(oos)), 'sharpe': stats.sharpe(oos), 'sum': float(oos.sum()),
                       'chosen_sum': float(chosen.sum())},
        'adopt_walk': {'start': st2, 'n': int(len(oos2)), 'sharpe': stats.sharpe(oos2), 'sum': float(oos2.sum()),
                       'adopted': int(used.sum())},
    }

    # what the exhibit shows
    sel = []
    for s in SELECTIONS:
        r = study.selection({'dates': dates, 'vix': vix, 'vxv': vxv, 'rvf': rvf}, s['year_from'], s['year_to'],
                            s['offset'], s['cost'], s['mode'], s['level'], s['notional'])
        sel.append({'state': s, 'result': r})
    out['selection'] = sel

    # today
    out['today'] = today.reading(live, vix, vxv, rvb, rvf, cost)
    return out
