"""Runs the whole study and returns one dictionary: every number on the site.

`run()` is what examples/make_results.py calls. It is written as small named
steps so each can be read (and tested) on its own. Nothing here writes files.

Conventions used throughout:
    * P&L is per 1 unit of vega notional, in "vega points" (multiply by dollars).
    * A month is a 21-trading-day cycle. Headline figures use the cycles that
      start at day 0 of the table (non-overlapping). The other 20 possible start
      offsets are reported as a check, because the answer must not depend on
      which day of the month you happened to start.
    * `cost` is the vol points taken off the strike when selling (see swap.py).
"""

import numpy as np

from . import BLOCK, HORIZON, PERIODS, REPS, SEED, conditions, filters, rv, stats, swap

COST = swap.DEFAULT_COST
ACCOUNT = 100_000.0           # example account for the sizing arithmetic
BUDGET = 0.05                 # the loss allowed in the worst month, as a share of the account
QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# Dates I am certain of (market-moving days). A loss window is named after one
# only when the event falls inside the 21 days it covers.
EVENTS = {
    '1997-10-27': 'asia_1997', '1998-08-31': 'russia_1998', '2001-09-17': 'sept_2001',
    '2008-09-29': 'lehman_2008', '2008-10-15': 'lehman_2008', '2010-05-06': 'flash_2010',
    '2011-08-08': 'downgrade_2011', '2015-08-24': 'china_2015', '2018-02-05': 'volmageddon_2018',
    '2020-03-16': 'covid_2020', '2025-04-04': 'tariffs_2025',
}


def summ(x, dates=None):
    """The standard summary of a monthly P&L series."""
    x = np.asarray(x, dtype=float)
    dd = stats.drawdown(x)
    out = {
        'n': int(len(x)), 'sharpe': stats.sharpe(x), 'mean': stats.mean(x), 'median': stats.quantile(x, 0.5),
        'sd': stats.sd(x), 'hit': stats.hit_rate(x), 'worst': float(x.min()), 'best': float(x.max()),
        'skew': stats.skew(x), 'es5': stats.shortfall(x, 0.05), 'total': float(x.sum()),
        'mean_over_worst': float(abs(x.min()) / x.mean()) if x.mean() > 0 else None,
        'dd': dd,
    }
    if dates is not None and dd['depth'] > 0:
        out['dd']['peak_date'] = str(dates[dd['peak']]) if dd['peak'] >= 0 else None
        out['dd']['trough_date'] = str(dates[dd['trough']])
    return out


def _ci(x):
    return {'sharpe_ci': stats.sharpe_ci(x), 'mean_ci': stats.mean_ci(x)}


def premium_block(vix, rv_fwd, dates):
    """Part 1: implied against realised, on every day and on non-overlapping months."""
    vix, rv_fwd = np.asarray(vix, dtype=float), np.asarray(rv_fwd, dtype=float)
    ok = np.isfinite(vix) & np.isfinite(rv_fwd)
    prem = (vix - rv_fwd)[ok]
    c = swap.cycles(vix, rv_fwd, 0)
    mp = c.strike - c.realised

    def block(p):
        return {'n': int(len(p)), 'mean': stats.mean(p), 'median': stats.quantile(p, 0.5),
                'pos_share': stats.hit_rate(p),
                'pct': {str(q): stats.quantile(p, q / 100) for q in QUANTILES},
                'min': float(p.min()), 'max': float(p.max()), 'es5': stats.shortfall(p, 0.05),
                'below_minus10': float((p < -10).sum() / len(p))}

    out = {'daily': block(prem), 'monthly': block(mp)}
    out['monthly']['mean_ci'] = stats.mean_ci(mp)
    # cross-check with an independent route: every day, 126-day blocks
    idx = stats.bootstrap_indices(len(prem), 126, 1000, SEED)
    out['daily']['mean_ci_block126'] = stats.boot_ci(prem[idx].sum(axis=1) / len(prem))
    out['mean_vix'] = float(vix[ok].mean())
    out['mean_rv'] = float(rv_fwd[ok].mean())
    out['mean_variance_gap'] = float((vix[ok] ** 2 - rv_fwd[ok] ** 2).mean())
    return out


def offsets_block(vix, rv_fwd, cost):
    """All 21 possible start days of the monthly cycle."""
    rows = []
    for off in range(HORIZON):
        p = swap.cycles(vix, rv_fwd, off, cost=cost).pnl
        rows.append({'offset': off, 'n': int(len(p)), 'sharpe': stats.sharpe(p), 'mean': stats.mean(p),
                     'hit': stats.hit_rate(p), 'worst': float(p.min()), 'skew': stats.skew(p),
                     'dd': stats.drawdown(p)['depth']})
    col = lambda k: np.array([r[k] for r in rows])
    return {'rows': rows,
            'sharpe': {'min': float(col('sharpe').min()), 'median': float(np.median(col('sharpe'))),
                       'max': float(col('sharpe').max())},
            'mean': {'min': float(col('mean').min()), 'median': float(np.median(col('mean'))),
                     'max': float(col('mean').max())},
            'worst': {'min': float(col('worst').min()), 'median': float(np.median(col('worst'))),
                      'max': float(col('worst').max())},
            'dd': {'min': float(col('dd').min()), 'median': float(np.median(col('dd'))),
                   'max': float(col('dd').max())}}


def worst_episodes(pnl_days, dates, vix, rv_fwd, top=10, separation=42):
    """The worst distinct loss episodes among windows opened on any day.

    Greedy: take the worst window, discard every window that starts within
    `separation` trading days of it (they are the same episode seen from a day
    earlier or later), take the next worst, and so on.
    """
    p = np.array(pnl_days, dtype=float)
    p[~np.isfinite(p)] = np.inf
    out = []
    for _ in range(top):
        i = int(np.argmin(p))
        if not np.isfinite(p[i]):
            break
        end = min(i + HORIZON, len(dates) - 1)
        name = None
        for k in range(i + 1, end + 1):
            name = EVENTS.get(str(dates[k])) or name
        out.append({'start': str(dates[i]), 'end': str(dates[end]), 'vix': float(vix[i]),
                    'rv': float(rv_fwd[i]), 'pnl': float(p[i]), 'event': name})
        p[max(0, i - separation): i + separation + 1] = np.inf
    return out


def subperiods(c, dates):
    years = np.array([int(str(dates[i])[:4]) for i in c.idx])
    spans = {'1990-1999': (1990, 1999), '2000-2009': (2000, 2009), '2010-2019': (2010, 2019), '2020-2026': (2020, 2026)}
    out = {}
    for name, (a, b) in spans.items():
        p = c.pnl[(years >= a) & (years <= b)]
        out[name] = {'n': int(len(p)), 'sharpe': stats.sharpe(p), 'mean': stats.mean(p), 'hit': stats.hit_rate(p),
                     'worst': float(p.min())}
    return out


def by_year(c, dates):
    years = np.array([int(str(dates[i])[:4]) for i in c.idx])
    return {int(y): float(c.pnl[years == y].sum()) for y in sorted(set(years))}


def break_tests(c, dates):
    """Try to break the headline: drop the worst months, drop the best or worst year."""
    p = c.pnl
    order = np.argsort(p)
    yrs = by_year(c, dates)
    years = np.array([int(str(dates[i])[:4]) for i in c.idx])
    best_y = max(yrs, key=yrs.get)
    worst_y = min(yrs, key=yrs.get)

    def s(mask):
        q = p[mask]
        return {'n': int(len(q)), 'sharpe': stats.sharpe(q), 'mean': stats.mean(q), 'worst': float(q.min())}

    keep = lambda drop: np.setdiff1d(np.arange(len(p)), drop)
    return {
        'all': s(np.ones(len(p), dtype=bool)),
        'without_worst_1': s(keep(order[:1])),
        'without_worst_3': s(keep(order[:3])),
        'without_worst_5pct': s(keep(order[:int(np.ceil(0.05 * len(p)))])),
        'since_2000': s(years >= 2000), 'since_2010': s(years >= 2010),
        'best_year': best_y, 'without_best_year': s(years != best_y),
        'worst_year': worst_y, 'without_worst_year': s(years != worst_y),
    }


def horizon_block(spx, vix, offset=0, cost=COST):
    """Realised volatility over 19 to 23 days instead of 21 (needs the closes)."""
    out = {}
    for h in (19, 20, 21, 22, 23):
        f = rv.fwd(spx, h)
        c = swap.cycles(vix, f, offset, horizon=h, cost=cost)
        out[str(h)] = {'n': len(c), 'sharpe': stats.sharpe(c.pnl), 'mean': stats.mean(c.pnl), 'worst': float(c.pnl.min())}
    return out


def conditions_block(c, feats, day_pnl, dates, vix, rv_fwd, cost, all_offsets):
    """Part 3: Sharpe, worst month and bootstrap ranges in each group."""
    out = {}
    valid_day = np.isfinite(day_pnl)
    for fam, (feat, edges, names) in conditions.FAMILIES.items():
        f = feats[feat]
        if edges is None:
            edges = conditions.tercile_edges(f[valid_day])
        day_lab = conditions.labels(f, edges)
        lab = day_lab[c.idx]
        groups = {}
        for g, name in enumerate(names):
            mask = lab == g
            p = c.pnl[mask]
            if len(p) == 0:
                groups[name] = {'n': 0}
                continue
            rest = lab >= 0
            rest &= ~mask
            ep = conditions.runs(c.j[mask])
            days = valid_day & (day_lab == g)
            row = {
                'n': int(len(p)), 'episodes': ep, 'sharpe': stats.sharpe(p), 'mean': stats.mean(p),
                'hit': stats.hit_rate(p), 'worst': float(p.min()), 'skew': stats.skew(p),
                'worst_any_start': float(day_pnl[days].min()) if days.any() else None,
                'days': int(days.sum()), 'mean_any_start': stats.mean(day_pnl[days]) if days.any() else None,
                'first': str(dates[c.idx[mask][0]]), 'last': str(dates[c.idx[mask][-1]]),
                'years': int(len(set(str(dates[i])[:4] for i in c.idx[mask]))),
            }
            known = lab >= 0
            if len(p) >= 8:
                row['sharpe_ci'] = stats.group_sharpe_ci(c.pnl[known], mask[known])
                if rest.sum() >= 8:
                    row['diff_ci'] = _diff_ci(c.pnl[known], mask[known], rest[known])
            # Sharpe of this group at each of the 21 start offsets
            sh = []
            for off in range(HORIZON):
                co = all_offsets[off]
                lo = day_lab[co.idx]
                q = co.pnl[lo == g]
                sh.append(stats.sharpe(q) if len(q) >= 8 else np.nan)
            sh = np.array(sh)
            row['sharpe_offsets'] = {'min': float(np.nanmin(sh)) if np.isfinite(sh).any() else None,
                                     'max': float(np.nanmax(sh)) if np.isfinite(sh).any() else None}
            row['thin'] = bool(len(p) < 30 or ep < 8)
            lo_, hi_ = row.get('diff_ci', [np.nan, np.nan])
            row['differs'] = bool(np.isfinite(lo_) and (lo_ > 0 or hi_ < 0))
            groups[name] = row
        out[fam] = {'edges': [float(e) for e in edges], 'groups': groups}
    return out


def _diff_ci(pnl, in_g, in_rest, block=BLOCK, reps=REPS, seed=SEED):
    """95% range for Sharpe(group) - Sharpe(all other months), same resamples."""
    idx = stats.bootstrap_indices(len(pnl), block, reps, seed)
    p = pnl[idx]
    a = stats._group_rows_sharpe(p, in_g[idx])
    b = stats._group_rows_sharpe(p, in_rest[idx])
    return stats.boot_ci(a - b)


def sizing_block(day_pnl_by_name):
    """What the worst month means for position size.

    `worst` is the worst window opened on ANY day (per 1 unit of vega). Sizing
    the position so that this worst window costs 5% of a 100,000 account gives the
    notional `n_star` in dollars per vol point, and the average yearly return that
    size would have earned.
    """
    out = {'account': ACCOUNT, 'budget': BUDGET}
    for name, d in day_pnl_by_name.items():
        d = d[np.isfinite(d)]
        worst = float(d.min())
        n_star = BUDGET * ACCOUNT / abs(worst)
        mean = float(d.mean())
        out[name] = {'worst': worst, 'mean': mean, 'n_star': n_star, 'month_at_n_star': mean * n_star,
                     'year_return_at_n_star': mean * PERIODS * n_star / ACCOUNT,
                     'windows': int(len(d))}
    return out


def filters_block(feats, vix, rv_fwd, dates, cost, c0):
    """Part 4: the term-structure filter and the walk-forward floor."""
    out = {'variants': dict(filters.VARIANTS)}
    out['variants']['groups_looked_at'] = sum(len(v[2]) for v in conditions.FAMILIES.values())
    out['variants']['total'] = filters.VARIANTS['term_structure'] + filters.VARIANTS['floor_grid']

    slope = feats['slope']

    # --- term structure: sell only when the curve is not inverted
    has = np.isfinite(slope[c0.idx])
    p = c0.pnl[has]
    sell = filters.sell_term(slope[c0.idx][has])
    f = filters.apply(p, sell)
    idxs = c0.idx[has]
    half = len(p) // 2
    out['term'] = {
        'first': str(dates[idxs[0]]), 'last': str(dates[idxs[-1]]), 'n': int(len(p)),
        'sold_share': float(sell.mean()), 'unfiltered': summ(p, [dates[i] for i in idxs]),
        'filtered': summ(f, [dates[i] for i in idxs]),
        'sharpe_diff_ci': stats.paired_diff_ci(f, p),
        'first_half': {'unfiltered': stats.sharpe(p[:half]), 'filtered': stats.sharpe(f[:half])},
        'second_half': {'unfiltered': stats.sharpe(p[half:]), 'filtered': stats.sharpe(f[half:])},
        'skipped_months': _skipped(p, sell, [dates[i] for i in idxs], c0.j[has]),
    }
    rows = []
    for off in range(HORIZON):
        co = swap.cycles(vix, rv_fwd, off, cost=cost)
        h = np.isfinite(slope[co.idx])
        q = co.pnl[h]
        fq = filters.apply(q, filters.sell_term(slope[co.idx][h]))
        rows.append({'offset': off, 'unfiltered': stats.sharpe(q), 'filtered': stats.sharpe(fq),
                     'worst_unfiltered': float(q.min()), 'worst_filtered': float(fq.min())})
    d = np.array([r['filtered'] - r['unfiltered'] for r in rows])
    out['term']['offsets'] = {'rows': rows, 'improved': int((d > 0).sum()),
                              'diff': {'min': float(d.min()), 'median': float(np.median(d)), 'max': float(d.max())},
                              'worst_filtered': {'min': float(min(r['worst_filtered'] for r in rows)),
                                                 'max': float(max(r['worst_filtered'] for r in rows))},
                              'worst_unfiltered': {'min': float(min(r['worst_unfiltered'] for r in rows)),
                                                   'max': float(max(r['worst_unfiltered'] for r in rows))}}
    oos_t, used, st = filters.walk_forward_adopt(p, sell)
    base_t = p[st:]
    dts_t = [dates[i] for i in idxs[st:]]
    out['term']['walk_forward'] = {
        'first': str(dts_t[0]), 'last': str(dts_t[-1]), 'n': int(len(base_t)),
        'unfiltered': summ(base_t, dts_t), 'adaptive': summ(oos_t, dts_t),
        'always_filtered': summ(f[st:], dts_t), 'adopted_share': float(used.mean()),
        'sharpe_diff_ci': stats.paired_diff_ci(oos_t, base_t)}

    # --- walk-forward floor
    oos, chosen, start = filters.walk_forward_floor(c0.strike, c0.pnl)
    base = c0.pnl[start:]
    dts = [dates[i] for i in c0.idx[start:]]
    share = {str(k): float((chosen == k).mean()) for k in sorted(set(chosen))}
    out['floor_walk'] = {
        'first': str(dts[0]), 'last': str(dts[-1]), 'n': int(len(base)),
        'unfiltered': summ(base, dts), 'filtered': summ(oos, dts),
        'sold_share': float((oos != 0).mean()), 'chosen_share': share,
        'sharpe_diff_ci': stats.paired_diff_ci(oos, base),
        'fixed_floor_in_sample': {str(fl): {'sharpe': stats.sharpe(filters.apply(c0.pnl, c0.strike >= fl)),
                                            'worst': float(filters.apply(c0.pnl, c0.strike >= fl).min()),
                                            'sold_share': float((c0.strike >= fl).mean())} for fl in filters.FLOOR_GRID},
    }
    return out


def _skipped(p, sell, dates, j):
    """The months the term-structure filter skipped, with their P&L."""
    k = np.where(~sell)[0]
    return {'n': int(len(k)), 'episodes': conditions.runs(np.asarray(j)[k]),
            'total': float(p[k].sum()) if len(k) else 0.0,
            'worst_avoided': float(p[k].min()) if len(k) else None,
            'best_missed': float(p[k].max()) if len(k) else None,
            'losses_over_20': int((p[k] < -20).sum()) if len(k) else 0,
            'loss_dates': [str(dates[i]) for i in k[p[k] < -20]] if len(k) else [],
            'wins': int((p[k] > 0).sum()) if len(k) else 0}


def named_windows(day_pnl, dates, vix, rv_fwd, vxv):
    """The worst window started in each of the months everyone asks about."""
    want = {'oct_2008': ('2008-09-01', '2008-10-31'), 'feb_2018': ('2018-01-02', '2018-02-28'),
            'mar_2020': ('2020-02-03', '2020-03-31'), 'aug_1998': ('1998-07-15', '1998-09-15')}
    out = {}
    d = np.asarray([str(x) for x in dates])
    for key, (a, b) in want.items():
        m = (d >= a) & (d <= b) & np.isfinite(day_pnl)
        i = int(np.where(m)[0][np.argmin(day_pnl[m])])
        out[key] = {'start': str(dates[i]), 'vix': float(vix[i]), 'rv': float(rv_fwd[i]), 'pnl': float(day_pnl[i]),
                    'slope': float(vxv[i] - vix[i]) if np.isfinite(vxv[i]) else None, 'window': [a, b]}
    return out


def run(table, fedfunds, usrec, live=None, cost=COST, fetch_log=None, tbill=None):
    """The whole study. `table` is data.build_table() (its `spx` column, if there,
    is only used for the horizon check); the rest is the monthly fed funds and NBER
    series and the live file."""
    from . import data, today
    dates = np.array([str(d.date()) for d in table.index])
    vix, vxv = table['vix'].values, table['vxv'].values
    rb, rf = table['rv_back'].values, table['rv_fwd'].values

    ff2 = data.monthly_lookup(fedfunds, table.index, conditions.RATE_LAG[0])
    ff14 = data.monthly_lookup(fedfunds, table.index, conditions.RATE_LAG[1])
    rec = data.monthly_lookup(usrec, table.index, 0)       # hindsight: the month itself
    feats = conditions.features(vix, vxv, rb, ff2, ff14, rec)

    c0 = swap.cycles(vix, rf, 0, cost=cost)
    g0 = swap.cycles(vix, rf, 0, cost=0.0)
    all_off = [swap.cycles(vix, rf, o, cost=cost) for o in range(HORIZON)]
    day = swap.daily_windows(vix, rf, cost)
    cd = [dates[i] for i in c0.idx]

    res = {'meta': {
        'first': dates[0], 'last': dates[-1], 'n_days': int(len(dates)),
        'last_vix': str(table['vix'].dropna().index[-1].date()),
        'first_vxv': str(table['vxv'].dropna().index[0].date()),
        'last_cycle_start': cd[-1], 'first_cycle_start': cd[0], 'cycles': int(len(c0)),
        'horizon': HORIZON, 'cost': cost, 'seed': SEED, 'block': BLOCK, 'reps': REPS,
        'days_without_vix': int(np.isnan(vix).sum()), 'fetch': fetch_log,
    }}
    res['premium'] = premium_block(vix, rf, dates)

    swp = {'cost': cost}
    swp['net'] = summ(c0.pnl, cd)
    swp['net'].update(_ci(c0.pnl))
    swp['gross'] = summ(g0.pnl, [dates[i] for i in g0.idx])
    swp['gross'].update(_ci(g0.pnl))
    st = swap.cycles(vix, rf, 0, cost=2.0)
    swp['stress'] = summ(st.pnl, [dates[i] for i in st.idx])
    swp['stress'].update(_ci(st.pnl))
    swp['offsets'] = offsets_block(vix, rf, cost)
    swp['offsets_gross'] = offsets_block(vix, rf, 0.0)
    swp['cost_sensitivity'] = {}
    for cc in (0.0, 0.5, 1.0, 2.0):
        sh = [stats.sharpe(swap.cycles(vix, rf, o, cost=cc).pnl) for o in range(HORIZON)]
        swp['cost_sensitivity'][str(cc)] = {'offset0': sh[0], 'median': float(np.median(sh))}
    swp['block_sensitivity'] = {str(b): stats.sharpe_ci(c0.pnl, block=b) for b in (3, 6, 12)}
    swp['worst_months'] = [{'start': cd[i], 'end': str(dates[min(c0.idx[i] + HORIZON, len(dates) - 1)]),
                            'vix': float(c0.strike[i]), 'rv': float(c0.realised[i]), 'pnl': float(c0.pnl[i])}
                           for i in np.argsort(c0.pnl)[:10]]
    swp['worst_episodes'] = worst_episodes(day, dates, vix, rf)
    swp['named'] = named_windows(day, dates, vix, rf, vxv)
    swp['subperiods'] = subperiods(c0, dates)
    swp['by_year'] = by_year(c0, dates)
    swp['break'] = break_tests(c0, dates)
    if 'spx' in table:
        swp['horizon'] = horizon_block(table['spx'].values, vix, 0, cost)
    swp['loss_over_25'] = int((c0.pnl < -25).sum())
    res['swap'] = swp

    res['conditions'] = conditions_block(c0, feats, day, dates, vix, rf, cost, all_off)
    res['filters'] = filters_block(feats, vix, rf, dates, cost, c0)

    # sizing: windows opened on any day, per 1 unit of vega
    slope = feats['slope']
    okd = np.isfinite(day) & np.isfinite(slope)
    d_all = day[okd]
    d_term = filters.apply(day[okd], filters.sell_term(slope[okd]))
    res['sizing'] = sizing_block({'unfiltered': day[np.isfinite(day)], 'unfiltered_term_period': d_all,
                                  'term_filter': d_term})
    res['sizing']['sizing_note'] = 'worst window opened on any day; filtered windows are zero when flat'

    # the reading for the latest data, from the file the page loads
    if live is None:
        live = today.live_payload(data.read_fred('SP500'), data.read_fred('VIXCLS'), data.read_fred('VXVCLS'),
                                  generated=(fetch_log or {}).get('fetched_utc', ''))
    hist_end = np.isfinite(rf)
    res['state'] = today.reading(live, vix, vxv, rb, rf, cost)
    res['state']['live_last'] = live['dates'][-1]
    res['state']['live_as_of'] = live['asOf']
    res['state']['hist_last_complete'] = dates[np.where(hist_end)[0][-1]]
    if tbill is not None:
        import pandas as pd
        when = pd.DatetimeIndex([pd.Timestamp(live['asOf'])])
        a = data.monthly_lookup(fedfunds, when, conditions.RATE_LAG[0])[0]
        b = data.monthly_lookup(fedfunds, when, conditions.RATE_LAG[1])[0]
        tb = tbill.dropna()
        ff = fedfunds.dropna()
        change = a - b
        res['state']['rates'] = {
            'tbill': float(tb.iloc[-1]), 'tbill_date': str(tb.index[-1].date()),
            'fedfunds': float(ff.iloc[-1]), 'fedfunds_month': str(ff.index[-1].date())[:7],
            'ff12': float(change), 'phase': ['falling', 'flat', 'rising'][int(conditions.labels([change], [-0.25, 0.25])[0])]}
    return res, live



MODES = ('all', 'floor', 'cap', 'curve')


def selection(hist, year_from, year_to, offset, cost, mode, level, notional, account=ACCOUNT):
    """What the page's exhibit computes, in Python (the reference for the JavaScript).

    `hist` is a dict of arrays: dates (ISO text), vix, vxv, rvf. The 21-day grid is
    anchored to the first day of the table, so moving the year range never moves
    the cycles. Mode: `all` sells every month; `floor` sells only when the VIX is
    at or above `level`; `cap` only when it is below; `curve` only when the 3-month
    VIX is not below the 1-month VIX, and it then looks only at months where the
    3-month VIX exists, for the baseline as well.
    """
    dates = np.asarray(hist['dates'])
    vix, vxv, rvf = (np.asarray(hist[k], dtype=float) for k in ('vix', 'vxv', 'rvf'))
    years = np.array([int(d[:4]) for d in dates])
    slope = vxv - vix
    c = swap.cycles(vix, rvf, offset, cost=cost)
    keep = (years[c.idx] >= year_from) & (years[c.idx] <= year_to)
    if mode == 'curve':
        keep &= np.isfinite(slope[c.idx])
    idx, pnl, strike = c.idx[keep], c.pnl[keep], c.strike[keep]
    sell = _sell(mode, level, strike, slope[idx])
    filt = filters.apply(pnl, sell)
    day = swap.daily_windows(vix, rvf, cost)
    dmask = np.isfinite(day) & (years >= year_from) & (years <= year_to)
    if mode == 'curve':
        dmask &= np.isfinite(slope)
    dsell = _sell(mode, level, vix, slope)
    dfilt = np.where(dsell, day, 0.0)[dmask]
    worst_any = float(dfilt.min()) if dmask.any() else None
    s = summ(filt, [dates[i] for i in idx])
    out = {'n': int(len(idx)), 'sold': int(sell.sum()), 'sharpe': s['sharpe'], 'mean': s['mean'],
           'hit': stats.hit_rate(pnl[sell]) if sell.any() else float('nan'),
           'worst': s['worst'], 'skew': s['skew'], 'dd': s['dd']['depth'], 'total': s['total'],
           'recovery': s['dd']['recovery_periods'], 'longest_under': s['dd']['longest_under'],
           'sharpe_ci': stats.sharpe_ci(filt), 'base_sharpe': stats.sharpe(pnl), 'base_worst': float(pnl.min()),
           'worst_any': worst_any,
           'worst_usd': s['worst'] * notional, 'dd_usd': s['dd']['depth'] * notional,
           'mean_usd': s['mean'] * notional, 'total_usd': s['total'] * notional}
    out['worst_pct'] = out['worst_usd'] / account
    out['n_star'] = BUDGET * account / abs(worst_any) if worst_any else None
    return out


def _sell(mode, level, strike, slope):
    strike = np.asarray(strike, dtype=float)
    if mode == 'floor':
        return strike >= level
    if mode == 'cap':
        return strike < level
    if mode == 'curve':
        return np.asarray(slope, dtype=float) >= 0.0
    return np.ones(len(strike), dtype=bool)
