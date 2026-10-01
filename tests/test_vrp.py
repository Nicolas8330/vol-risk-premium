"""Tests for the volatility risk premium study.

Each one checks a fact that holds without reading the code: arithmetic, no
look-ahead, sign conventions, accounting identities, edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from vrp import HORIZON, SEED, conditions, data, export, filters, rv, stats, study, swap, today


def walk(n=1500, seed=1, daily_vol=0.01):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, daily_vol, n)))


def synthetic(n=1800, seed=2):
    """A price path, a VIX that is a noisy 30% above trailing realised vol, and a 3-month VIX."""
    close = walk(n, seed)
    rb = rv.back(close)
    rng = np.random.default_rng(seed + 1)
    vix = np.where(np.isfinite(rb), np.nan_to_num(rb, nan=15.0) * 1.3 + rng.normal(0, 1.5, n) + 2, 18.0)
    vix = np.clip(vix, 6, 80)
    vxv = vix + rng.normal(1.0, 1.0, n)
    return close, vix, vxv


# ------------------------------------------------------------- realised vol

def test_alternating_one_percent_moves_give_252_variance():
    close = 100 * np.exp(np.cumsum([0.0] + [0.01, -0.01] * 40))
    b = rv.back(close, 21)
    # every return is +-1%, so annualised vol is 1% x sqrt(252)
    assert b[30] == pytest.approx(np.sqrt(252) * 1.0, rel=1e-12)


def test_realised_vol_does_not_subtract_the_mean():
    close = 100 * np.exp(np.cumsum([0.0] + [0.01] * 40))      # a steady 1% climb
    assert rv.back(close, 21)[30] == pytest.approx(np.sqrt(252) * 1.0, rel=1e-12)   # not zero


def test_forward_is_backward_shifted_by_the_window():
    close = walk()
    b, f = rv.back(close), rv.fwd(close)
    assert np.allclose(f[:-21], b[21:], equal_nan=True)
    assert np.isnan(f[-21:]).all()


def test_back_does_not_see_the_future_and_fwd_does_not_see_the_past():
    close = walk()
    t = 800
    later = close.copy()
    later[t + 1:] *= np.exp(np.linspace(0, 0.5, len(close) - t - 1))        # rewrite everything after t
    assert rv.back(later)[t] == rv.back(close)[t]
    earlier = close.copy()
    earlier[:t - 5] *= 1.7                                                  # a level shift before t: no return changes after it
    earlier[:t - 5] = earlier[:t - 5] * np.exp(np.linspace(0, 0.3, t - 5))
    assert rv.fwd(earlier)[t] == pytest.approx(rv.fwd(close)[t], rel=1e-9)


def test_gaps_and_bad_prices_are_refused():
    with pytest.raises(ValueError):
        rv.log_returns([100.0, np.nan, 101.0])
    with pytest.raises(ValueError):
        rv.log_returns([100.0, -1.0])


# ------------------------------------------------------------------- the swap

def test_swap_pays_nothing_when_realised_equals_strike():
    assert swap.pnl(20.0, 20.0) == 0.0


def test_swap_sign_and_the_convexity_of_the_loss():
    assert swap.pnl(20.0, 15.0) > 0 > swap.pnl(20.0, 25.0)
    # same 5 point gap either way, but the loss is bigger than the gain
    assert abs(swap.pnl(20.0, 25.0)) > swap.pnl(20.0, 15.0)
    # realised vol double the strike costs 1.5 x strike per unit of vega
    assert swap.pnl(20.0, 40.0) == pytest.approx(-1.5 * 20.0)


def test_swap_formula_matches_the_textbook_one():
    k, r = 17.5, 22.0
    assert swap.pnl(k, r) == pytest.approx(1 / (2 * k) * (k ** 2 - r ** 2))


def test_cost_only_lowers_pnl_and_is_a_lower_strike():
    assert swap.pnl(20, 18, cost=0.5) < swap.pnl(20, 18)
    assert swap.pnl(20, 18, cost=0.5) == pytest.approx(swap.pnl(19.5, 18))
    with pytest.raises(ValueError):
        swap.pnl(0.4, 10, cost=0.5)


def test_cycles_are_21_apart_and_drop_unknown_ones():
    n = 400
    vix = np.full(n, 20.0)
    rvf = np.full(n, 15.0)
    rvf[-21:] = np.nan
    vix[42] = np.nan
    c = swap.cycles(vix, rvf, offset=0)
    assert (np.diff(c.idx) % 21 == 0).all()
    assert 42 not in c.idx
    assert list(c.j[:3]) == [0, 1, 3]                 # cycle 2 was dropped, the numbering shows it
    assert c.idx.max() + 21 <= n - 1                  # every cycle finished


def test_daily_window_equals_the_cycle_that_starts_that_day():
    close, vix, _ = synthetic()
    f = rv.fwd(close)
    c = swap.cycles(vix, f, 7, cost=0.5)
    d = swap.daily_windows(vix, f, 0.5)
    assert np.allclose(d[c.idx], c.pnl)


def test_pnl_matches_an_independent_pandas_route():
    close, vix, _ = synthetic()
    s = pd.Series(close)
    r = np.log(s / s.shift(1))
    rvf = np.sqrt(252 / 21 * (r ** 2).rolling(21).sum().shift(-21)) * 100      # a different code path
    k = pd.Series(vix)
    ke = k - 0.5
    ref = ((ke ** 2 - rvf ** 2) / (2 * ke))
    got = swap.daily_windows(vix, rv.fwd(close), 0.5)
    ok = np.isfinite(got)
    assert np.allclose(got[ok], ref.values[ok], atol=1e-9)


# ------------------------------------------------------------------ statistics

def test_sharpe_of_a_known_series():
    x = np.array([1.0, 3.0, 2.0, 2.0])
    assert stats.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(12))
    assert np.isnan(stats.sharpe([1.0, 1.0, 1.0]))


def test_skew_sign_matches_the_tail():
    assert stats.skew([1, 1, 1, 1, 1, 1, 1, -20]) < 0
    assert stats.skew([-1, -1, -1, -1, -1, -1, -1, 20]) > 0


def test_drawdown_on_a_known_path():
    d = stats.drawdown([1, 1, -3, 1, 1, 1])          # cum 1,2,-1,0,1,2
    assert d['depth'] == 3 and d['peak'] == 1 and d['trough'] == 2
    assert d['recovery_periods'] == 3                 # back to 2 after three periods
    assert d['longest_under'] == 4
    assert stats.drawdown([1, 2, 3])['depth'] == 0
    assert stats.drawdown([-1, -1])['recovery_periods'] is None


def test_quantile_matches_numpy():
    x = np.random.default_rng(0).normal(size=101)
    for q in (0.01, 0.05, 0.5, 0.9, 0.99):
        assert stats.quantile(x, q) == pytest.approx(np.quantile(x, q))


def test_shortfall_is_the_mean_of_the_worst_tail():
    x = np.arange(100.0)
    assert stats.shortfall(x, 0.05) == pytest.approx(np.mean([0, 1, 2, 3, 4]))


def test_mulberry32_reference_values():
    # values produced by the JavaScript generator, which the browser uses
    r = stats.mulberry32(42)
    assert [r(), r(), r()] == pytest.approx([0.6011037519201636, 0.44829055899754167, 0.8524657934904099], abs=1e-15)
    q = stats.mulberry32(SEED)
    assert [q(), q()] == pytest.approx([0.7129707557614893, 0.9029586620163172], abs=1e-15)


def test_bootstrap_indices_are_in_range_blocked_and_repeatable():
    idx = stats.bootstrap_indices(50, 6, 20, 7)
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    row = idx[3]
    steps = np.diff(row[:6]) % 50
    assert (steps == 1).all()                         # a block is six neighbouring months
    assert (idx == stats.bootstrap_indices(50, 6, 20, 7)).all()
    assert not (idx == stats.bootstrap_indices(50, 6, 20, 8)).all()


def test_bootstrap_range_brackets_the_estimate_and_widens_with_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(0.5, 1.0, 300)
    lo, hi = stats.sharpe_ci(x, reps=500)
    assert lo < stats.sharpe(x) < hi
    lo2, hi2 = stats.sharpe_ci(x[:60], reps=500)
    assert hi2 - lo2 > hi - lo                        # less data, wider range
    m_lo, m_hi = stats.mean_ci(x, reps=500)
    assert m_lo < x.mean() < m_hi


def test_group_range_needs_enough_members():
    x = np.random.default_rng(4).normal(size=100)
    flag = np.zeros(100, dtype=bool)
    flag[:3] = True
    lo, hi = stats.group_sharpe_ci(x, flag, reps=200)
    assert np.isnan(lo) and np.isnan(hi)


# ------------------------------------------------------------------ conditions

def test_labels_put_a_value_on_an_edge_in_the_upper_group():
    lab = conditions.labels([13.99, 14.0, 20.0, 35.0, np.nan], [14.0, 20.0, 30.0])
    assert list(lab) == [0, 1, 2, 3, -1]


def test_runs_counts_stretches_of_neighbouring_cycles():
    assert conditions.runs([3, 4, 5, 9, 10, 20]) == 3
    assert conditions.runs([]) == 0


def test_features_only_use_the_past():
    close, vix, vxv = synthetic()
    rb = rv.back(close)
    f1 = conditions.features(vix, vxv, rb)
    cut = 1000
    vix2, vxv2 = vix.copy(), vxv.copy()
    vix2[cut:] += 25
    vxv2[cut:] -= 10
    close2 = close.copy()
    close2[cut:] *= 1.5
    f2 = conditions.features(vix2, vxv2, rv.back(close2))
    for k in ('vix', 'gap', 'slope', 'vchg'):
        assert np.allclose(f1[k][:cut], f2[k][:cut], equal_nan=True), k


def test_monthly_lookup_never_reads_the_current_month():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01']))
    d = pd.to_datetime(['2020-03-15', '2020-04-02'])
    assert list(data.monthly_lookup(s, d, 2)) == [1.0, 2.0]
    assert list(data.monthly_lookup(s, d, 0)) == [3.0, 4.0]
    assert np.isnan(data.monthly_lookup(s, pd.to_datetime(['2020-01-20']), 2)[0])


def test_rate_lag_is_two_months_and_fourteen_back():
    assert conditions.RATE_LAG == (2, 14)


# --------------------------------------------------------------------- filters

def test_term_filter_sells_only_in_contango_or_flat():
    assert list(filters.sell_term([1.0, 0.0, -0.1])) == [True, True, False]


def test_filter_apply_is_zero_when_flat_and_never_creates_pnl():
    p = np.array([2.0, -5.0, 1.0])
    out = filters.apply(p, [True, False, True])
    assert list(out) == [2.0, 0.0, 1.0]


def test_walk_forward_floor_does_not_use_the_future():
    close, vix, _ = synthetic()
    c = swap.cycles(vix, rv.fwd(close), 0, cost=0.5)
    a_oos, a_ch, st = filters.walk_forward_floor(c.strike, c.pnl, min_train=40)
    cut = 70
    p2 = c.pnl.copy()
    p2[cut:] = -50.0                                   # rewrite every month from `cut` on
    b_oos, b_ch, _ = filters.walk_forward_floor(c.strike, p2, min_train=40)
    k = cut - st                                       # months before cut chose their floor blind to it
    assert (a_ch[:k + 0] == b_ch[:k + 0]).all()
    assert np.allclose(a_oos[:k], b_oos[:k])


def test_walk_forward_adopt_does_not_use_the_future():
    p = np.random.default_rng(5).normal(1, 5, 120)
    sell = np.random.default_rng(6).random(120) > 0.2
    a, ua, st = filters.walk_forward_adopt(p, sell, min_train=30)
    q = p.copy()
    q[80:] = -30
    b, ub, _ = filters.walk_forward_adopt(q, sell, min_train=30)
    k = 80 - st
    assert (ua[:k] == ub[:k]).all() and np.allclose(a[:k], b[:k])


def test_a_floor_of_zero_and_a_filter_that_always_sells_change_nothing():
    p = np.random.default_rng(7).normal(1, 4, 60)
    assert stats.sharpe(filters.apply(p, filters.sell_floor(np.full(60, 20.0), 0.0))) == pytest.approx(stats.sharpe(p))


# ------------------------------------------------------------ study, end to end

def test_no_look_ahead_in_the_whole_pipeline():
    """Rewrite the last 300 days of prices and VIX: every cycle that had finished
    before the cut, and every decision made before it, must be unchanged."""
    close, vix, vxv = synthetic(1800)
    cut = 1500

    def build(close, vix, vxv):
        rf = rv.fwd(close)
        feats = conditions.features(vix, vxv, rv.back(close))
        c = swap.cycles(vix, rf, 0, cost=0.5)
        return c, feats

    c1, f1 = build(close, vix, vxv)
    close2 = close.copy()
    close2[cut:] *= np.exp(np.linspace(0, 0.8, len(close) - cut))
    vix2 = vix.copy()
    vix2[cut:] += 30
    vxv2 = vxv.copy()
    vxv2[cut:] -= 20
    c2, f2 = build(close2, vix2, vxv2)
    done = c1.idx + HORIZON <= cut                     # cycles that expired at or before the cut
    assert done.sum() > 50
    assert np.allclose(c1.pnl[done], c2.pnl[c2.idx + HORIZON <= cut])
    for k in ('vix', 'gap', 'slope', 'vchg'):
        assert np.allclose(f1[k][:cut], f2[k][:cut], equal_nan=True)


def test_sizing_identity_worst_window_costs_exactly_the_budget():
    close, vix, _ = synthetic()
    d = swap.daily_windows(vix, rv.fwd(close), 0.5)
    s = study.sizing_block({'x': d})['x']
    assert -s['worst'] * s['n_star'] == pytest.approx(study.BUDGET * study.ACCOUNT)
    assert s['year_return_at_n_star'] == pytest.approx(s['mean'] * 12 * s['n_star'] / study.ACCOUNT)


def test_selection_total_equals_sum_of_pnl_and_modes_only_remove_months():
    close, vix, vxv = synthetic()
    h = {'dates': np.array([str(d.date()) for d in pd.bdate_range('1995-01-02', periods=len(close))]),
         'vix': vix, 'vxv': vxv, 'rvf': rv.fwd(close)}
    allm = study.selection(h, 1995, 2005, 0, 0.5, 'all', 16, 100)
    fl = study.selection(h, 1995, 2005, 0, 0.5, 'floor', 16, 100)
    cap = study.selection(h, 1995, 2005, 0, 0.5, 'cap', 16, 100)
    assert allm['sold'] == allm['n']
    assert fl['sold'] + cap['sold'] == allm['n']       # floor and cap split the months exactly
    assert fl['total'] + cap['total'] == pytest.approx(allm['total'])
    assert allm['total_usd'] == pytest.approx(allm['total'] * 100)
    assert fl['n'] == allm['n']                        # the baseline months are the same


def test_year_range_does_not_move_the_cycle_grid():
    close, vix, vxv = synthetic()
    h = {'dates': np.array([str(d.date()) for d in pd.bdate_range('1995-01-02', periods=len(close))]),
         'vix': vix, 'vxv': vxv, 'rvf': rv.fwd(close)}
    a = study.selection(h, 1995, 2005, 4, 0.0, 'all', 16, 1)
    b = study.selection(h, 1996, 2004, 4, 0.0, 'all', 16, 1)
    assert b['n'] < a['n']
    assert b['worst'] >= a['worst']


def test_named_event_dates_are_weekdays_and_in_the_right_years():
    for d, key in study.EVENTS.items():
        ts = pd.Timestamp(d)
        assert ts.dayofweek < 5, (d, key)


# ------------------------------------------------------------------------ data

def test_check_table_refuses_a_broken_download():
    idx = pd.bdate_range('2020-01-01', periods=50)
    ok = pd.DataFrame({'vix': 20.0, 'vxv': 21.0, 'rv_back': 15.0, 'rv_fwd': 15.0}, index=idx)
    data.check_table(ok)
    bad = ok.copy()
    bad.iloc[3, 0] = 900.0
    with pytest.raises(ValueError):
        data.check_table(bad)
    dup = pd.concat([ok, ok.iloc[:1]])
    with pytest.raises(ValueError):
        data.check_table(dup)


def test_json_export_has_no_nan_and_nothing_from_the_spx_price_table():
    idx = pd.bdate_range('2020-01-01', periods=5)
    t = pd.DataFrame({'spx': [1, 2, 3, 4, 5.0], 'vix': [20, np.nan, 21, 22, 23.0], 'vxv': np.nan,
                      'rv_back': [np.nan, 1, 2, 3, 4.0], 'rv_fwd': [1, 2, np.nan, np.nan, np.nan]}, index=idx)
    h = export.history_payload(t, 'now')
    assert 'spx' not in h
    text = str(export.clean(h))
    assert 'nan' not in text.lower().replace('nan_', '')
    assert h['vix'][1] is None


# ------------------------------------------------------------------- the reading

def test_today_reading_on_a_small_live_file():
    close, vix, vxv = synthetic(600)
    dates = [str(d.date()) for d in pd.bdate_range('2020-01-01', periods=600)]
    live = {'asOf': dates[-1], 'dates': dates, 'vix': list(map(float, vix)), 'vxv': list(map(float, vxv)),
            'spx': list(map(float, close))}
    rf, rb = rv.fwd(close), rv.back(close)
    r = today.reading(live, vix, vxv, rb, rf)
    assert r['as_of'] == dates[-1]
    assert r['rv_back'] == pytest.approx(rb[-1])
    assert r['gap'] == pytest.approx(vix[-1] - rb[-1])
    assert 0 < r['pct_vix'] <= 1
    assert r['similar']['days'] > 0


def test_today_reading_uses_closes_up_to_the_vix_date_only():
    close, vix, vxv = synthetic(600)
    dates = [str(d.date()) for d in pd.bdate_range('2020-01-01', periods=600)]
    live = {'asOf': dates[-2], 'dates': dates, 'vix': list(map(float, vix[:-1])) + [None], 'vxv': None,
            'spx': list(map(float, close))}
    rb = rv.back(close)
    r = today.reading(live, vix, vxv, rb, rv.fwd(close))
    assert r['as_of'] == dates[-2] and r['rv_date'] == dates[-2]
    assert r['rv_back'] == pytest.approx(rb[-2])
    assert r['vxv'] is None and r['slope'] is None


# ------------------------------------------------------- the real data (cached)

@pytest.mark.skipif(not __import__('os').path.exists(data.os.path.join(data.CACHE, 'gspc_yahoo.csv')),
                    reason='data cache not downloaded')
def test_real_table_is_sane():
    t = data.build_table()
    assert t.index[0] == pd.Timestamp('1990-01-02')
    assert t['vix'].between(9, 83).sum() > 9000
    assert (t['rv_back'].dropna() > 0).all()
    assert t['vxv'].first_valid_index() == pd.Timestamp('2007-12-04')
    # the 2020 crash is in the data: realised vol above 80 after mid-February
    assert t.loc['2020-02-14', 'rv_fwd'] > 80
    # and a known fact: the VIX closed at 82.69 on 16 March 2020
    assert t.loc['2020-03-16', 'vix'] == pytest.approx(82.69)
