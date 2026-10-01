"""Write README.md with every number taken from data/results.json.

    python examples/make_readme.py

The README quotes numbers, so it is generated from the same file as the site rather
than typed by hand.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, 'data', 'results.json')))
P, S, N, G, X = r['premium'], r['swap'], r['swap']['net'], r['swap']['gross'], r['swap']['stress']
F, Z, M, O = r['filters'], r['sizing'], r['meta'], r['swap']['offsets']
t = F['term']
we = S['worst_episodes'][0]


def f(x, d=2):
    return f'{x:.{d}f}'


def pc(x, d=0):
    return f'{100 * x:.{d}f}%'


def ci(c):
    return f'{f(c[0])} to {f(c[1])}'


TEXT = f'''# The volatility risk premium: getting paid to insure the market

The VIX is the market's price for the next month of S&P 500 movement. Most of the
time it is higher than the movement that follows. This repository tests what
happens if you sell that gap every month, and what it costs when it goes wrong.

Live version: [quant.orbiseo.fr/vol-risk-premium](https://quant.orbiseo.fr/vol-risk-premium/).
Research only. Nothing here was traded and nothing here is advice.

## The short answer

Selling the gap paid in most months and lost a lot in a few. Numbers are per 1 unit
of vega notional ("vega points": multiply by your dollars per vol point). One "month"
is a 21-trading-day cycle. Data: {M['first']} to {M['last']} ({M['cycles']} cycles).

| | No cost | Cost {S['cost']} vol point | Stress, cost 2 |
|---|---:|---:|---:|
| Sharpe ratio | {f(G['sharpe'])} | {f(N['sharpe'])} | {f(X['sharpe'])} |
| 95% range for the Sharpe | {ci(G['sharpe_ci'])} | {ci(N['sharpe_ci'])} | {ci(X['sharpe_ci'])} |
| Winning months | {pc(G['hit'], 1)} | {pc(N['hit'], 1)} | {pc(X['hit'], 1)} |
| Average month | {f(G['mean'])} | {f(N['mean'])} | {f(X['mean'])} |
| Worst month on this path | {f(G['worst'], 1)} | {f(N['worst'], 1)} | {f(X['worst'], 1)} |
| Skewness | {f(G['skew'], 1)} | {f(N['skew'], 1)} | {f(X['skew'], 1)} |
| Deepest fall from a peak | {f(G['dd']['depth'], 1)} | {f(N['dd']['depth'], 1)} | {f(X['dd']['depth'], 1)} |

* Implied against realised: the VIX was above the volatility of the next 21 days on
  {pc(P['daily']['pos_share'], 1)} of days, by {f(P['daily']['mean'])} points on average
  (95% range for non-overlapping months {ci(P['monthly']['mean_ci'])}).
* The start day matters. Over the 21 possible start days the Sharpe (cost {S['cost']}) runs from
  {f(O['sharpe']['min'])} to {f(O['sharpe']['max'])}, and the worst month from
  {f(O['worst']['min'], 0)} to {f(O['worst']['max'], 0)} vega points.
* Sizing: the worst window opened on any day lost {f(abs(Z['unfiltered']['worst']), 0)} vega points
  (opened {we['start']}, VIX {f(we['vix'], 1)}, realised volatility {f(we['rv'], 1)} after).
  Sized so that day costs 5% of a $100,000 account, you sell ${f(Z['unfiltered']['n_star'], 0)} per vol point,
  and the average year earns {pc(Z['unfiltered']['year_return_at_n_star'], 1)} of the account.
* Conditions (VIX level, gap to realised, curve, VIX change, rate phase, recessions): no group differs
  from the others by the resampling test. The thinnest ones (inverted curve, recessions) have
  {r['conditions']['slope']['groups']['inverted']['n']} and {r['conditions']['recession']['groups']['recession']['n']} months.
* One filter, chosen before the run: do not sell when the curve is inverted. Sharpe {f(t['unfiltered']['sharpe'])} to
  {f(t['filtered']['sharpe'])} from {t['first'][:7]} on, but the 95% range of the difference is {ci(t['sharpe_diff_ci'])}
  (it includes zero), and the worst start day is unchanged. {F['variants']['total']} variants tried in all,
  {F['variants']['groups_looked_at']} groups looked at.

## What is not modelled

A real trade uses VIX futures, SPX options or a dealer variance swap, with a bid-offer
spread, margin and financing. Here a flat cost is taken off the strike. The VIX is not exactly the strike
of a 30-day variance swap (jumps, discretisation, skew): the gap was not modelled and can go either way.
No margin calls or forced exits. Data starts in 1990, so the 1987 crash is not in it.
The recession split uses NBER dates, which are only known months later: it is descriptive, not a rule.

## The method

1. **Realised volatility** (`vrp/rv.py`): square root of the average squared daily log return, annualised
   with 252 days, no mean subtracted, 21-day windows. `fwd[t]` uses days t+1 to t+21.
2. **The trade** (`vrp/swap.py`): seller of a 1-month variance swap struck at the VIX read at the close of day t.
   Profit per 1 unit of vega notional is `(K^2 - RV^2) / (2 K)`, with K the strike net of cost.
3. **Cycles**: non-overlapping 21-day cycles from a start offset (0 to 20). The headline uses offset 0.
   All 21 offsets are reported.
4. **Uncertainty** (`vrp/stats.py`): circular moving-block bootstrap (blocks of {M['block']} months, {M['reps']} resamples,
   seed {M['seed']}) with a seeded generator (mulberry32) that the JavaScript on the site also uses, so the ranges match.
5. **Conditions** (`vrp/conditions.py`): features known at the close of the start day, cut-offs written before the results.
6. **Filters** (`vrp/filters.py`): a term-structure filter (no parameter) and a floor filter tested walk-forward.

## Run it

```
pip install -r requirements.txt
python -m pytest -q
python examples/make_results.py             # uses the cached data, writes data/*.json
python examples/make_results.py --refresh   # downloads everything again first
python examples/make_readme.py              # rewrites this file from data/results.json
```

`make_results.py` writes `data/results.json` (every number on the site), `data/history.json` (the derived daily table the
page loads), `data/vix.json` (the built-in copy of the live file) and `data/ref_check.json` (reference values that the
JavaScript port must reproduce to 1e-9).

## Data

| Series | Source | Use |
|---|---|---|
| VIXCLS, VXVCLS | FRED | VIX and 3-month VIX (from December 2007) |
| SP500 | FRED | last ten years of closes, for the live reading |
| ^GSPC | Yahoo through yfinance | closes since 1989, for realised volatility |
| FEDFUNDS, USREC | FRED | rate phase and NBER recession months |
| DTB3 | FRED | rate context for the reading of today |

The Yahoo file is not committed (it is a raw price table): `python -m vrp.data` downloads it again. FRED downloads
stall if a custom User-Agent is sent, so `vrp/data.py` sends none.

## Files

| File | What it does |
|---|---|
| `vrp/data.py` | downloads, cache, the one daily table |
| `vrp/rv.py`, `vrp/swap.py` | realised volatility, the variance swap and its cycles |
| `vrp/stats.py` | Sharpe, drawdown, skew, quantiles, seeded bootstrap |
| `vrp/conditions.py`, `vrp/filters.py` | groups known at the start, filters, walk-forward |
| `vrp/study.py`, `vrp/today.py`, `vrp/export.py` | the whole study, the live reading, files for the site |
| `tests/test_vrp.py` | units, no look-ahead, sign conventions, accounting identities, edge cases |
| `REVIEW.md` | ways the result could be wrong and what was done about each |
'''

with open(os.path.join(ROOT, 'README.md'), 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(TEXT)
print('wrote README.md')
