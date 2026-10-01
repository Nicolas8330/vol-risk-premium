# The volatility risk premium: getting paid to insure the market

The VIX is the market's price for the next month of S&P 500 movement. Most of the
time it is higher than the movement that follows. This repository tests what
happens if you sell that gap every month, and what it costs when it goes wrong.

Live version: [quant.orbiseo.fr/vol-risk-premium](https://quant.orbiseo.fr/vol-risk-premium/).
Research only. Nothing here was traded and nothing here is advice.

## The short answer

Selling the gap paid in most months and lost a lot in a few. Numbers are per 1 unit
of vega notional ("vega points": multiply by your dollars per vol point). One "month"
is a 21-trading-day cycle. Data: 1990-01-02 to 2026-09-29 (439 cycles).

| | No cost | Cost 0.5 vol point | Stress, cost 2 |
|---|---:|---:|---:|
| Sharpe ratio | 1.19 | 0.98 | 0.37 |
| 95% range for the Sharpe | 0.60 to 2.46 | 0.44 to 2.10 | -0.01 to 1.11 |
| Winning months | 84.3% | 82.7% | 75.2% |
| Average month | 2.72 | 2.29 | 0.93 |
| Worst month on this path | -88.9 | -90.5 | -95.7 |
| Skewness | -6.1 | -6.1 | -5.9 |
| Deepest fall from a peak | 119.7 | 122.9 | 153.5 |

* Implied against realised: the VIX was above the volatility of the next 21 days on
  85.9% of days, by 4.10 points on average
  (95% range for non-overlapping months 3.26 to 4.71).
* The start day matters. Over the 21 possible start days the Sharpe (cost 0.5) runs from
  0.47 to 1.04, and the worst month from
  -262 to -67 vega points.
* Sizing: the worst window opened on any day lost 262 vega points
  (opened 2020-02-14, VIX 13.7, realised volatility 84.2 after).
  Sized so that day costs 5% of a $100,000 account, you sell $19 per vol point,
  and the average year earns 0.5% of the account.
* Conditions (VIX level, gap to realised, curve, VIX change, rate phase, recessions): no group differs
  from the others by the resampling test. The thinnest ones (inverted curve, recessions) have
  27 and 35 months.
* One filter, chosen before the run: do not sell when the curve is inverted. Sharpe 0.54 to
  0.99 from 2007-12 on, but the 95% range of the difference is -0.15 to 1.11
  (it includes zero), and the worst start day is unchanged. 6 variants tried in all,
  21 groups looked at.

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
4. **Uncertainty** (`vrp/stats.py`): circular moving-block bootstrap (blocks of 6 months, 2000 resamples,
   seed 20260930) with a seeded generator (mulberry32) that the JavaScript on the site also uses, so the ranges match.
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
