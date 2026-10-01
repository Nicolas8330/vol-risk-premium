"""Does selling volatility get paid, and what does it cost when it goes wrong?

The package tests one idea on public data: the VIX (the market's price for the
next 30 days of S&P 500 movement) is usually higher than the movement that then
happens. Selling that gap looks like free money most months. The question is
what it costs in the other months.

Modules, in the order you would read them:

    data        FRED and Yahoo downloads, the cache, and the one daily table
    rv          realised volatility from closing prices
    swap        the trade: a short 1-month variance swap, as a proxy
    stats       Sharpe, drawdown, skew, and the block bootstrap
    conditions  what was known on the day, and the groups it defines
    filters     one simple filter, tested walk-forward
    today       the reading for the latest data
"""

TRADING_DAYS = 252      # trading days in a year, the usual convention
HORIZON = 21            # trading days in the 30 calendar days the VIX covers
PERIODS = 12            # 21-day cycles per year (252 / 21)
SEED = 20260930         # one seed for every bootstrap, shared with the JavaScript
BLOCK = 6               # bootstrap block length, in monthly cycles
REPS = 2000             # bootstrap resamples on the page and in the tests
