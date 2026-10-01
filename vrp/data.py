"""Downloads, the cache, and the one daily table the study runs on.

Sources (all free, no key):
    FRED   VIXCLS   VIX close, 1990 to today
    FRED   VXVCLS   3-month VIX close (called VIX3M today), December 2007 on
    FRED   SP500    S&P 500 close, only the last ten years (a licence limit)
    FRED   DTB3, DGS3MO   3-month Treasury bill, for rate context
    FRED   USREC    NBER recession months (0 or 1, monthly)
    FRED   FEDFUNDS effective fed funds rate, monthly average
    Yahoo  ^GSPC    S&P 500 close from 1989 (through the yfinance package)

The cache lives in data/cache/. `python -m vrp.data` refreshes it. The Yahoo
file is kept out of git (it is a raw price table from an exchange index); the
site only ever receives numbers derived from it (realised volatility).

Everything downstream uses `build_table()`, whose realised-volatility columns
are rounded to 4 decimals. The JavaScript loads the same rounded numbers, so
the browser and Python start from identical inputs.
"""

import io
import json
import os
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import HORIZON, rv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'data', 'cache')
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'
FRED_SERIES = ['VIXCLS', 'VXVCLS', 'SP500', 'DTB3', 'DGS3MO', 'USREC', 'FEDFUNDS']
STUDY_START = '1990-01-02'      # the first day the VIX exists on FRED
ROUND = 4                        # decimals kept for realised volatility


def _fred_path(series):
    return os.path.join(CACHE, f'fred_{series}.csv')


def fetch_fred(series, timeout=60):
    """Download one FRED series as text (the CSV the graph page serves)."""
    # No custom User-Agent: FRED's graph endpoint stalls requests that send one.
    with urllib.request.urlopen(FRED_URL.format(series), timeout=timeout) as response:
        return response.read().decode('utf-8')


def fetch_yahoo(start='1989-12-01'):
    """S&P 500 closes from Yahoo through yfinance, as a Series indexed by date."""
    import yfinance as yf
    end = (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    frame = yf.download('^GSPC', start=start, end=end, auto_adjust=False, progress=False)
    close = frame['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = 'Close'
    return close


def refresh_cache():
    """Download every source again and write the cache plus a small log."""
    os.makedirs(CACHE, exist_ok=True)
    log = {'fetched_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    for series in FRED_SERIES:
        text = fetch_fred(series)
        with open(_fred_path(series), 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        s = read_fred(series)
        log[series] = {'first': str(s.dropna().index[0].date()), 'last': str(s.dropna().index[-1].date()),
                       'rows': int(s.notna().sum())}
    yahoo = fetch_yahoo()
    yahoo.to_csv(os.path.join(CACHE, 'gspc_yahoo.csv'), header=True)
    log['GSPC'] = {'first': str(yahoo.index[0].date()), 'last': str(yahoo.index[-1].date()), 'rows': int(len(yahoo))}
    with open(os.path.join(CACHE, 'fetch_log.json'), 'w') as f:
        json.dump(log, f, indent=1)
    return log


def read_fred(series):
    """One cached FRED series, gaps as NaN, indexed by date."""
    frame = pd.read_csv(_fred_path(series), index_col=0, parse_dates=True, na_values=['.', ''])
    s = frame.iloc[:, 0].astype(float)
    s.name = series
    return s


def read_spx():
    """The cached Yahoo S&P 500 closes."""
    frame = pd.read_csv(os.path.join(CACHE, 'gspc_yahoo.csv'), index_col=0, parse_dates=True)
    return frame['Close'].astype(float).dropna()


def build_table(spx=None, vix=None, vxv=None, start=STUDY_START):
    """The daily table: one row per S&P 500 trading day from `start`.

    Columns: vix, vxv (NaN when not published), rv_back (realised vol of the last
    21 returns, known at the close), rv_fwd (realised vol of the next 21 returns,
    NaN for the last 21 days). The calendar is the S&P 500's, so realised
    volatility never skips a day; the VIX is read on those days and is NaN on the
    rare day it was not published.
    """
    spx = read_spx() if spx is None else spx
    vix = read_fred('VIXCLS') if vix is None else vix
    vxv = read_fred('VXVCLS') if vxv is None else vxv
    spx = spx[~spx.index.duplicated(keep='last')].sort_index()
    frame = pd.DataFrame({'spx': spx})
    frame['vix'] = vix.reindex(frame.index)
    frame['vxv'] = vxv.reindex(frame.index)
    frame['rv_back'] = rv.back(frame['spx'].values)
    frame['rv_fwd'] = rv.fwd(frame['spx'].values)
    frame = frame.loc[start:]
    frame['rv_back'] = frame['rv_back'].round(ROUND)
    frame['rv_fwd'] = frame['rv_fwd'].round(ROUND)
    check_table(frame)
    return frame


def check_table(frame):
    """Refuse a table that looks broken, rather than compute on it."""
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError('dates must be strictly increasing')
    vix = frame['vix'].dropna()
    if len(vix) and (vix.min() < 5 or vix.max() > 120):
        raise ValueError('VIX outside 5 to 120: the download looks wrong')
    for col in ('rv_back', 'rv_fwd'):
        v = frame[col].dropna()
        if len(v) and (v.min() < 0 or v.max() > 300):
            raise ValueError(f'{col} outside 0 to 300')
    if frame['vix'].isna().mean() > 0.01:
        raise ValueError('more than 1% of days have no VIX')


def monthly_lookup(series, dates, lag_months):
    """For each date, the value of the monthly series `lag_months` months earlier.

    Monthly series are dated on the first of the month. A value for month m is
    only known after the month ends (and NBER dates far later), so callers ask
    for a lag and never read the month they are in.
    """
    key = {(d.year * 12 + d.month - 1): v for d, v in series.dropna().items()}
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        out[i] = key.get(d.year * 12 + d.month - 1 - lag_months, np.nan)
    return out


if __name__ == '__main__':
    print(json.dumps(refresh_cache(), indent=1))
