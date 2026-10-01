"""Run the whole study and write every file the site needs.

    python examples/make_results.py            # uses the cached data
    python examples/make_results.py --refresh  # download everything again first

Writes, in data/:
    results.json      every number quoted on the site (single source)
    history.json      the derived daily table the exhibit loads (VIX, VXV, realised vol)
    vix.json          the built-in copy of the live file (last ten years, FRED)
    ref_check.json    reference values the JavaScript must reproduce to 1e-9
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from vrp import data, export, rv, stats, study, swap  # noqa: E402


def cross_vendor(table):
    """Second route for the closes: FRED's S&P 500 (last ten years) instead of Yahoo's.

    Same VIX, same trade, closes from a different vendor. If the result moved, a
    data error would be to blame, not the idea.
    """
    fred = data.read_fred('SP500').dropna()
    vix = data.read_fred('VIXCLS').reindex(fred.index)
    rf = rv.fwd(fred.values)
    yahoo = table.loc[fred.index[0]:]
    common = fred.index.intersection(yahoo.index)
    a = pd_series(rf, fred.index).reindex(common)
    b = yahoo['rv_fwd'].reindex(common)
    ok = a.notna() & b.notna()
    out = {'days_compared': int(ok.sum()), 'max_rv_fwd_diff': float((a[ok] - b[ok]).abs().max()),
           'days_over_0_1': int(((a[ok] - b[ok]).abs() > 0.1).sum()),
           'close_days_that_differ': int((fred.reindex(common).round(2) != (yahoo['spx'].reindex(common)).round(2)).sum())}
    for name, f, v in (('fred', rf, vix.values), ('yahoo', yahoo['rv_fwd'].values, yahoo['vix'].values)):
        c = swap.cycles(v, f, 0, cost=study.COST)
        out[name] = {'n': int(len(c)), 'sharpe': stats.sharpe(c.pnl), 'mean': stats.mean(c.pnl), 'worst': float(c.pnl.min())}
    return out


def pd_series(values, index):
    import pandas as pd
    return pd.Series(values, index=index)


def main():
    if '--refresh' in sys.argv:
        print('downloading...')
        print(json.dumps(data.refresh_cache()))
    log_path = os.path.join(data.CACHE, 'fetch_log.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else None

    table = data.build_table()
    results, live = study.run(table, data.read_fred('FEDFUNDS'), data.read_fred('USREC'),
                              fetch_log=log, tbill=data.read_fred('DTB3'))
    results['checks'] = {'cross_vendor': cross_vendor(table)}
    generated = (log or {}).get('fetched_utc', '')
    hist = export.history_payload(table, generated)
    hist_arrays = export.as_arrays(json.loads(json.dumps(hist)))
    out = os.path.join(data.ROOT, 'data')

    def write(name, obj, compact=False):
        with open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(export.clean(obj), f, indent=None if compact else 1,
                      separators=(',', ':') if compact else None)
        print('wrote', name, os.path.getsize(os.path.join(out, name)), 'bytes')

    write('history.json', hist, compact=True)
    write('vix.json', live, compact=True)
    write('ref_check.json', export.ref_checks(json.loads(json.dumps(hist)), json.loads(json.dumps(live))))
    write('results.json', results)
    print('headline: Sharpe %.2f, worst month %.1f, last cycle %s'
          % (results['swap']['net']['sharpe'], results['swap']['net']['worst'], results['meta']['last_cycle_start']))


if __name__ == '__main__':
    main()
