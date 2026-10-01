# Review: how this result could be wrong

Written before the page. For each risk: what I did about it, and what is left.
The numbers come from `data/results.json` (run of 30 September 2026).

1. **Look-ahead in the features or decisions.**
   Every feature uses only the close of the start day. Tests rewrite the last 300 days of prices, VIX and 3-month VIX
   and check that every cycle that had finished, and every feature before the cut, is unchanged
   (`test_no_look_ahead_in_the_whole_pipeline`, `test_features_only_use_the_past`). The walk-forward floor and the
   walk-forward "adopt the filter" rule are tested the same way (rewrite months from month j on: nothing chosen before j moves).
   Fed funds is read two months back (and compared with 14 months back) because a month is not known until it ends.
   **Left:** the NBER recession split is hindsight by construction (NBER dates recessions months late). It is labelled as
   descriptive on the page and is not offered as a rule.

2. **Overlapping windows make the result look more certain than it is.**
   The headline uses non-overlapping 21-day cycles. Ranges come from a circular block bootstrap (blocks of 6 months,
   2000 resamples), and blocks of 3 and 12 months give nearly the same range (0.43 to 2.08, 0.44 to 2.10, 0.46 to 2.13).
   A separate route for the daily gap (every day, 126-day blocks) gives 3.41 to 4.70 against 3.26 to 4.71 for the
   monthly one. **Left:** nothing: both routes agree.

3. **The result depends on a few episodes.**
   It does, and this is the finding, not a flaw. Without the worst 3 months the Sharpe is 1.76, without the worst 5% it is 3.94.
   Without the best year (2009) it is 0.92, without the worst year (2008) it is 1.25. The term-structure filter's gain rests on
   about four months (four skipped losses of more than 20 vega points). Stated on the page.

4. **The Sharpe hides the risk.**
   Skewness is -6.1. The page reports the worst month (-90.5), the worst window over all start days (-262), the
   fall from the peak (123 vega points), the time to recover (21 months) and shows the sizing consequence
   (the average year earns about 0.5% of an account sized to the worst window). The Sharpe is never shown alone.

5. **The start day is luck.**
   The headline uses start day 1. All 21 start days are computed: Sharpe 0.47 to 1.04, worst month -262 to -67. The exhibit has a slider
   for the start day. The worst month on the headline path (-90.5) is not the worst month that could have been drawn (-262).

6. **The VIX is not the strike of a variance swap.**
   It is close (that is what the VIX formula measures) but differs by discretisation, the strikes available, and jumps. Not modelled;
   the sign of the error is not certain. **Left, on the page.**

7. **Costs are invented.**
   A flat 0.5 vol point off the strike is an assumption. Sharpe at 0, 0.5, 1 and 2 points: 1.19, 0.98, 0.77, 0.37 (start day 1)
   and 1.06, 0.88, 0.70, 0.32 (median over start days). Real costs are larger in stress; no margin, financing or roll is modelled.

8. **The data could be wrong.**
   Two vendors for the closes: Yahoo (used) and FRED (last ten years). 9 closes differ at the second decimal in ten years (the largest gap is on 11 August 2021),
   no day's forward realised volatility moves by more than 0.07 vol point, and the Sharpe over that decade is 0.516 with both.
   The table refuses a download with VIX outside 5 to 120, negative volatility, duplicate or unsorted dates, or more than 1% missing VIX
   (`check_table`). Four days have no VIX (1991-03-01, 1997-01-31, 1997-11-26 and the last day): their cycles are dropped and
   the gap shows in the cycle numbering. VIX values on US holidays that FRED carries are ignored (the calendar is the S&P 500's).
   Known events cross-check the table: VIX close 82.69 on 16 March 2020 (`test_real_table_is_sane`).

9. **The 1990s VIX is a back-calculation and may not be comparable.**
   As I understand it, the exchange recomputed the VIX before 2003 with the later method. I did not check that. The 1990s
   Sharpe (3.16) is far above every other period (2000s 0.60, 2010s 1.16, since 2020 0.43). The page says not to lean on it.
   Since 2000 the Sharpe is 0.68, since 2010 it is 0.72. **Left:** unverified.

10. **Multiple testing.**
    21 groups in 7 families and 6 filter variants were looked at, with cut-offs and the filter written down before the run.
    The resampling test on each group against the other months finds no group that differs. Nothing is presented as a rule.
    The page says a lucky group is likely among those looked at.

11. **The filter was chosen with knowledge of history.**
    I know curves inverted in 2008 and 2020. The walk-forward "adopt if ahead so far" rule (from December 2012) adopted the filter
    in 100% of months and gave a Sharpe of 1.08 against 0.59, with a 95% range for the difference of -0.34 to 1.09. That is
    not strong evidence. The floor filter never won a walk-forward choice: every floor lowered the Sharpe. The minimum training
    lengths (120 and 60 months) were fixed in advance and not varied. **Left:** stated on the page.

12. **The horizon.**
    The VIX covers 30 calendar days and I use 21 trading days. With 19, 20, 22 and 23 days the Sharpe is 0.86, 1.03, 0.52 and 0.55:
    the same start-day effect: which cycle the 2020 losses fall into changes with the window.

13. **The code.**
    41 pytest tests. A JavaScript port of the same maths, checked against Python on 366 values to 1e-9 (bootstrap ranges included,
    with the same seeded generator). An independent pandas-only route from the raw files gives a Sharpe of 0.97888 against 0.97888
    (the difference is 2e-6, from rounding realised volatility to 4 decimals in the shipped table) and the same daily gap (4.0997).
    The JavaScript also recomputes realised volatility from FRED closes and compares it with the shipped columns built from Yahoo: the
    largest difference is 0.07 vol point.

14. **Things I did not try to break.**
    Different underlyings, dividends, event days (CPI, FOMC, ECB), and intraday data. The 1987 crash is not in the sample. Recession months
    and rate phases are few long stretches (4 recessions, 9 to 18 rate stretches), so those splits cannot be tested well.
