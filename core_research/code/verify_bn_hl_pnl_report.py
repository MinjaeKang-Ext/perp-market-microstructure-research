"""Independent checks for the two-model performance tables.

Recomputes the decomposition on sampled ledger days with pandas merge_asof
instead of the searchsorted path, re-derives the pair tables straight from the
per-market summary, and re-reads the prediction guards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from work.bn_hl_pnl_decompose import BASE, LEDGERS, OUT, date_bounds, native_book

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = [('hyperliquid_SKHX', 500, '2026-10-16'), ('hyperliquid_SKHX', 1000, '2026-10-27'),
          ('hyperliquid_SKHY', 500, '2026-10-30'), ('hyperliquid_SKHY', 1000, '2026-10-20')]


def independent_day(market, cadence, day):
    event, mid = native_book(day, market)
    book = pd.DataFrame(dict(t=event.astype('int64'), mid=mid)).sort_values('t')
    f = pd.read_parquet(LEDGERS / market / str(cadence) / f'{day}.fills.parquet')
    f = f.sort_values('time_us').reset_index(drop=True)
    now = pd.merge_asof(f[['time_us']].rename(columns={'time_us': 't'}), book, on='t',
                        allow_exact_matches=False, tolerance=1_000_000)
    later = pd.merge_asof(pd.DataFrame({'t': f.time_us + 60_000_000}), book, on='t',
                          allow_exact_matches=True, tolerance=1_000_000)
    _, end = date_bounds(day)
    mature = now.mid.notna() & later.mid.notna() & (f.time_us + 60_000_000 <= end)
    signed = f.side.to_numpy(float) * f.qty.to_numpy(float)
    has_now = now.mid.notna().to_numpy()
    return dict(spread_capture_usd=float(np.sum(signed[has_now] * (now.mid.to_numpy()[has_now] - f.price.to_numpy()[has_now]))),
                markout_60_usd=float(np.sum(signed[mature] * (later.mid.to_numpy()[mature] - now.mid.to_numpy()[mature]))),
                fee_pnl_usd=float(-f.fee.sum()),
                turnover_usd=float((f.qty * f.price).sum()),
                mature_60_fills=int(mature.sum()))


def main():
    daily = pd.read_csv(OUT / 'daily.csv')
    checks = {}
    worst = 0.
    for market, cadence, day in SAMPLE:
        got = independent_day(market, cadence, day)
        row = daily[(daily.market == market) & (daily.cadence_ms == cadence) & (daily.day == day)]
        assert len(row) == 1
        for k, v in got.items():
            diff = abs(float(row[k].iloc[0]) - v)
            scale = max(1., abs(v))
            worst = max(worst, diff / scale)
            assert diff / scale < 1e-9, (market, cadence, day, k, row[k].iloc[0], v)
    checks['independent_recompute_max_rel_diff'] = worst

    # Pair tables must equal the per-market summary totals.
    summary = pd.read_csv(OUT / 'summary.csv')
    report = pd.read_csv(OUT / 'report_decomposition.csv')
    gap = 0.
    for cadence in (500, 1000):
        for metric in ('spread_capture_usd', 'markout_60_usd', 'fee_pnl_usd', 'total_pnl_usd'):
            s = summary[(summary.cadence_ms == cadence) & (summary.metric == metric)
                        & (summary.partition == 'ALL_INDEPENDENT_EPISODES')].total_usd.sum()
            r = report[(report.cadence_ms == cadence) & (report.column == metric)].total_usd_ALL.iloc[0]
            gap = max(gap, abs(s - r))
    checks['pair_vs_per_market_summary_max_usd'] = gap
    assert gap < 1e-6

    # Row identity and the component sum both hold on every ledger day.
    d = pd.read_csv(OUT / 'report_pair_daily.csv')
    recon = (d.spread_capture_usd + d.markout_60_usd + d.fee_pnl_usd
             + d.mature_reconciliation_usd + d.immature_reconciliation_usd
             + d.no_fill_mid_contribution_usd + d.opening_inventory_mtm_usd
             + d.funding_usd - d.total_pnl_usd).abs().max()
    checks['pair_daily_identity_max_usd'] = float(recon)
    assert recon < 1e-6

    # Guards produced when the predictions were built.
    for name in ('prediction_independent_verification.json', 'prediction_label_guard_verification.json'):
        checks[name] = json.loads((BASE / name).read_text())

    manifest = json.loads((OUT / 'manifest.json').read_text())
    checks['decompose_days'] = manifest['days']
    checks['decompose_max_identity_error_usd'] = manifest['max_abs_identity_error_usd']
    assert manifest['days'] == 72
    checks['status'] = 'pass'
    (OUT / 'report_verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
