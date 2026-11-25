"""Pair-combined performance tables for the two BN->HL next-cadence models.

Model (1) cadence 1000ms, model (2) cadence 500ms. Quotes are submitted on the
model's own grid. Latency inputs are the recorded ACK medians already written
into every ledger (submit 440ms, cancel 750ms); the signal axis is the
collector receive clock, so feed delay is not added again.

Outputs go to out_bn_hl_pnl/analysis/report_*.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from work.bn_hl_pnl_decompose import LEDGERS, OUT, asof_book, date_bounds, native_book

ROOT = Path(__file__).resolve().parents[1]
IS_DAYS = [f'2026-10-{d}' for d in range(15, 24)]
OOS_DAYS = [f'2026-10-{d}' for d in range(24, 32)] + ['2026-11-01']
HORIZON_US = 60_000_000

COMPONENTS = {
    'spread_capture_usd': '스프레드 수취',
    'markout_60_usd': '역선택 (체결 후 60초 가격 이동)',
    'fee_pnl_usd': '수수료',
    'fill_edge_usd': '체결 우위 합계',
    'inventory_pnl_usd': '재고 손익 (60초 이후)',
    'funding_usd': '펀딩',
    'total_pnl_usd': '하루 손익',
}


def load_daily():
    d = pd.read_csv(OUT / 'daily.csv')
    d['inventory_pnl_usd'] = (d.mature_reconciliation_usd + d.immature_reconciliation_usd
                              + d.no_fill_mid_contribution_usd + d.opening_inventory_mtm_usd)
    d['fill_edge_usd'] = d.partial_60_fill_edge_usd
    check = (d.fill_edge_usd + d.inventory_pnl_usd + d.funding_usd - d.total_pnl_usd).abs().max()
    assert check < 1e-6, f'row identity {check}'
    assert set(d.day) == set(IS_DAYS + OOS_DAYS), 'unexpected calendar'
    return d


def ledger_stats():
    rows = []
    for path in sorted(LEDGERS.glob('*/*/*.daily.json')):
        j = json.loads(path.read_text())
        rows.append({k: j[k] for k in ('day', 'market', 'cadence_ms', 'submitted',
                                       'activated', 'rejected', 'cancel_requested',
                                       'fill_events', 'valid_forecast_rows',
                                       'max_abs_inventory', 'allow_quotes',
                                       'order_submit_effect_us', 'cancel_effect_us',
                                       'fees_bp')})
    s = pd.DataFrame(rows)
    assert s.order_submit_effect_us.nunique() == 1 and s.cancel_effect_us.nunique() == 1
    return s


def fill_level(cadence):
    """Per-fill 60s edge in USD and bp, pooled over both markets."""
    out = []
    for market in ('hyperliquid_SKHX', 'hyperliquid_SKHY'):
        folder = LEDGERS / market / str(cadence)
        for path in sorted(folder.glob('*.fills.parquet')):
            day = path.name.split('.')[0]
            event, mid = native_book(day, market)
            f = pd.read_parquet(path)
            if not len(f):
                continue
            stamp = f.time_us.to_numpy(np.int64)
            m0, _ = asof_book(event, mid, stamp, strict=True)
            mh, _ = asof_book(event, mid, stamp + HORIZON_US)
            _, end = date_bounds(day)
            mature = np.isfinite(m0) & np.isfinite(mh) & (stamp + HORIZON_US <= end)
            signed = f.side.to_numpy(float) * f.qty.to_numpy(float)
            edge = signed * (mh - f.price.to_numpy(float)) - f.fee.to_numpy(float)
            notional = f.qty.to_numpy(float) * f.price.to_numpy(float)
            out.append(pd.DataFrame(dict(day=day, market=market, mature=mature,
                                         edge_usd=edge, notional=notional)))
    d = pd.concat(out, ignore_index=True)
    d['partition'] = np.where(d.day.isin(IS_DAYS), 'IS', 'OOS')
    return d


def risk(series):
    """Daily-close PnL series -> annualized Sharpe, MDD and Calmar."""
    v = series.to_numpy(float)
    equity = np.r_[0., np.cumsum(v)]
    mdd = float(np.max(np.maximum.accumulate(equity) - equity))
    sd = float(v.std(ddof=1))
    sharpe = float(v.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    annual = float(v.mean() * 252)
    return dict(total_usd=float(v.sum()), daily_mean_usd=float(v.mean()),
                day_win_rate=float((v > 0).mean()), sharpe=sharpe, mdd_usd=mdd,
                calmar=annual / mdd if mdd > 0 else np.nan)


def build():
    daily = load_daily()
    stats = ledger_stats()
    pair = daily.groupby(['cadence_ms', 'day'], as_index=False).sum(numeric_only=True)
    pair['partition'] = np.where(pair.day.isin(IS_DAYS), 'IS', 'OOS')

    decomp_rows, perf_rows, fill_rows = [], [], []
    for cadence in sorted(pair.cadence_ms.unique()):
        g = pair[pair.cadence_ms == cadence]
        blocks = {'IS': g[g.partition == 'IS'], 'OOS': g[g.partition == 'OOS'], 'ALL': g}
        turnover_all = float(g.turnover_usd.sum())
        for key, label in COMPONENTS.items():
            row = dict(cadence_ms=cadence, item=label, column=key)
            for name, b in blocks.items():
                row[f'daily_mean_usd_{name}'] = float(b[key].sum()) / len(b)
                row[f'total_usd_{name}'] = float(b[key].sum())
            row['bp_turnover_ALL'] = float(g[key].sum()) / turnover_all * 1e4
            decomp_rows.append(row)

        f = fill_level(cadence)
        for name in ('IS', 'OOS', 'ALL'):
            b = f if name == 'ALL' else f[f.partition == name]
            m = b[b.mature]
            fill_rows.append(dict(cadence_ms=cadence, partition=name, fills=len(b),
                                  mature_fills=int(len(m)),
                                  mature_fraction=len(m) / len(b),
                                  fill_win_rate=float((m.edge_usd > 0).mean()),
                                  edge_bp=float(m.edge_usd.sum() / m.notional.sum() * 1e4),
                                  edge_usd_per_fill=float(m.edge_usd.mean())))
            sb = stats[stats.cadence_ms == cadence]
            sb = sb if name == 'ALL' else sb[sb.day.isin(IS_DAYS if name == 'IS' else OOS_DAYS)]
            bb = blocks[name]
            perf_rows.append(dict(cadence_ms=cadence, partition=name, days=len(bb),
                                  turnover_usd=float(bb.turnover_usd.sum()),
                                  submitted=int(sb.submitted.sum()),
                                  activated=int(sb.activated.sum()),
                                  rejected=int(sb.rejected.sum()),
                                  quote_opportunities=int(sb.valid_forecast_rows.sum()),
                                  fill_events=int(sb.fill_events.sum()),
                                  max_abs_inventory=float(sb.max_abs_inventory.max()),
                                  end_inventory=float(bb.end_inventory.iloc[-1]),
                                  **risk(bb.set_index('day').sort_index().total_pnl_usd)))

    d = pd.DataFrame(decomp_rows); p = pd.DataFrame(perf_rows); fl = pd.DataFrame(fill_rows)
    d.to_csv(OUT / 'report_decomposition.csv', index=False)
    p.to_csv(OUT / 'report_performance.csv', index=False)
    fl.to_csv(OUT / 'report_fill_edge.csv', index=False)
    pair.to_csv(OUT / 'report_pair_daily.csv', index=False)
    meta = dict(models={'1000': 'BN 1s -> HL next 1s best', '500': 'BN 500ms -> HL next 500ms best'},
                quote_cadence='equal to model cadence',
                latency_us=dict(submit_effective=int(stats.order_submit_effect_us.iloc[0]),
                                cancel_effective=int(stats.cancel_effect_us.iloc[0])),
                maker_fee_bp=float(stats.fees_bp.iloc[0]),
                markets=['hyperliquid_SKHX', 'hyperliquid_SKHY'],
                pair_combination='same-day sum of both contracts',
                risk='daily closing PnL; Sharpe annualized at 252; MDD on daily equity',
                episodes='IS 2026-10-15..23 and OOS 2026-10-24..11-01 reset separately')
    (OUT / 'report_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    for frame in (d, p, fl):
        print(frame.to_string(), flush=True)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    build()
