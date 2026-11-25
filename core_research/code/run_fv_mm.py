"""Run all four venues/contracts with past-only FV-distance calibration.

Uses complete-date public trade coverage; missing-trade dates have no new
maker orders. Existing inventory and end-of-day IOC accounting are retained.
All strategies in a comparison use the intersection of causal feature masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mm_stage1 import funding6
from work.fv_mm_calibrate import AGES, HORIZONS, WIDTHS, choose_distances, daily_surface
from work.fv_mm_replay import replay_day

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out_fv_mm'
MARKETS = ('binance_SKHYNIXUSDT', 'binance_SKHYUSDT',
           'hyperliquid_SKHX', 'hyperliquid_SKHY')
DAYS = pd.date_range('2026-10-15', '2026-11-01').strftime('%Y-%m-%d').tolist()
POLICIES = ('mid', 'fv', 'fv_inventory', 'fv_inventory_toxicity',
            'own_inventory_toxicity', 'lag_inventory_toxicity')
MISSING_DAYS = {'binance_SKHYNIXUSDT': {'2026-10-27'},
                'binance_SKHYUSDT': {'2026-10-27'},
                'hyperliquid_SKHX': {'2026-11-01'},
                'hyperliquid_SKHY': {'2026-10-25'}}


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def json_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def funding_for_day(g, market, fill, state, oracle_cache):
    name = market
    events = json.loads((ROOT / 'work/funding_momentum' / (name + '.json')).read_text())
    bn = market.startswith('binance')
    start, end = int(g['grid_us'][0]), int(g['grid_us'][-1])
    times = fill['time_us']
    cashflows, records = np.zeros(len(g['grid_us'])), []
    for event in events:
        stamp = int(event['fundingTime' if bn else 'time']) * 1000
        if not start <= stamp <= end:
            continue
        k = np.searchsorted(times, stamp, side='left') - 1
        q = float(fill['inventory'][k]) if k >= 0 else float(state.get('inventory', 0))
        if abs(q) < 1e-10:
            q = 0.
        rate = float(event['fundingRate'])
        price, status = None, 'flat'
        if q:
            if bn:
                price, status = float(event['markPrice']), 'funding_history_mark'
            else:
                contract = 'xyz_' + market.split('_')[-1]
                price, _, _ = funding6._oracle_before(
                    ROOT / 'perp_data/trading_reference_data/processed',
                    contract, stamp * 1000, oracle_cache)
                status = 'past_observed_oracle_proxy'
        cash = -q * price * rate if q else 0.
        i = int(np.searchsorted(g['grid_us'], stamp, side='left'))
        cashflows[i] += cash
        records.append(dict(time_us=stamp, position=q, rate=rate,
                            settlement_price=price, source=status, cashflow=cash))
    return np.cumsum(cashflows), records


def validate_ledger(g, result, state):
    f = result['fills']
    cash0, q0 = float(state.get('cash', 0)), float(state.get('inventory', 0))
    cash = cash0 - np.cumsum(f['side'] * f['qty'] * f['price'] + f['fee'])
    q = q0 + np.cumsum(f['side'] * f['qty'])
    cash_error = float(np.max(np.abs(cash - f['cash']), initial=0))
    q_error = float(np.max(np.abs(q - f['inventory']), initial=0))
    if cash_error > 1e-6 or q_error > 1e-8:
        raise AssertionError(f'Cash/inventory ledger mismatch: {cash_error}/{q_error}')
    if len(cash):
        if abs(cash[-1] - result['endstate']['cash']) > 1e-6:
            raise AssertionError('Final cash differs from fills')
    elif abs(cash0 - result['endstate']['cash']) > 1e-6:
        raise AssertionError('Cash changes without executions')
    # Both fills and quotes must be based on this day's clock interval.
    if len(f['time_us']) and (np.diff(f['time_us']) < 0).any():
        raise AssertionError('Fill ledger not chronological')
    return dict(cash_max_error=cash_error, inventory_max_error=q_error)


def config_key(horizon, age, queue, policy):
    return f'h{horizon}_a{age}_q{queue}_{policy}'


def maker_diagnostics(g, fills, market, horizon):
    maker = fills['maker']
    if not np.any(maker):
        return dict(maker_markout_net_usd=0., maker_markout_notional_usd=0.,
                    maker_halfspread_rebate_weighted_sum=0.)
    stamp = fills['time_us'][maker]
    index = ((stamp - g['grid_us'][0]) // 2000000 * 2).astype(int)
    target_index = index + horizon
    lag = 100 if market.startswith('binance') else 440
    in_range = target_index < len(g['grid_us'])
    mark = np.full(len(index), np.nan)
    mark[in_range] = g[f'truth_mid_{lag}'][target_index[in_range]]
    valid = np.isfinite(mark)
    price, qty = fills['price'][maker], fills['qty'][maker]
    notional = price * qty
    net = fills['side'][maker] * qty * (mark - price) - fills['fee'][maker]
    rebate = .5 if market.startswith('binance') else .03
    threshold = .5 * g['spread_bp'][index] + rebate
    return dict(maker_markout_net_usd=float(np.sum(net[valid])),
                maker_markout_notional_usd=float(np.sum(notional[valid])),
                maker_halfspread_rebate_weighted_sum=float(np.sum(threshold[valid] * notional[valid])))


def run_market(market, wait_seconds=3600, profiles=None):
    selected_profiles = profiles or [(h, a, q) for h in HORIZONS for a in AGES for q in (0, 1)]
    history = {(h, a): np.zeros((3, 2, len(WIDTHS), 6)) for h in HORIZONS for a in AGES}
    states = {}
    diagnostics = []
    for day_index, day in enumerate(DAYS):
        paths = (OUT / 'inputs' / market / (day + '.npz'),
                 OUT / 'inputs' / market / (day + '.l2.npz'),
                 OUT / 'forecasts' / market / (day + '.npz'))
        deadline = time.monotonic() + wait_seconds
        while not all(p.exists() for p in paths):
            if time.monotonic() > deadline:
                raise FileNotFoundError(str(paths))
            print('WAIT_REPLAY', market, day, flush=True)
            time.sleep(5)
        g, l2, forecasts = [load_npz(path) for path in paths]
        complete = day not in MISSING_DAYS[market]
        if day_index:
            choices = {(h, a): choose_distances(history[h, a]) for h in HORIZONS for a in AGES}
            selections = []
            for (h, a), choice in choices.items():
                for side, params in zip(('bid', 'ask'), choice):
                    selections.append(dict(market=market, day=day, horizon=h, max_age=a,
                                           train_through=DAYS[day_index-1], side=side, **params))
            p = OUT / 'replay' / market / day / 'selection.json'
            json_save(p, selections)
            oracle_cache = {}
            for horizon, age, queue in selected_profiles:
                choice = choices[horizon, age]
                pred = {model: forecasts[f'pred_exchange_{model}_h{horizon}_a{age}'].astype(float)
                        for model in ('own', 'cross', 'crosslag1')}
                valid = np.ones(len(g['grid_us']), bool)
                for model in pred:
                    valid &= forecasts[f'decision_valid_{model}_a{age}'] & np.isfinite(pred[model])
                # Horizon/clock bounds are known at the decision; do not use
                # future quote freshness to decide whether to place an order.
                valid[-max(60, horizon + 2):] = False
                negative = g['imbalance'] < 0
                tox = [np.where(negative, c['toxicity_negative_obi_bp'],
                                c['toxicity_nonnegative_obi_bp']) for c in choice]
                for policy in POLICIES:
                    key = config_key(horizon, age, queue, policy)
                    state = states.setdefault(key, {})
                    output = OUT / 'replay' / market / day / (key + '.json')
                    if output.exists():
                        saved = json.loads(output.read_text())
                        states[key] = saved['endstate']
                        diagnostics.append(saved['ledger_validation'])
                        continue
                    alpha = np.zeros(len(valid)) if policy == 'mid' else pred[
                        'own' if policy.startswith('own') else ('crosslag1' if policy.startswith('lag') else 'cross')]
                    inventory = 0. if policy in ('mid', 'fv') else 1.
                    with_tox = policy.endswith('toxicity')
                    result = replay_day(g, l2, alpha, valid,
                                        choice[0]['width_ticks'], choice[1]['width_ticks'],
                                        tox[0] if with_tox else 0., tox[1] if with_tox else 0.,
                                        inventory_skew_ticks=inventory, queue_mode=queue,
                                        market=market, initial_state=state, allow_quotes=complete)
                    check = validate_ledger(g, result, state)
                    cum_funding, settlements = funding_for_day(g, market, result['fills'], state, oracle_cache)
                    equity = result['equity'] + cum_funding
                    result['endstate']['cash'] += float(cum_funding[-1])
                    prior_eq = float(state.get('cash', 0)) + float(state.get('inventory', 0)) * float(state.get('last_mid', g['mid'][np.flatnonzero(np.isfinite(g['mid']))[0]]))
                    prior_peak = float(state.get('equity_peak', max(0., prior_eq)))
                    peak = np.maximum.accumulate(np.r_[prior_peak, equity])
                    current_mdd = float(np.max(peak[1:] - equity))
                    overall_mdd = max(float(state.get('overall_mdd', 0)), current_mdd)
                    result['endstate'].update(equity_peak=float(peak[-1]), overall_mdd=overall_mdd)
                    metrics = result['metrics']
                    metrics.update(maker_diagnostics(g, result['fills'], market, horizon))
                    metrics.update(pnl_before_funding_usd=float(equity[-1] - cum_funding[-1] - prior_eq),
                                   funding_usd=float(cum_funding[-1]), pnl_usd=float(equity[-1] - prior_eq),
                                   end_equity_usd=float(equity[-1]), start_equity_usd=prior_eq,
                                   mdd_usd=current_mdd, overall_mdd_usd=overall_mdd)
                    # Portfolio PnL is independently cash+inventory, rather than
                    # a sum of selected completed round trips.
                    expected_end = result['endstate']['cash'] + result['endstate']['inventory'] * result['endstate']['last_mid']
                    if abs(expected_end - equity[-1]) > 1e-6:
                        raise AssertionError('Funding-adjusted equity does not match account state')
                    record = dict(market=market, day=day, horizon=horizon, max_age=age, queue=queue,
                                  policy=policy, complete_trade_day=complete, metrics=metrics,
                                  endstate=result['endstate'], ledger_validation=check,
                                  settlements=settlements, bid_params=choice[0], ask_params=choice[1])
                    if horizon == 1 and age == 1 and queue == 1:
                        keep = OUT / 'replay' / market / day / (key + '.fills.npz')
                        np.savez_compressed(keep, **result['fills'])
                        np.savez_compressed(keep.with_name(key + '.equity.npz'),
                                            grid_us=g['grid_us'], equity=equity,
                                            inventory=result['inventory_1s'])
                    json_save(output, record)
                    states[key] = result['endstate']
                    diagnostics.append(check)
            print('REPLAY', market, day, 'configs', len(selected_profiles) * len(POLICIES),
                  'complete_trade_day', complete, flush=True)
        # Update after every current-day decision has been evaluated. First-day
        # warmup uses mid only; later updates use genuine prior-day FV forecasts.
        surface_path = OUT / 'calibration' / market / (day + '.npz')
        if complete:
            if surface_path.exists():
                surfaces = load_npz(surface_path)
            else:
                surfaces = {f'h{h}_a{a}': daily_surface(g, l2, forecasts, market, h, a, day_index == 0)
                            for h in HORIZONS for a in AGES}
                surface_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(surface_path, **surfaces)
            for h in HORIZONS:
                for a in AGES:
                    history[h, a] += surfaces[f'h{h}_a{a}']
        print('CALIBRATED', market, day, 'warmup', day_index == 0, flush=True)
    json_save(OUT / 'replay' / market / 'validation.json',
              dict(days=len(DAYS), oos_days=len(DAYS)-1, ledger_runs=len(diagnostics),
                   cash_max_error=max((d['cash_max_error'] for d in diagnostics), default=0),
                   inventory_max_error=max((d['inventory_max_error'] for d in diagnostics), default=0)))


def summarize():
    records, selections = [], []
    for market in MARKETS:
        for day in DAYS[1:]:
            directory = OUT / 'replay' / market / day
            if not directory.exists():
                continue
            path = directory / 'selection.json'
            if path.exists():
                selections.extend(json.loads(path.read_text()))
            for path in sorted(directory.glob('h*.json')):
                row = json.loads(path.read_text())
                record = {key: row[key] for key in ('market', 'day', 'horizon', 'max_age', 'queue', 'policy', 'complete_trade_day')}
                record.update(row['metrics'])
                records.append(record)
    frame = pd.DataFrame(records).sort_values(['market', 'horizon', 'max_age', 'queue', 'policy', 'day'])
    frame.to_csv(OUT / 'replay_daily.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT / 'distance_selection.csv', index=False)
    rows = []
    keys = ['market', 'horizon', 'max_age', 'queue', 'policy']
    for key, block in frame.groupby(keys, sort=True):
        block = block.sort_values('day')
        daily = block.pnl_usd.to_numpy()
        sd = daily.std(ddof=1) if len(daily) > 1 else 0.
        sharpe = float(daily.mean() / sd * math.sqrt(252)) if sd > 0 else None
        mdd = float(block.overall_mdd_usd.max())
        pnl = float(daily.sum())
        calmar = float(pnl / len(daily) * 252 / mdd) if mdd > 0 else None
        closes = block.matched_close_events.sum()
        win = float((block.matched_close_win_rate.fillna(0) * block.matched_close_events).sum() / closes) if closes else None
        out = dict(zip(keys, key))
        out.update(days=len(daily), maker_enabled_days=int(block.complete_trade_day.sum()),
                   total_pnl_usd=pnl, daily_pnl_usd=float(daily.mean()),
                   daily_win_rate=float(np.mean(daily > 0)), matched_close_win_rate=win,
                   sharpe=sharpe, mdd_usd=mdd, calmar=calmar,
                   total_turnover_usd=float(block.turnover_usd.sum()),
                   fees_usd=float(block.fees_usd.sum()), funding_usd=float(block.funding_usd.sum()),
                   fills=int(block.fills.sum()), maker_fills=int(block.maker_fills.sum()),
                   taker_fills=int(block.taker_fills.sum()),
                   matched_close_events=int(closes),
                   decision_opportunities=int(block.decision_opportunities.sum()),
                   submitted_sides=int(block.submitted_sides.sum()),
                   filled_order_sides=int(block.filled_order_sides.sum()),
                   postonly_rejects=int(block.postonly_rejects.sum()),
                   partial_fill_events=int(block.partial_fill_events.sum()),
                   unknown_queue_sides=int(block.unknown_queue_sides.sum()),
                   l2_seed_sides=int(block.l2_seed_sides.sum()),
                   end_inventory=float(block.end_inventory.iloc[-1]),
                   end_inventory_notional_usd=float(block.end_inventory_notional_usd.iloc[-1]))
        out['fee_net_edge_bp_per_one_way_notional'] = pnl / out['total_turnover_usd'] * 10000 if out['total_turnover_usd'] else None
        maker_notional = float(block.maker_markout_notional_usd.sum())
        out['maker_markout_net_bp'] = float(block.maker_markout_net_usd.sum() / maker_notional * 10000) if maker_notional else None
        out['maker_halfspread_rebate_benchmark_bp'] = float(block.maker_halfspread_rebate_weighted_sum.sum() / maker_notional) if maker_notional else None
        edge_alarm = bool(maker_notional and out['maker_markout_net_bp'] > out['maker_halfspread_rebate_benchmark_bp'])
        out['edge_alarm'] = edge_alarm
        out['performance_alarm'] = bool((sharpe is not None and sharpe > 3) or (calmar is not None and calmar > 5) or edge_alarm)
        rows.append(out)
    pd.DataFrame(rows).to_csv(OUT / 'replay_summary.csv', index=False)
    return frame, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=MARKETS)
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--central-only', action='store_true')
    parser.add_argument('--no-summary', action='store_true')
    args = parser.parse_args()
    if not args.summary_only:
        for market in ([args.market] if args.market else MARKETS):
            run_market(market, profiles=[(1, 1, 1)] if args.central_only else None)
    if args.no_summary:
        return
    frame, summary = summarize()
    manifest = dict(
        markets=list(MARKETS), dates=DAYS, oos_dates=DAYS[1:], forecast_horizons=list(HORIZONS),
        ages=list(AGES), widths_ticks=WIDTHS.tolist(), quantity_notional_usd=2000,
        inventory_limit_units=1, inventory_skew_ticks=1,
        decision_grid_seconds=1, submission_period_seconds=2, active_seconds=1,
        arrivals_ms={'binance': 100, 'hyperliquid': 440},
        cancellation_ms={'binance': 170, 'hyperliquid': 750},
        cancellation_request='arrival+1second-cancel_latency; timer scheduled after assumed acknowledgement',
        queue_modes={'0': 'zero queue optimistic sensitivity', '1': 'BBO/L2 initial queue with through-price reset'},
        missing_trade_dates={m: sorted(d) for m, d in MISSING_DAYS.items()},
        missing_day_policy='no new maker orders on known incomplete-coverage dates; inventory retained and IOC unwind allowed',
        funding='historical rates; Binance settlement mark; HL latest observed past oracle proxy',
        fv_target='exchange midpoint at t+assumed_arrival+horizon',
        toxicity='positive conditional residual increase by OBI sign, above average already priced into width',
        comparison_mask='same intersection of own/cross/crosslag1 causal feature and finite prediction masks',
        limitations=['clock synchronization unverified', 'ACK used as activation proxy, not measured one-way arrival',
                     'Binance order/cancel latency assumed', 'L2 is sampled; personal queue unobserved',
                     'small-impact public-history counterfactual; no market response to our orders',
                     'scheduled one-second quote experiments, not continuous replacement',
                     'used research period, not untouched holdout', 'current lot/tick metadata not historical specification archive'],
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), ROOT / 'work/fv_mm_calibrate.py', ROOT / 'work/fv_mm_replay.py']},
        daily_rows=len(frame), summary_rows=len(summary))
    json_save(OUT / 'manifest.json', manifest)
    print('DONE', 'daily_rows', len(frame), 'summary_rows', len(summary), flush=True)


if __name__ == '__main__':
    main()
