"""Causal two-sided, one-shot FV quotes against native trades.

Decisions are two seconds apart. Orders activate after the venue latency and
expire one second later. No BBO-cross synthetic fills are used. Queue seeds are
scenario assumptions, not observed personal queue positions. Funding is added
by the experiment driver. This is not a continuous replacement engine.
"""
from __future__ import annotations

import math
from time import perf_counter
from typing import Mapping

import numpy as np
from numba import njit

MARKET_CODES = {"binance_SKHYNIXUSDT": 0, "binance_SKHYUSDT": 0,
                "hyperliquid_SKHX": 1, "hyperliquid_SKHY": 2}
FILL_COLUMNS = ("time_us", "side", "qty", "price", "fee", "maker", "cash",
                "inventory", "realized_pnl", "closed_qty")


@njit(cache=True)
def tick_at_price(price, marketcode):
    if marketcode == 0:
        return 0.01
    decimals = 3 if marketcode == 1 else 4
    # Integer prices are always allowed, even above five significant figures.
    return min(1.0, max(10.0 ** (math.floor(math.log10(price)) - 4),
                        10.0 ** (-decimals)))


@njit(cache=True)
def price_floor(price, marketcode):
    if price <= 0 or not np.isfinite(price):
        return np.nan
    step = tick_at_price(price, marketcode)
    return math.floor(price / step + 1e-9) * step


@njit(cache=True)
def price_ceil(price, marketcode):
    if price <= 0 or not np.isfinite(price):
        return np.nan
    step = tick_at_price(price, marketcode)
    return math.ceil(price / step - 1e-9) * step


@njit(cache=True)
def previous_price(price, marketcode):
    # At 1000 the valid predecessor is 999.99, not 999.9.
    step = tick_at_price(price * (1 - 1e-12), marketcode)
    return price_floor(price - step, marketcode)


@njit(cache=True)
def next_price(price, marketcode):
    return price_ceil(price + tick_at_price(price, marketcode), marketcode)


@njit(cache=True)
def raw_target_prices(mid, alpha_bp, qnorm, hb_ticks, ha_ticks,
                      toxb_bp, toxa_bp, skew_ticks, marketcode):
    tick = tick_at_price(mid, marketcode)
    reservation = mid * (1 + alpha_bp / 10000) - skew_ticks * tick * qnorm
    return (reservation - hb_ticks * tick - toxb_bp * mid / 10000,
            reservation + ha_ticks * tick + toxa_bp * mid / 10000)


@njit(cache=True)
def project_prices(bid_raw, ask_raw, best_bid, best_ask, marketcode):
    return (min(price_floor(bid_raw, marketcode), previous_price(best_ask, marketcode)),
            max(price_ceil(ask_raw, marketcode), next_price(best_bid, marketcode)))


@njit(cache=True)
def _queue_seed(price, side, best_bid, best_ask, bq, aq, arrival,
                l2_event, l2_bpx, l2_bqty, l2_apx, l2_aqty, queue_mode):
    if queue_mode == 0:
        return 0.0, -1.0, 0
    tol = 1e-8
    if side == 1:
        if abs(price - best_bid) < tol:
            return max(0.0, bq), -1.0, 0
        if price > best_bid:
            return 0.0, -1.0, 0
    else:
        if abs(price - best_ask) < tol:
            return max(0.0, aq), -1.0, 0
        if price < best_ask:
            return 0.0, -1.0, 0
    ix = np.searchsorted(l2_event, arrival, side="right") - 1
    if ix >= 0:
        for level in range(20):
            px = l2_bpx[ix, level] if side == 1 else l2_apx[ix, level]
            if abs(price - px) < tol:
                qty = l2_bqty[ix, level] if side == 1 else l2_aqty[ix, level]
                return max(0.0, qty), (arrival - l2_event[ix]) / 1e6, 0
    # Unknown deep queue is not silently assigned zero. Exact-price trades
    # cannot fill; a strictly through execution can remove the unknown queue.
    return np.inf, -1.0, 1


@njit(cache=True)
def _book_valid(bid, ask, bq, aq):
    return (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(bq)
            and np.isfinite(aq) and 0 < bid < ask and bq >= 0 and aq >= 0)


@njit(cache=True)
def _apply_fill(cash, q, basis, side, qty, px, fee_bp):
    fee = qty * px * fee_bp / 10000
    cash -= side * qty * px + fee
    realized, closed = 0.0, 0.0
    per_qty_fee = fee / qty
    if q * side >= 0:
        new_q = q + side * qty
        basis = ((abs(q) * basis + qty * (px + side * per_qty_fee))
                 / abs(new_q))
    else:
        closed = min(abs(q), qty)
        realized = (1 if q > 0 else -1) * (px - basis) * closed - per_qty_fee * closed
        new_q = q + side * qty
        if q * new_q < 0:
            basis = px + side * per_qty_fee
        elif abs(new_q) < 1e-10:
            new_q, basis = 0.0, 0.0
    return cash, new_q, basis, fee, realized, closed


@njit(cache=True, inline="always")
def _record_fill(fills, count, stamp, side, qty, px, fee, maker,
                 cash, q, realized, closed):
    fills[count, 0] = stamp
    fills[count, 1] = side
    fills[count, 2] = qty
    fills[count, 3] = px
    fills[count, 4] = fee
    fills[count, 5] = maker
    fills[count, 6] = cash
    fills[count, 7] = q
    fills[count, 8] = realized
    fills[count, 9] = closed


@njit(cache=True, inline="always")
def _update_ioc_liquidity(book, bid, ask, bq, aq):
    # Each side: price, previous public display, unconsumed hypothetical depth.
    # A fresh event timestamp is not evidence that our consumed size returned.
    for offset in (0, 3):
        px = bid if offset == 0 else ask
        shown = bq if offset == 0 else aq
        if abs(book[offset] - px) > 1e-8:
            book[offset], book[offset + 1], book[offset + 2] = px, shown, shown
        else:
            replenished = max(0., shown - book[offset + 1])
            book[offset + 2] = min(shown, book[offset + 2] + replenished)
            book[offset + 1] = shown


@njit(cache=True)
def _replay(grid, bid, ask, mid, observed_bq, observed_aq, alpha, valid,
            truth_bid, truth_ask, truth_bq, truth_aq, truth_event,
            tt, tside, tprice, tqty, l2_event, l2_bpx, l2_bqty, l2_apx, l2_aqty,
            hb, ha, tox_b, tox_a, skew, queue_mode, marketcode, arrival_us,
            maker_bp, taker_bp, unit, lot, cash0, q0, basis0,
            allow_quotes, liquidation_seconds, ioc_book):
    n = len(grid)
    # A native trade can contribute at most one fill. IOC contributes at most
    # one fill per two-second decision. Never resize earlier entries ex post.
    fills = np.empty((len(tt) + n // 2 + 2, 10), dtype=np.float64)
    count = 0
    cash, q, basis = cash0, q0, basis0
    # stats: decisions, placed sides, arrival rejects, filled sides,
    # partial fills, unknown queues, L2 seeds, L2 age sum/max,
    # IOC attempts, turnover, fees, maker turnover, max_abs_inventory,
    # raw target movement sides, capped sides, valid decision count.
    stats = np.zeros(17, dtype=np.float64)
    stats[13] = abs(q)
    for i in range(0, n, 2):
        # Track the displayed price/quantity path at every sampled arrival,
        # including between liquidation attempts and across carried days.
        if _book_valid(truth_bid[i], truth_ask[i], truth_bq[i], truth_aq[i]):
            _update_ioc_liquidity(ioc_book, truth_bid[i], truth_ask[i],
                                  truth_bq[i], truth_aq[i])
        if not (_book_valid(bid[i], ask[i], observed_bq[i], observed_aq[i])
                and np.isfinite(mid[i]) and mid[i] > 0):
            continue
        # Close-window decisions use observed inventory and book only. Arrival
        # liquidity is checked after the decision, never as a hindsight gate.
        liquidate = i >= n - liquidation_seconds
        if liquidate:
            if abs(q) <= 1e-10:
                continue
            stats[9] += 1
            if not _book_valid(truth_bid[i], truth_ask[i], truth_bq[i], truth_aq[i]):
                continue
            side = -1 if q > 0 else 1
            px = truth_bid[i] if side == -1 else truth_ask[i]
            offset = 0 if side == -1 else 3
            avail = ioc_book[offset + 2]
            qty = min(abs(q), avail)
            qty = math.floor(qty / lot + 1e-8) * lot
            if qty <= 1e-10:
                continue
            cash, q, basis, fee, realized, closed = _apply_fill(
                cash, q, basis, side, qty, px, taker_bp)
            ioc_book[offset + 2] -= qty
            _record_fill(fills, count, grid[i] + arrival_us, side, qty, px, fee,
                         0, cash, q, realized, closed)
            count += 1
            stats[10] += qty * px
            stats[11] += fee
            stats[13] = max(stats[13], abs(q))
            continue
        if not allow_quotes or not valid[i] or not np.isfinite(alpha[i]):
            continue
        stats[0] += 1
        stats[16] += 1
        raw_b, raw_a = raw_target_prices(mid[i], alpha[i], q / unit, hb, ha,
                                         tox_b[i], tox_a[i], skew, marketcode)
        if not (np.isfinite(raw_b) and np.isfinite(raw_a) and raw_b > 0 and raw_a > raw_b):
            continue
        pb, pa = project_prices(raw_b, raw_a, bid[i], ask[i], marketcode)
        if not (np.isfinite(pb) and np.isfinite(pa) and 0 < pb < pa):
            continue
        stats[14] += (abs(pb - bid[i]) > 1e-8) + (abs(pa - ask[i]) > 1e-8)
        stats[15] += (pb < price_floor(raw_b, marketcode) - 1e-8) + (pa > price_ceil(raw_a, marketcode) + 1e-8)
        # Fixed outstanding quantities: both sides can fill, and any execution
        # order keeps inventory within the initial capacity interval.
        qb = math.floor(min(unit, max(0., unit - q)) / lot + 1e-8) * lot
        qa = math.floor(min(unit, max(0., unit + q)) / lot + 1e-8) * lot
        stats[1] += (qb > 1e-10) + (qa > 1e-10)
        if not _book_valid(truth_bid[i], truth_ask[i], truth_bq[i], truth_aq[i]):
            stats[2] += (qb > 1e-10) + (qa > 1e-10)
            continue
        if qb > 1e-10 and pb >= truth_ask[i] - 1e-8:
            qb = 0.
            stats[2] += 1
        if qa > 1e-10 and pa <= truth_bid[i] + 1e-8:
            qa = 0.
            stats[2] += 1
        arrival = grid[i] + arrival_us
        expiry = arrival + 1_000_000
        queue_b, age_b, unknown_b = _queue_seed(pb, 1, truth_bid[i], truth_ask[i],
            truth_bq[i], truth_aq[i], arrival, l2_event, l2_bpx, l2_bqty, l2_apx, l2_aqty, queue_mode)
        queue_a, age_a, unknown_a = _queue_seed(pa, -1, truth_bid[i], truth_ask[i],
            truth_bq[i], truth_aq[i], arrival, l2_event, l2_bpx, l2_bqty, l2_apx, l2_aqty, queue_mode)
        stats[5] += unknown_b * (qb > 1e-10) + unknown_a * (qa > 1e-10)
        if qb > 1e-10 and age_b >= 0:
            stats[6] += 1
            stats[7] += age_b
            stats[8] = max(stats[8], age_b)
        if qa > 1e-10 and age_a >= 0:
            stats[6] += 1
            stats[7] += age_a
            stats[8] = max(stats[8], age_a)
        first = np.searchsorted(tt, arrival, side="right")
        stop = np.searchsorted(tt, expiry, side="left")
        filled_b, filled_a = False, False
        for j in range(first, stop):
            side = 1 if tside[j] < 0 else -1
            px = pb if side == 1 else pa
            remaining = qb if side == 1 else qa
            if remaining <= 1e-10:
                continue
            eligible = tprice[j] <= px + 1e-8 if side == 1 else tprice[j] >= px - 1e-8
            if not eligible:
                continue
            through = tprice[j] < px - 1e-8 if side == 1 else tprice[j] > px + 1e-8
            queue = queue_b if side == 1 else queue_a
            if through:
                queue = 0.
            available = tqty[j]
            take_queue = min(queue, available)
            available -= take_queue
            queue -= take_queue
            if side == 1:
                queue_b = queue
            else:
                queue_a = queue
            qty = min(remaining, max(0., available))
            # Native trade quantities and accepted order quantities are on the
            # lot grid; floating point noise must not create sublot inventory.
            qty = math.floor(qty / lot + 1e-8) * lot
            if qty <= 1e-10:
                continue
            if qty < remaining - 1e-8:
                stats[4] += 1
            if side == 1:
                qb -= qty
                if not filled_b:
                    stats[3] += 1
                    filled_b = True
            else:
                qa -= qty
                if not filled_a:
                    stats[3] += 1
                    filled_a = True
            cash, q, basis, fee, realized, closed = _apply_fill(
                cash, q, basis, side, qty, px, maker_bp)
            _record_fill(fills, count, tt[j], side, qty, px, fee,
                         1, cash, q, realized, closed)
            count += 1
            stats[10] += qty * px
            stats[11] += fee
            stats[12] += qty * px
            stats[13] = max(stats[13], abs(q))
    return fills[:count].copy(), stats, cash, q, basis


def replay_day(g: Mapping, l2: Mapping, alpha_bp, valid, hb_ticks, ha_ticks,
               toxb_bp=0., toxa_bp=0., inventory_skew_ticks=0., queue_mode=1,
               market="binance_SKHYNIXUSDT", initial_state=None,
               allow_quotes=True, liquidation_seconds=60):
    """Return daily metrics, a timestamped fill ledger, equity and carry state.

    ``valid`` must use decision-time information only. Execution arrays can
    contain future liquidity, but are used strictly after each attempted order.
    The close window consumes native snapshot liquidity without resizing entry.
    """
    started = perf_counter()
    if market not in MARKET_CODES:
        raise ValueError(f"Unsupported market: {market}")
    if hb_ticks <= 0 or ha_ticks <= 0 or inventory_skew_ticks < 0:
        raise ValueError("Positive widths and nonnegative risk adjustments required")
    if not np.isfinite([hb_ticks, ha_ticks, inventory_skew_ticks]).all():
        raise ValueError("Nonfinite policy parameter")
    if queue_mode not in (0, 1) or liquidation_seconds < 0:
        raise ValueError("Invalid queue mode or liquidation interval")
    grid = np.asarray(g["grid_us"], np.int64)
    n = len(grid)
    if not n or np.any(np.diff(grid) != 1_000_000):
        raise ValueError("Expected contiguous 1-second grid")
    alpha, mask = np.asarray(alpha_bp, float), np.asarray(valid, bool)
    if len(alpha) != n or len(mask) != n:
        raise ValueError("Forecast/mask length mismatch")
    tox = [np.full(n, float(value)) if np.ndim(value) == 0 else np.asarray(value, float)
           for value in (toxb_bp, toxa_bp)]
    if any(v.shape != (n,) or not np.isfinite(v).all() or (v < 0).any() for v in tox):
        raise ValueError("Toxicity must be nonnegative finite scalar or day array")
    tt = np.asarray(g["tt_us"], np.int64)
    if np.any(np.diff(tt) < 0):
        raise ValueError("Native trade timestamps must be ordered")
    for name in ("tside", "tprice", "tqty"):
        if len(g[name]) != len(tt):
            raise ValueError("Native trade array length mismatch")
    if not np.isin(g["tside"], [-1, 1]).all():
        raise ValueError("Unknown aggressive trade side")
    if np.any(~np.isfinite(g["tprice"])) or np.any(~np.isfinite(g["tqty"])) or np.any(g["tqty"] <= 0):
        raise ValueError("Invalid native trades")
    code = MARKET_CODES[market]
    arrival_ms = 100 if code == 0 else 440
    lot = .01 if code in (0, 2) else .001
    first_mid = np.asarray(g["mid"], float)
    finite = np.isfinite(first_mid) & (first_mid > 0)
    if not finite.any():
        raise ValueError("No valid midpoint in day")
    first_px = float(first_mid[np.flatnonzero(finite)[0]])
    unit = math.floor((2000 / first_px) / lot + 1e-8) * lot
    if unit <= 0:
        raise ValueError("Unit quantity rounds to zero")
    state = initial_state or {}
    cash0, q0, basis0 = (float(state.get("cash", 0)), float(state.get("inventory", 0)),
                         float(state.get("avg_entry_price", 0)))
    if q0 and basis0 <= 0:
        raise ValueError("Carry inventory needs a positive entry cost basis")
    if not np.isfinite([cash0, q0, basis0]).all():
        raise ValueError("Nonfinite initial account state")
    prior_ioc = state.get("ioc_liquidity", {})
    ioc_book = np.array([float(prior_ioc.get(key, 0.)) for key in
                        ("bid_price", "bid_last_display", "bid_available",
                         "ask_price", "ask_last_display", "ask_available")])
    if not np.isfinite(ioc_book).all() or (ioc_book < 0).any():
        raise ValueError("Invalid carried IOC liquidity ledger")
    keys = ("bid", "ask", "mid", "bq", "aq")
    observed = [np.asarray(g[key], float) for key in keys]
    if any(len(values) != n for values in observed):
        raise ValueError("Observed book array length mismatch")
    truth = [np.asarray(g[f"truth_{key}_{arrival_ms}"], float)
             for key in ("bid", "ask", "bq", "aq")]
    events = np.asarray(g[f"truth_event_us_{arrival_ms}"], np.int64)
    if any(len(v) != n for v in truth) or len(events) != n:
        raise ValueError("Arrival book array length mismatch")
    event_ok = events >= 0
    if np.any(events[event_ok] > (grid + arrival_ms * 1000)[event_ok]):
        raise ValueError("Arrival query contains a future native book")
    le = np.asarray(l2.get("l2_event_us", np.empty(0)), np.int64)
    if np.any(np.diff(le) < 0):
        raise ValueError("L2 snapshots must be ordered by event time")
    empty_book = np.empty((0, 20), float)
    l2_arrays = [np.asarray(l2.get(key, empty_book), float)
                 for key in ("l2_bid_px", "l2_bid_qty", "l2_ask_px", "l2_ask_qty")]
    if any(v.shape != (len(le), 20) for v in l2_arrays):
        raise ValueError("L2 snapshots require 20 levels on each side")
    raw, stats, cash, q, basis = _replay(
        grid, *observed, alpha, mask, *truth, events,
        tt, np.asarray(g["tside"], np.int8), np.asarray(g["tprice"], float),
        np.asarray(g["tqty"], float), le, *l2_arrays,
        float(hb_ticks), float(ha_ticks), *tox,
        float(inventory_skew_ticks), queue_mode, code, arrival_ms * 1000,
        -.5 if code == 0 else -.03, .6273 if code == 0 else .288,
        unit, lot, cash0, q0, basis0, bool(allow_quotes), liquidation_seconds,
        ioc_book)
    fills = {name: raw[:, j].copy() for j, name in enumerate(FILL_COLUMNS)}
    fills["time_us"] = fills["time_us"].astype(np.int64)
    fills["side"] = fills["side"].astype(np.int8)
    fills["maker"] = fills["maker"].astype(bool)
    fills["liquidity"] = np.where(fills["maker"], "maker", "taker")
    # Actual fill timestamps, not interval endpoints, determine each mark.
    at = np.searchsorted(fills["time_us"], grid, side="right") - 1
    cq = np.full(n, q0)
    cc = np.full(n, cash0)
    known = at >= 0
    if len(raw):
        cq[known] = fills["inventory"][at[known]]
        cc[known] = fills["cash"][at[known]]
    valid_mark = np.isfinite(first_mid) & (first_mid > 0)
    last_valid = np.maximum.accumulate(np.where(valid_mark, np.arange(n), -1))
    marks = np.where(last_valid >= 0, first_mid[np.maximum(last_valid, 0)],
                     float(state.get("last_mid", first_px)))
    equity = cc + cq * marks
    start_equity = cash0 + q0 * float(state.get("last_mid", marks[0]))
    peak = np.maximum.accumulate(np.r_[start_equity, equity])
    mdd = float(np.max(peak - np.r_[start_equity, equity]))
    closed = fills["closed_qty"] > 0
    nclosed = int(closed.sum())
    metric_names = ("decision_opportunities", "submitted_sides", "postonly_rejects",
                    "filled_order_sides", "partial_fill_events", "unknown_queue_sides",
                    "l2_seed_sides", "l2_seed_age_sum_seconds", "l2_seed_age_max_seconds",
                    "ioc_attempts", "turnover_usd", "fees_usd", "maker_turnover_usd",
                    "max_abs_inventory", "moved_quote_sides", "postonly_capped_sides",
                    "valid_decisions")
    metrics = dict(zip(metric_names, stats.tolist()))
    metrics.update(pnl_usd=float(equity[-1] - start_equity), end_equity_usd=float(equity[-1]),
                   start_equity_usd=float(start_equity), mdd_usd=mdd, fills=len(raw),
                   maker_fills=int(fills["maker"].sum()), taker_fills=int((~fills["maker"]).sum()),
                   matched_close_events=nclosed,
                   matched_close_win_rate=float(np.mean(fills["realized_pnl"][closed] > 0)) if nclosed else None,
                   realized_pnl_usd=float(fills["realized_pnl"].sum()),
                   unit_qty=float(unit), end_inventory=float(q),
                   end_inventory_notional_usd=float(q * marks[-1]),
                   runtime_seconds=perf_counter() - started)
    endstate = dict(cash=float(cash), inventory=float(q), avg_entry_price=float(basis),
                    last_mid=float(marks[-1]))
    endstate["ioc_liquidity"] = dict(zip(
        ("bid_price", "bid_last_display", "bid_available",
         "ask_price", "ask_last_display", "ask_available"), ioc_book.tolist()))
    return dict(metrics=metrics, equity=equity, inventory_1s=cq,
                fills=fills, endstate=endstate)
