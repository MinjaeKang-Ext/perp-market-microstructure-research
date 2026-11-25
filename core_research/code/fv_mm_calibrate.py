"""Past-only distance calibration from isolated one-second maker submissions.

No inventory path is simulated here. Candidate distances are scored on a fixed
quantity, using the same public-volume/initial-queue proxy as the replay. These
are training observations, not independently tradable overlapping PnL streams.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

WIDTHS = np.array([0.5, 1., 2., 4., 8., 16.])
HORIZONS = (1, 5, 30, 60)
AGES = (1, 5)


@njit(cache=True)
def tick_at(price, market_code):
    if market_code == 0:
        return 0.01
    decimal_floor = 0.001 if market_code == 1 else 0.0001
    return min(1., max(decimal_floor, 10. ** (math.floor(math.log10(price)) - 4)))


@njit(cache=True)
def rounded(price, direction, market_code):
    if price <= 0 or not math.isfinite(price):
        return np.nan
    step = tick_at(price, market_code)
    if direction < 0:
        return math.floor(price / step + 1e-9) * step
    return math.ceil(price / step - 1e-9) * step


@njit(cache=True)
def projected(price, side, best_bid, best_ask, market_code):
    if side == 1:
        below = best_ask - tick_at(best_ask * (1. - 1e-10), market_code)
        return min(rounded(price, -1, market_code), rounded(below, -1, market_code))
    above = best_bid + tick_at(best_bid, market_code)
    return max(rounded(price, 1, market_code), rounded(above, 1, market_code))


@njit(cache=True)
def independent_surface(indices, grid, mid, bid, ask, alpha, future_mid,
                        ab, aa, abq, aaq, l2_idx, l2bp, l2bq, l2ap, l2aq,
                        tt, ts, tp, tq, market_code, arrival_us, maker_rate,
                        unit_qty):
    # side, width, [opportunities, fill_qty, net$, notional$, adverse$, fills]
    out = np.zeros((2, len(WIDTHS), 6))
    for i in indices:
        if not (math.isfinite(alpha[i]) and math.isfinite(future_mid[i])):
            continue
        fv = mid[i] * (1. + alpha[i] / 10000.)
        step = tick_at(mid[i], market_code)
        activation = grid[i] + arrival_us
        expiry = activation + 1000000
        lo = np.searchsorted(tt, activation, side='right')
        hi = np.searchsorted(tt, expiry, side='left')
        for si in range(2):
            side = 1 if si == 0 else -1
            for wi in range(len(WIDTHS)):
                p = projected(fv - side * WIDTHS[wi] * step,
                              side, bid[i], ask[i], market_code)
                out[si, wi, 0] += 1
                # Price acceptance is resolved at arrival, not used to resize
                # an already filled order or to remove a decision opportunity.
                if not math.isfinite(p) or p <= 0:
                    continue
                if (side == 1 and p >= aa[i] - 1e-10) or (side == -1 and p <= ab[i] + 1e-10):
                    continue
                queue = -1.  # unknown deeper queue: exact-price volume ignored
                touch = ab[i] if side == 1 else aa[i]
                if abs(p - touch) <= 1e-8:
                    queue = abq[i] if side == 1 else aaq[i]
                elif (side == 1 and p > touch) or (side == -1 and p < touch):
                    queue = 0.
                elif l2_idx[i] >= 0:
                    row = l2_idx[i]
                    for level in range(l2bp.shape[1]):
                        lp = l2bp[row, level] if side == 1 else l2ap[row, level]
                        if abs(lp - p) <= 1e-8:
                            queue = l2bq[row, level] if side == 1 else l2aq[row, level]
                            break
                remaining = unit_qty
                filled = 0.
                for k in range(lo, hi):
                    if ts[k] != -side:
                        continue
                    through = tp[k] < p - 1e-8 if side == 1 else tp[k] > p + 1e-8
                    exact = abs(tp[k] - p) <= 1e-8
                    if not through and not exact:
                        continue
                    if through:
                        queue = 0.
                    if queue < 0:
                        continue
                    used = min(queue, tq[k])
                    queue -= used
                    quantity = min(remaining, tq[k] - used)
                    filled += quantity
                    remaining -= quantity
                    if remaining < 1e-12:
                        break
                if filled > 0:
                    notional = filled * p
                    gross = filled * side * (future_mid[i] - p)
                    adverse = filled * side * (fv - future_mid[i])
                    out[si, wi, 1] += filled
                    out[si, wi, 2] += gross - notional * maker_rate
                    out[si, wi, 3] += notional
                    out[si, wi, 4] += adverse
                    out[si, wi, 5] += 1
    return out


def unit_quantity(g, market):
    mid = g['mid'][np.isfinite(g['mid']) & (g['mid'] > 0)][0]
    lot = .001 if market == 'hyperliquid_SKHX' else .01
    return math.floor(2000. / mid / lot + 1e-9) * lot


def daily_surface(g, l2, forecast, market, horizon, age, warmup=False):
    bn = market.startswith('binance')
    lag = 100 if bn else 440
    code = 0 if bn else (1 if market.endswith('SKHX') else 2)
    alpha = np.zeros(len(g['mid'])) if warmup else forecast[f'pred_exchange_cross_h{horizon}_a{age}'].astype(float)
    valid = forecast[f'decision_valid_cross_a{age}'].copy()
    if warmup:
        valid &= np.isfinite(g['mid'])
    future = np.full(len(alpha), np.nan)
    future[:-horizon] = g[f'truth_mid_{lag}'][horizon:]
    future_age = np.full(len(alpha), np.inf)
    future_age[:-horizon] = g[f'truth_age_{lag}'][horizon:]
    valid &= np.isfinite(alpha) & np.isfinite(future) & (future_age >= 0) & (future_age <= age)
    # End-of-day horizon purge and fixed ex-ante submission schedule.
    valid &= np.arange(len(alpha)) < len(alpha) - max(horizon + 2, 60)
    valid &= np.arange(len(alpha)) % 2 == 0
    arrival = g['grid_us'] + lag * 1000
    li = np.searchsorted(l2['l2_event_us'], arrival, side='right') - 1
    buckets = []
    for state in (g['imbalance'] < 0, g['imbalance'] >= 0):
        buckets.append(independent_surface(
            np.flatnonzero(valid & state), g['grid_us'], g['mid'], g['bid'], g['ask'], alpha, future,
            g[f'truth_bid_{lag}'], g[f'truth_ask_{lag}'],
            g[f'truth_bq_{lag}'], g[f'truth_aq_{lag}'], li,
            l2['l2_bid_px'], l2['l2_bid_qty'], l2['l2_ask_px'], l2['l2_ask_qty'],
            g['tt_us'], g['tside'], g['tprice'], g['tqty'], code, lag * 1000,
            (-.5 if bn else -.03) / 10000., unit_quantity(g, market)))
    return np.stack([buckets[0] + buckets[1], *buckets])


def choose_distances(history):
    result = []
    for side in range(2):
        table = history[0, side]
        score = np.divide(table[:, 2], table[:, 0], out=np.full(len(WIDTHS), -np.inf),
                          where=table[:, 0] > 0)
        if np.isfinite(score).any():
            index = int(np.argmax(score))
        else:
            index = 0
        row = table[index]
        base_tau = row[4] / row[3] * 10000. if row[3] > 0 else 0.
        incremental_tox = []
        for bucket in (1, 2):
            state_row = history[bucket, side, index]
            tau = state_row[4] / state_row[3] * 10000. if state_row[3] > 0 else base_tau
            incremental_tox.append(float(max(0., tau - base_tau)))
        result.append(dict(width_ticks=float(WIDTHS[index]), base_tau_bp=float(base_tau),
                           toxicity_negative_obi_bp=incremental_tox[0],
                           toxicity_nonnegative_obi_bp=incremental_tox[1],
                           train_opportunities=int(row[0]), train_fills=int(row[5]),
                           train_net_dollars_per_opportunity=float(score[index]) if np.isfinite(score[index]) else None))
    return result
