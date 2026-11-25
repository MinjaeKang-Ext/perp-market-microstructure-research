"""First chronological eligible FV/inventory quote examples from saved replay.

The examples explain target-price mechanics. They are public-history model
counterfactuals, not observed personal executions or selected profitable trades.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from work.fv_mm_replay import (MARKET_CODES, raw_target_prices, project_prices,
                                tick_at_price, price_floor, price_ceil)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_fv_mm"
MARKETS = tuple(MARKET_CODES)
KEY = "h1_a1_q1_fv_inventory_toxicity"
KST = ZoneInfo("Asia/Seoul")


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def previous_state(market, day):
    earlier = sorted((OUT / "replay" / market).glob(f"*/{KEY}.json"))
    earlier = [p for p in earlier if p.parent.name < day]
    return json.loads(earlier[-1].read_text())["endstate"] if earlier else {}


def decision_state(stamp, fills, initial, settlements):
    """Only fills strictly before decision; funding is first at equal time."""
    before = np.searchsorted(fills["time_us"], stamp, side="left") - 1
    if before >= 0:
        q = float(fills["reconstructed_inventory"][before])
        cash = float(fills["reconstructed_cash"][before])
    else:
        q, cash = float(initial.get("inventory", 0.)), float(initial.get("cash", 0.))
    cash += sum(float(e["cashflow"]) for e in settlements if int(e["time_us"]) <= stamp)
    return q, cash


def quote_row(g, forecasts, fills, record, initial, index):
    market = record["market"]
    code = MARKET_CODES[market]
    stamp = int(g["grid_us"][index])
    q, cash = decision_state(stamp, fills, initial, record["settlements"])
    mid = float(g["mid"][index])
    alpha = float(forecasts["pred_exchange_cross_h1_a1"][index])
    fv = mid * (1. + alpha / 10000.)
    tick = float(tick_at_price(mid, code))
    unit = float(record["metrics"]["unit_qty"])
    qnorm = q / unit
    shift = tick * qnorm
    r = fv - shift
    negative = g["imbalance"][index] < 0
    tkey = "toxicity_negative_obi_bp" if negative else "toxicity_nonnegative_obi_bp"
    bp, ap = record["bid_params"], record["ask_params"]
    hb, ha = float(bp["width_ticks"]), float(ap["width_ticks"])
    tb, ta = float(bp[tkey]), float(ap[tkey])
    rb, ra = raw_target_prices(mid, alpha, qnorm, hb, ha, tb, ta, 1., code)
    rb0, ra0 = raw_target_prices(mid, alpha, 0., hb, ha, tb, ta, 1., code)
    b, a = project_prices(rb, ra, float(g["bid"][index]), float(g["ask"][index]), code)
    # Independent arithmetic identities verify the sign and magnitude of q's
    # effect, holding the contemporaneous FV and toxicity unchanged.
    raw_error = max(abs(rb - (r - hb * tick - tb * mid / 10000.)),
                    abs(ra - (r + ha * tick + ta * mid / 10000.)),
                    abs((rb - rb0) + shift), abs((ra - ra0) + shift))
    if raw_error > 1e-8:
        raise AssertionError("Inventory/raw quote arithmetic does not match policy")
    if not (b < float(g["ask"][index]) and a > float(g["bid"][index]) and b < a):
        raise AssertionError("Decision target is not post-only")
    grid_error = max(abs(price_floor(b, code) - b), abs(price_ceil(b, code) - b),
                     abs(price_floor(a, code) - a), abs(price_ceil(a, code) - a))
    if grid_error > 1e-8:
        raise AssertionError("Candidate-specific price grid violated")
    lot = .001 if code == 1 else .01
    qty_b = math.floor(min(unit, max(0., unit - q)) / lot + 1e-8) * lot
    qty_a = math.floor(min(unit, max(0., unit + q)) / lot + 1e-8) * lot
    lag = 100 if code == 0 else 440
    accept_b = qty_b > 1e-10 and b < float(g[f"truth_ask_{lag}"][index]) - 1e-8
    accept_a = qty_a > 1e-10 and a > float(g[f"truth_bid_{lag}"][index]) + 1e-8
    return dict(market=market, day=record["day"],
                decision_time_kst=datetime.fromtimestamp(stamp / 1e6, KST).isoformat(),
                decision_time_us=stamp, decision_index=index,
                best_bid=float(g["bid"][index]), best_ask=float(g["ask"][index]), mid=mid,
                alpha_bp=alpha, fair_value=fv, inventory=q, inventory_units=qnorm,
                cash_before_decision_usd=cash, unit_qty=unit, tick_at_mid=tick,
                inventory_shift_price=shift, reservation_price=r,
                half_width_bid_ticks=hb, half_width_ask_ticks=ha,
                toxicity_bid_bp=tb, toxicity_ask_bp=ta,
                bid_raw=float(rb), ask_raw=float(ra),
                bid_raw_if_flat=float(rb0), ask_raw_if_flat=float(ra0),
                target_bid=float(b), target_ask=float(a),
                distance_bid=float(fv - b), distance_ask=float(a - fv),
                raw_distance_bid=float(fv - rb), raw_distance_ask=float(ra - fv),
                bid_order_qty=float(qty_b), ask_order_qty=float(qty_a),
                bid_submitted=bool(qty_b > 1e-10), ask_submitted=bool(qty_a > 1e-10),
                bid_accepted_at_arrival=bool(accept_b), ask_accepted_at_arrival=bool(accept_a),
                ledger_inventory_reconstruction_error=float(fills["inventory_reconstruction_error"]),
                ledger_cash_reconstruction_error=float(fills["cash_reconstruction_error"]),
                raw_inventory_identity_error=float(raw_error), price_grid_error=float(grid_error))


def examples_for_market(market):
    wanted = {"first_maker_buy_pair", "first_fv_rise_and_long_pair"}
    found = {}
    files = sorted((OUT / "replay" / market).glob(f"*/{KEY}.json"))
    for path in files:
        record = json.loads(path.read_text())
        if not record["complete_trade_day"]:
            continue
        day = record["day"]
        fills_path = path.with_name(KEY + ".fills.npz")
        if not fills_path.exists():
            continue
        g = load_npz(OUT / "inputs" / market / (day + ".npz"))
        forecasts = load_npz(OUT / "forecasts" / market / (day + ".npz"))
        fills = load_npz(fills_path)
        initial = previous_state(market, day)
        # Independently rebuild the account from signed executions and fees,
        # then compare with the engine's stored cumulative ledger columns.
        fills["reconstructed_inventory"] = float(initial.get("inventory", 0.)) + np.cumsum(
            fills["side"] * fills["qty"])
        fills["reconstructed_cash"] = float(initial.get("cash", 0.)) - np.cumsum(
            fills["side"] * fills["qty"] * fills["price"] + fills["fee"])
        qe = float(np.max(np.abs(fills["reconstructed_inventory"] - fills["inventory"]), initial=0))
        ce = float(np.max(np.abs(fills["reconstructed_cash"] - fills["cash"]), initial=0))
        if qe > 1e-8 or ce > 1e-6:
            raise AssertionError("Independent execution account reconstruction failed")
        fills["inventory_reconstruction_error"] = np.array(qe)
        fills["cash_reconstruction_error"] = np.array(ce)
        mask = np.ones(len(g["grid_us"]), bool)
        for model in ("own", "cross", "crosslag1"):
            mask &= forecasts[f"decision_valid_{model}_a1"]
            mask &= np.isfinite(forecasts[f"pred_exchange_{model}_h1_a1"])
        mask[-60:] = False
        seen = set()
        for j in np.flatnonzero(fills["maker"] & (fills["side"] == 1)):
            elapsed = int(fills["time_us"][j] - g["grid_us"][0])
            index = elapsed // 2_000_000 * 2
            if index in seen or index < 0 or index + 2 >= len(mask):
                continue
            seen.add(index)
            if not mask[index] or not mask[index + 2]:
                continue
            before = quote_row(g, forecasts, fills, record, initial, index)
            after = quote_row(g, forecasts, fills, record, initial, index + 2)
            if abs(float(fills["price"][j]) - before["target_bid"]) > 1e-8:
                raise AssertionError("Saved first buy price differs from reconstructed decision quote")
            cats = []
            if "first_maker_buy_pair" not in found:
                cats.append("first_maker_buy_pair")
            if ("first_fv_rise_and_long_pair" not in found
                    and after["fair_value"] > before["fair_value"] + 1e-10
                    and after["inventory"] > max(before["inventory"], 0.) + 1e-10):
                cats.append("first_fv_rise_and_long_pair")
            for category in cats:
                pair = []
                for phase, row in (("source_decision", before), ("next_decision", after)):
                    pair.append(dict(row, case=category, phase=phase,
                                     selected_buy_time_us=int(fills["time_us"][j]),
                                     selected_buy_time_kst=datetime.fromtimestamp(int(fills["time_us"][j]) / 1e6, KST).isoformat(),
                                     selected_buy_qty=float(fills["qty"][j]),
                                     selected_buy_price=float(fills["price"][j]),
                                     paired_fv_change=after["fair_value"] - before["fair_value"],
                                     paired_inventory_change=after["inventory"] - before["inventory"]))
                found[category] = pair
            if set(found) == wanted:
                return found
    if "first_maker_buy_pair" not in found:
        raise RuntimeError(f"No eligible completed primary quote example yet: {market}")
    return found


def main():
    cases = {market: examples_for_market(market) for market in MARKETS}
    rows = [row for per_market in cases.values() for pair in per_market.values() for row in pair]
    pd.DataFrame(rows).to_csv(OUT / "quote_examples.csv", index=False)
    payload = dict(
        selection="For each market, first chronological maker-buy whose source and next 2-second decisions are both eligible; separately first such pair with FV rising and inventory increasing into a long position. No PnL filter.",
        configuration=dict(horizon=1, max_age=1, queue=1, policy="fv_inventory_toxicity"),
        state_rule="Fills strictly before decision; funding settlements at or before decision; previous-day account state retained.",
        caveats=["Public-history counterfactual fills, not observed personal executions.",
                 "Both target prices are shown even when inventory cap makes one order quantity zero.",
                 "Flat-inventory raw targets hold the same contemporaneous FV, widths and toxicity; they isolate inventory price shifts.",
                 "FV rise plus inventory increase does not guarantee either projected quote rises, because tick, toxicity, and inventory shift can dominate."],
        examples=cases, rows=len(rows),
        verification=dict(max_raw_inventory_identity_error=max(r["raw_inventory_identity_error"] for r in rows),
                          max_price_grid_error=max(r["price_grid_error"] for r in rows),
                          max_ledger_cash_reconstruction_error=max(r["ledger_cash_reconstruction_error"] for r in rows),
                          max_ledger_inventory_reconstruction_error=max(r["ledger_inventory_reconstruction_error"] for r in rows),
                          selected_buy_quote_prices_match=True))
    (OUT / "quote_examples.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=len(rows), markets=list(cases), verification=payload["verification"])), flush=True)
    for market, found in cases.items():
        for category, pair in found.items():
            a, b = pair
            print(market, category, a["decision_time_kst"],
                  "FV", a["fair_value"], "->", b["fair_value"],
                  "q", a["inventory"], "->", b["inventory"],
                  "quotes", (a["target_bid"], a["target_ask"]), "->", (b["target_bid"], b["target_ask"]), flush=True)


if __name__ == "__main__":
    main()
