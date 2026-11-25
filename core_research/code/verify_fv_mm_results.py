"""Independent read-only checks of completed FV/MM experiment artifacts.

Does not invoke the replay or refit forecasts. Uses saved native inputs, actual
fill ledgers, prior-day calibration surfaces and account identities. Personal
queue position, counterfactual impact and clock synchronization remain unknown.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_fv_mm"
MARKETS = ("binance_SKHYNIXUSDT", "binance_SKHYUSDT", "hyperliquid_SKHX", "hyperliquid_SKHY")
DAYS = pd.date_range("2026-10-15", "2026-11-01").strftime("%Y-%m-%d").tolist()
POLICIES = ("mid", "fv", "fv_inventory", "fv_inventory_toxicity",
            "own_inventory_toxicity", "lag_inventory_toxicity")
WIDTHS = np.array([.5, 1., 2., 4., 8., 16.])
HORIZONS, AGES = (1, 5, 30, 60), (1, 5)
MISSING = {MARKETS[0]: {"2026-10-27"}, MARKETS[1]: {"2026-10-27"},
           MARKETS[2]: {"2026-11-01"}, MARKETS[3]: {"2026-10-25"}}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def close(a, b, label, atol=2e-6):
    if not np.allclose(a, b, rtol=1e-10, atol=atol, equal_nan=True):
        raise AssertionError(label)


def json_load(path):
    return json.loads(path.read_text())


def key(h, age, queue, policy):
    return f"h{h}_a{age}_q{queue}_{policy}"


def past_selection(history, side):
    table = history[0, side]
    score = np.full(6, -np.inf)
    has_opportunities = table[:, 0] > 0
    score[has_opportunities] = table[has_opportunities, 2] / table[has_opportunities, 0]
    index = int(np.argmax(score)) if np.isfinite(score).any() else 0
    chosen = table[index]
    base = 10000 * chosen[4] / chosen[3] if chosen[3] > 0 else 0.
    conditional = []
    for bucket in (1, 2):
        row = history[bucket, side, index]
        tau = 10000 * row[4] / row[3] if row[3] > 0 else base
        conditional.append(max(0., tau - base))
    return dict(width_ticks=float(WIDTHS[index]), base_tau_bp=float(base),
                toxicity_negative_obi_bp=float(conditional[0]),
                toxicity_nonnegative_obi_bp=float(conditional[1]),
                train_opportunities=int(chosen[0]), train_fills=int(chosen[5]))


def check_parameters():
    count = 0
    for market in MARKETS:
        histories = {(h, age): np.zeros((3, 2, 6, 6)) for h in HORIZONS for age in AGES}
        for day in DAYS:
            directory = OUT / "replay" / market / day
            path = directory / "selection.json"
            if path.exists():
                for row in json_load(path):
                    if not row["train_through"] < day:
                        raise AssertionError("Parameter fit uses current/future day")
                    expected = past_selection(histories[row["horizon"], row["max_age"]],
                                              0 if row["side"] == "bid" else 1)
                    for name, value in expected.items():
                        close(row[name], value, f"Past-only distance selection {market}/{day}/{name}")
                    count += 1
            # Current surface is intentionally added only AFTER checking the
            # day's chosen parameters, independently of the driver function.
            surface = OUT / "calibration" / market / (day + ".npz")
            if day not in MISSING[market] and surface.exists():
                with np.load(surface) as z:
                    for h, age in histories:
                        histories[h, age] += z[f"h{h}_a{age}"]
    return count


def tick(price, market):
    if market.startswith("binance"):
        return .01
    floor = .001 if market.endswith("SKHX") else .0001
    return min(1., max(floor, 10. ** (math.floor(math.log10(price)) - 4)))


def floor_price(price, market):
    step = tick(price, market)
    return math.floor(price / step + 1e-9) * step


def ceil_price(price, market):
    step = tick(price, market)
    return math.ceil(price / step - 1e-9) * step


def mark_prices(g, state):
    mid = g["mid"]
    ok = np.isfinite(mid) & (mid > 0)
    last = np.maximum.accumulate(np.where(ok, np.arange(len(mid)), -1))
    fallback = float(state.get("last_mid", mid[np.flatnonzero(ok)[0]]))
    return np.where(last >= 0, mid[np.maximum(last, 0)], fallback)


def expected_mask(g, forecast, h, age):
    mask = np.ones(len(g["grid_us"]), bool)
    for model in ("own", "cross", "crosslag1"):
        mask &= forecast[f"decision_valid_{model}_a{age}"]
        mask &= np.isfinite(forecast[f"pred_exchange_{model}_h{h}_a{age}"])
    mask[-max(60, h + 2):] = False
    mask &= np.arange(len(mask)) % 2 == 0
    mask &= (np.isfinite(g["bid"]) & np.isfinite(g["ask"]) & np.isfinite(g["mid"])
             & np.isfinite(g["bq"]) & np.isfinite(g["aq"])
             & (g["bid"] > 0) & (g["ask"] > g["bid"])
             & (g["bq"] >= 0) & (g["aq"] >= 0))
    return mask


def check_ioc_liquidity(g, fill, state, record, lot, arrival):
    """Reconstruct sampled same-price depth without timestamp replenishment."""
    grid = g["grid_us"]
    lag = arrival // 1000
    bid, ask, bq, aq = (g[f"truth_{name}_{lag}"] for name in ("bid", "ask", "bq", "aq"))
    native_ok = (np.isfinite(bid) & np.isfinite(ask) & np.isfinite(bq) & np.isfinite(aq)
                 & (bid > 0) & (ask > bid) & (bq >= 0) & (aq >= 0))
    start = len(grid) - 60
    taker = np.flatnonzero(~fill["maker"].astype(bool))
    if len(taker) and np.any(fill["time_us"][taker] < grid[start] + arrival):
        raise AssertionError("Unscheduled IOC outside close window")
    prior = state.get("ioc_liquidity", {})
    pools = []
    pre = np.arange(0, start, 2)
    pre = pre[native_ok[pre]]
    for label, prices, displayed in (("bid", bid, bq), ("ask", ask, aq)):
        pool = [float(prior.get(label + "_price", 0)),
                float(prior.get(label + "_last_display", 0)),
                float(prior.get(label + "_available", 0))]
        if len(pre):
            changes = np.abs(prices[pre] - np.r_[pool[0], prices[pre[:-1]]]) > 1e-8
            if changes.any():
                # No IOC occurs before this day's close window. Once price
                # changes, its unconsumed book remains equal to display.
                pool = [float(prices[pre[-1]]), float(displayed[pre[-1]]), float(displayed[pre[-1]])]
            else:
                for i in pre:
                    shown = float(displayed[i])
                    pool[2] = min(shown, pool[2] + max(0., shown - pool[1]))
                    pool[1] = shown
        pools.append(pool)
    q0 = float(state.get("inventory", 0))
    reconstructed_q = np.rint((q0 + np.cumsum(fill["side"] * fill["qty"])) / lot) * lot
    matched = 0
    for i in range(start, len(grid), 2):
        if native_ok[i]:
            for pool, prices, displayed in zip(pools, (bid, ask), (bq, aq)):
                px, shown = float(prices[i]), float(displayed[i])
                if abs(pool[0] - px) > 1e-8:
                    pool[:] = [px, shown, shown]
                else:
                    pool[2] = min(shown, pool[2] + max(0., shown - pool[1]))
                    pool[1] = shown
        at_time = taker[fill["time_us"][taker] == grid[i] + arrival]
        if len(at_time) > 1:
            raise AssertionError("Multiple IOC fills on one close attempt")
        if not len(at_time):
            continue
        k = at_time[0]
        before = reconstructed_q[k-1] if k else q0
        side = int(fill["side"][k])
        if not before * side < 0 or not native_ok[i]:
            raise AssertionError("IOC increases inventory or lacks native book")
        pool = pools[0] if side == -1 else pools[1]
        close(fill["price"][k], pool[0], "IOC native arrival execution price")
        expected = math.floor(min(abs(before), pool[2]) / lot + 1e-8) * lot
        close(fill["qty"][k], expected, "IOC same-price liquidity not reused", atol=1e-7)
        pool[2] -= float(fill["qty"][k])
        matched += 1
    if matched != len(taker):
        raise AssertionError("IOC time is not a scheduled arrival")
    for label, pool in zip(("bid", "ask"), pools):
        for name, value in zip(("price", "last_display", "available"), pool):
            close(record["endstate"]["ioc_liquidity"][label + "_" + name], value,
                  "Carried IOC shadow liquidity ledger")
    return matched


def check_central_fills(market, day, g, forecasts, record, fill, state):
    grid = g["grid_us"]
    bn = market.startswith("binance")
    arrival = 100_000 if bn else 440_000
    lot = .001 if market == "hyperliquid_SKHX" else .01
    unit = math.floor(2000 / g["mid"][np.flatnonzero(np.isfinite(g["mid"]) & (g["mid"] > 0))[0]] / lot + 1e-8) * lot
    q0, cash0 = float(state.get("inventory", 0)), float(state.get("cash", 0))
    stamp, side, qty, px = (fill[n] for n in ("time_us", "side", "qty", "price"))
    maker = fill["maker"].astype(bool)
    if not np.all(np.diff(stamp) >= 0):
        raise AssertionError("Fill time ordering")
    if not np.all(qty > 0):
        raise AssertionError("Nonpositive fill")
    close(qty / lot, np.round(qty / lot), "Lot grid", atol=1e-6)
    rates = np.where(maker, -.5 if bn else -.03, .6273 if bn else .288) / 10000
    close(fill["fee"], qty * px * rates, "Fee cashflow")
    cash = cash0 - np.cumsum(side * qty * px + fill["fee"])
    q = q0 + np.cumsum(side * qty)
    close(cash, fill["cash"], "Independent native fill cash ledger")
    close(q, fill["inventory"], "Independent native fill inventory ledger")
    # Position is an integer number of lots; eliminate arithmetic remnants
    # before choosing flat/nonflat funding semantics and per-order capacity.
    q = np.rint(q / lot) * lot
    if len(q):
        close(record["endstate"]["inventory"], q[-1], "Final inventory")
    close(record["endstate"]["cash"], (cash[-1] if len(cash) else cash0)
          + record["metrics"]["funding_usd"], "Funding-adjusted final cash")
    close(record["metrics"]["fees_usd"], np.sum(fill["fee"]), "Daily fee total")
    before_q = np.r_[q0, q[:-1]] if len(q) else np.empty(0)
    close(fill["closed_qty"], np.where(before_q * side < 0,
                                       np.minimum(np.abs(before_q), qty), 0), "Closed quantities")
    close(record["metrics"]["unit_qty"], unit, "Ex-ante fixed daily unit")

    # Each one-second maker window has a constant price and fixed quantities.
    # Decision inventory is reconstructed from only fills before that decision.
    maker_indices = np.flatnonzero(maker)
    initial_bounds = {}
    native_demands = defaultdict(lambda: [0., None])
    policy = record["policy"]
    h, age = record["horizon"], record["max_age"]
    model = "own" if policy.startswith("own") else "crosslag1" if policy.startswith("lag") else "cross"
    predictions = forecasts[f"pred_exchange_{model}_h{h}_a{age}"]
    mask = expected_mask(g, forecasts, h, age)
    for k in maker_indices:
        elapsed = int(stamp[k] - grid[0] - arrival)
        i = elapsed // 2_000_000 * 2
        if i < 0 or i >= len(grid) or not grid[i] + arrival < stamp[k] < grid[i] + arrival + 1_000_000:
            raise AssertionError("Maker fill outside strict active interval")
        if not mask[i] or day in MISSING[market]:
            raise AssertionError("Maker order used invalid decision mask or incomplete day")
        at_decision = np.searchsorted(stamp, grid[i], side="left") - 1
        q_decision = q[at_decision] if at_decision >= 0 else q0
        amount = math.floor(min(unit, max(0., unit - side[k] * q_decision)) / lot + 1e-8) * lot
        bound_key = (i, int(side[k]))
        if bound_key not in initial_bounds:
            initial_bounds[bound_key] = [amount, 0.]
        initial_bounds[bound_key][1] += float(qty[k])
        if initial_bounds[bound_key][1] > amount + 1e-7:
            raise AssertionError("Order quantity resized or exceeded initial inventory capacity")
        alpha = 0. if policy == "mid" else float(predictions[i])
        step = tick(float(g["mid"][i]), market)
        skew = 0. if policy in ("mid", "fv") else step * q_decision / unit
        reservation = g["mid"][i] * (1 + alpha / 10000) - skew
        params = record["bid_params"] if side[k] == 1 else record["ask_params"]
        toxicity = (params["toxicity_negative_obi_bp"] if g["imbalance"][i] < 0
                    else params["toxicity_nonnegative_obi_bp"]) if policy.endswith("toxicity") else 0.
        raw = reservation - side[k] * (params["width_ticks"] * step + toxicity * g["mid"][i] / 10000)
        if side[k] == 1:
            predecessor = floor_price(g["ask"][i] - tick(g["ask"][i] * (1 - 1e-12), market), market)
            target = min(floor_price(raw, market), predecessor)
        else:
            successor = ceil_price(g["bid"][i] + tick(g["bid"][i], market), market)
            target = max(ceil_price(raw, market), successor)
        close(px[k], target, "Target price uses decision-time FV/inventory/toxicity", atol=1e-7)
        arrival_i = 100 if bn else 440
        if side[k] == 1 and not px[k] < g[f"truth_ask_{arrival_i}"][i]:
            raise AssertionError("Marketable bid was accepted as maker")
        if side[k] == -1 and not px[k] > g[f"truth_bid_{arrival_i}"][i]:
            raise AssertionError("Marketable ask was accepted as maker")
        demand = native_demands[(int(stamp[k]), int(side[k]))]
        if demand[1] is not None:
            close(demand[1], px[k], "Different simultaneous prices on same order side")
        demand[0] += float(qty[k])
        demand[1] = float(px[k])
    # No native print volume is reused within a strategy. Same-timestamp
    # distinct native trades are aggregated only within their own direction.
    for (when, our_side), (amount, our_price) in native_demands.items():
        lo, hi = np.searchsorted(g["tt_us"], [when, when], side="left")
        hi = int(np.searchsorted(g["tt_us"], when, side="right"))
        native_side = g["tside"][lo:hi]
        native_px = g["tprice"][lo:hi]
        eligible = native_side == -our_side
        eligible &= native_px <= our_price + 1e-8 if our_side == 1 else native_px >= our_price - 1e-8
        available = float(np.sum(g["tqty"][lo:hi][eligible]))
        if amount > available + 1e-7:
            raise AssertionError("Nonexistent/incorrect-side print or reused native volume")

    # Rebuild funding at each stored settlement using pre-settlement position.
    funding_grid = np.zeros(len(grid))
    for settlement in record["settlements"]:
        when = int(settlement["time_us"])
        i = np.searchsorted(stamp, when, side="left") - 1
        position = q[i] if i >= 0 else q0
        position = 0. if abs(position) < 1e-10 else position
        close(settlement["position"], position, "Funding position before settlement")
        expected = -position * settlement["settlement_price"] * settlement["rate"] if position else 0.
        close(expected, settlement["cashflow"], "Funding settlement arithmetic")
        funding_grid[np.searchsorted(grid, when, side="left")] += expected
    funding = np.cumsum(funding_grid)
    close(funding[-1], record["metrics"]["funding_usd"], "Funding total")
    at = np.searchsorted(stamp, grid, side="right") - 1
    cq, cc = np.full(len(grid), q0), np.full(len(grid), cash0)
    known = at >= 0
    if len(stamp):
        cq[known], cc[known] = q[at[known]], cash[at[known]]
    equity = cc + cq * mark_prices(g, state) + funding
    metrics = record["metrics"]
    close(equity[-1], metrics["end_equity_usd"], "End equity equals cash plus inventory mark")
    close(equity[-1] - metrics["start_equity_usd"], metrics["pnl_usd"], "Daily PnL identity")
    saved = OUT / "replay" / market / day / (key(h, age, 1, policy) + ".equity.npz")
    with np.load(saved) as z:
        close(z["equity"], equity, "Independent equity time series")
        close(z["inventory"], cq, "Independent inventory time series")
    prior_peak = float(state.get("equity_peak", max(0., metrics["start_equity_usd"])))
    peaks = np.maximum.accumulate(np.r_[prior_peak, equity])[1:]
    mdd = float(np.max(peaks - equity))
    close(metrics["mdd_usd"], mdd, "Daily chronological MDD")
    close(metrics["overall_mdd_usd"], max(float(state.get("overall_mdd", 0)), mdd), "Cumulative MDD")
    ioc_fills = check_ioc_liquidity(g, fill, state, record, lot, arrival)
    return dict(fills=len(stamp), maker_fills=len(maker_indices), ioc_fills=ioc_fills,
                native_print_groups=len(native_demands), fixed_order_sides=len(initial_bounds))


def verify(allow_partial=False):
    daily = pd.read_csv(OUT / "replay_daily.csv")
    summary = pd.read_csv(OUT / "replay_summary.csv")
    keys = ["market", "horizon", "max_age", "queue", "policy"]
    if not allow_partial and (len(daily) != 6528 or len(summary) != 384):
        raise AssertionError(f"Expected 6528 daily/384 summary rows; found {len(daily)}/{len(summary)}")
    if daily.duplicated(keys + ["day"]).any():
        raise AssertionError("Duplicate experiment row")
    total_groups = 0
    central_counts = defaultdict(int)
    max_driver_cash_error, max_driver_inventory_error = 0., 0.
    for config, block in daily.groupby(keys, sort=False):
        block = block.sort_values("day")
        if not allow_partial and list(block["day"]) != DAYS[1:]:
            raise AssertionError("Configuration does not cover all 17 forward dates")
        matched = summary
        for name, value in zip(keys, config):
            matched = matched[matched[name] == value]
        if len(matched) != 1:
            raise AssertionError("Summary configuration mapping")
        result = matched.iloc[0]
        close(block["pnl_usd"].sum(), result["total_pnl_usd"], "Summary sums daily account PnL")
        close(block["pnl_usd"].sum(), block["end_equity_usd"].iloc[-1], "Total PnL equals terminal marked equity")
        close(block["end_equity_usd"].to_numpy()[:-1], block["start_equity_usd"].to_numpy()[1:], "Continuous account across days")
        close(block["funding_usd"].sum(), result["funding_usd"], "Summary funding")
        close(block["overall_mdd_usd"].max(), result["mdd_usd"], "Summary MDD")
        sd = block["pnl_usd"].std(ddof=1)
        if sd > 0:
            close(block["pnl_usd"].mean() / sd * np.sqrt(252), result["sharpe"], "Daily Sharpe arithmetic")
        for _, row in block.iterrows():
            path = OUT / "replay" / row.market / row.day / (key(row.horizon, row.max_age, row.queue, row.policy) + ".json")
            saved = json_load(path)
            max_driver_cash_error = max(max_driver_cash_error, saved["ledger_validation"]["cash_max_error"])
            max_driver_inventory_error = max(max_driver_inventory_error, saved["ledger_validation"]["inventory_max_error"])
            close(saved["endstate"]["cash"] + saved["endstate"]["inventory"] * saved["endstate"]["last_mid"],
                  row.end_equity_usd, "Every profile terminal cash/inventory ledger")
        total_groups += 1
    # All six policies share the causal opportunity mask within each profile.
    comparison_groups = 0
    for _, block in daily.groupby(["market", "day", "horizon", "max_age", "queue"]):
        if not allow_partial and set(block.policy) != set(POLICIES):
            raise AssertionError("Incomplete paired policy comparison")
        if block.decision_opportunities.nunique() != 1:
            raise AssertionError("Policies have unequal decision opportunity masks")
        comparison_groups += 1
    coverage_rows = []
    for market in MARKETS:
        states = {}
        for day in DAYS[1:]:
            block = daily[(daily.market == market) & (daily.day == day)]
            if block.empty:
                continue
            if not block.complete_trade_day.eq(day not in MISSING[market]).all():
                raise AssertionError("Incomplete trade coverage declaration")
            if day in MISSING[market] and (block.maker_fills.sum() or block.decision_opportunities.sum()):
                raise AssertionError("Maker orders on incomplete trade coverage day")
            coverage_rows.append(dict(market=market, day=day, complete_trade_day=day not in MISSING[market],
                                      profiles=len(block)))
            with np.load(OUT / "inputs" / market / (day + ".npz")) as z:
                need = ("grid_us", "mid", "bid", "ask", "bq", "aq", "imbalance", "tt_us", "tside", "tprice", "tqty")
                g = {n: z[n] for n in need}
                lag = 100 if market.startswith("binance") else 440
                for n in ("bid", "ask", "bq", "aq", "event_us"):
                    g[f"truth_{n}_{lag}"] = z[f"truth_{n}_{lag}"]
            forecast = load(OUT / "forecasts" / market / (day + ".npz"))
            for h in HORIZONS:
                for age in AGES:
                    expected = int(expected_mask(g, forecast, h, age).sum()) if day not in MISSING[market] else 0
                    rows = block[(block.horizon == h) & (block.max_age == age)]
                    if len(rows) and not rows.decision_opportunities.eq(expected).all():
                        raise AssertionError("Decision opportunity count uses a noncausal or unequal mask")
            for policy in POLICIES:
                record_path = OUT / "replay" / market / day / (key(1, 1, 1, policy) + ".json")
                if not record_path.exists():
                    continue
                record = json_load(record_path)
                fills = load(record_path.with_suffix(".fills.npz"))
                try:
                    counts = check_central_fills(market, day, g, forecast, record, fills, states.get(policy, {}))
                except Exception as exc:
                    raise AssertionError(f"{market}/{day}/{policy}: {exc}") from exc
                for name, value in counts.items():
                    central_counts[name] += value
                central_counts["ledgers"] += 1
                states[policy] = record["endstate"]
    parameters = check_parameters()
    alarm = summary["performance_alarm"].astype(bool)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "work/run_fv_mm.py", ROOT / "work/fv_mm_calibrate.py",
                               ROOT / "work/fv_mm_replay.py", Path(__file__))}
    manifest_hashes = json_load(OUT / "manifest.json")["source_hashes"]
    if any(source_hashes[name] != value for name, value in manifest_hashes.items()):
        raise AssertionError("Final source files differ from run manifest hashes")
    checks = dict(status="pass", full_grid_complete=len(daily) == 6528 and len(summary) == 384,
                  daily_rows=len(daily), summary_rows=len(summary), configuration_groups=total_groups,
                  paired_policy_groups=comparison_groups, prior_only_parameter_selections=parameters,
                  central_ledger_validation=dict(central_counts), coverage=coverage_rows,
                  driver_max_cash_error=max_driver_cash_error,
                  driver_max_inventory_error=max_driver_inventory_error,
                  final_source_hashes_match_manifest=True,
                  performance_alarm_profiles=int(alarm.sum()),
                  economic_alarm_interpretation="Requires scrutiny; high ratios are not automatically valid performance or automatically a ledger error.",
                  evidence=dict(raw_parquet_asof_values_checked=448,
                                input_manifest="out_fv_mm/inputs/manifest.json",
                                input_samples="out_fv_mm/inputs/validation_samples.json"),
                  limits=["Central h1/a1/q1 profiles have detailed fill-ledger validation; other profiles have daily account and aggregate checks.",
                          "No monotonicity requirement between queue scenarios; neither is a mathematical PnL bound.",
                          "Native print volume is a counterfactual fill proxy, not proof of personal queue execution.",
                          "Exchange/collector clock synchronization and equal-timestamp ordering remain unverified.",
                          "Calibration is reconstructed from past surfaces; surfaces themselves use observed future labels only after that day.",
                          "Different raw and deduplicated native HL trade counts are expected, not missing files."],
                  source_hashes=source_hashes)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = OUT / "final_validation.json"
    try:
        checks = verify(args.allow_partial)
    except Exception as exc:
        output.write_text(json.dumps(dict(status="fail", error=repr(exc)), indent=2) + "\n")
        raise
    output.write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in checks.items() if k not in ("coverage", "limits", "source_hashes")}, indent=2))


if __name__ == "__main__":
    main()
