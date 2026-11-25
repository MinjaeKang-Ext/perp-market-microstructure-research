"""Build causal 1-second BBO features and native trade inputs for four perps.

Features are ordered by received_at; execution-book queries use native BBO
event time. Binance trade execution uses event_time (trade time), never the
later exchange_event_time publication field. All late_parts are included.
No fills, positions or trading returns are inferred by this input builder.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "perp_data/trading_data/processed"
OUT = ROOT / "out_fv_mm/inputs"
KST = ZoneInfo("Asia/Seoul")
START = datetime(2026, 10, 15, tzinfo=KST)
END = datetime(2026, 11, 2, tzinfo=KST)
OFFSETS_MS = (0, 100, 440, 1100, 1170, 1440, 1750)
MARKETS = {
    "binance_SKHYNIXUSDT": ("binance", "SKHYNIXUSDT"),
    "binance_SKHYUSDT": ("binance", "SKHYUSDT"),
    "hyperliquid_SKHX": ("hyperliquid", "xyz_SKHX"),
    "hyperliquid_SKHY": ("hyperliquid", "xyz_SKHY"),
}
VERSION = 2


def us(column):
    return column.to_numpy().astype("datetime64[us]").astype(np.int64)


def partition(root: Path, dt: datetime):
    return root / f"date={dt:%Y-%m-%d}" / f"hour={dt:%H}" / f"minute={dt.minute // 10 * 10:02d}"


def files_in_slot(root: Path, dt: datetime):
    return sorted(partition(root, dt).rglob("*.parquet"))


def _advancing_order(stamps, sequences, allow_equal=False):
    """No stale/duplicate sequence may overwrite a causally newer book."""
    order = np.lexsort((sequences, stamps))
    if not len(order):
        return order
    seq = sequences[order]
    keep = np.empty(len(seq), dtype=bool)
    keep[0] = True
    previous_max = np.maximum.accumulate(seq[:-1])
    keep[1:] = seq[1:] >= previous_max if allow_equal else seq[1:] > previous_max
    return order[keep]


def _sample(query, stamps, event, bid, ask, bq, aq):
    index = np.searchsorted(stamps, query, side="right") - 1
    good = index >= 0
    safe = np.maximum(index, 0)
    def take(values):
        result = values[safe].astype(float)
        result[~good] = np.nan
        return result
    age = (query - event[safe]) / 1_000_000
    age[~good] = np.nan
    return index, take(bid), take(ask), take(bq), take(aq), age


def read_bbo_hour(market: str, hour: datetime):
    venue, contract = MARKETS[market]
    root = SOURCE / venue / contract / "bbo"
    # Carry state/OFI from previous 10m, and truth at hour-end +1750ms from
    # the following partition. This is not future feature information.
    slots = [hour + timedelta(minutes=minute) for minute in range(-10, 70, 10)]
    files = [f for dt in slots for f in files_in_slot(root, dt)]
    current = [f for dt in slots[1:7] for f in files_in_slot(root, dt)]
    if len(current) < 6:
        raise FileNotFoundError(f"Missing BBO partition: {market} {hour}")
    event_name = "exchange_event_time" if venue == "binance" else "event_time"
    cols = [event_name, "received_at", "best_bid_price", "best_ask_price",
            "best_bid_quantity", "best_ask_quantity"]
    if venue == "binance":
        cols.append("update_id")
    pieces = {col: [] for col in cols}
    for path in files:
        table = pq.ParquetFile(path).read(columns=cols)
        for col in cols:
            pieces[col].append(us(table[col]) if col in (event_name, "received_at")
                               else table[col].to_numpy())
    data = {name: np.concatenate(parts) for name, parts in pieces.items()}
    event, received = data[event_name], data["received_at"]
    seq = data["update_id"] if venue == "binance" else event
    bid, ask = data["best_bid_price"], data["best_ask_price"]
    bq, aq = data["best_bid_quantity"], data["best_ask_quantity"]
    valid = (np.isfinite(bid) & np.isfinite(ask) & np.isfinite(bq) & np.isfinite(aq)
             & (bid > 0) & (ask > bid) & (bq > 0) & (aq > 0)
             & (received >= event))
    return tuple(x[valid] for x in (event, received, seq, bid, ask, bq, aq))


def extract_hour(market: str, hour: datetime):
    event, received, seq, bid, ask, bq, aq = read_bbo_hour(market, hour)
    if not len(event):
        raise ValueError(f"No valid book: {market} {hour}")
    start_us = int(hour.timestamp() * 1_000_000)
    grid = start_us + np.arange(3600, dtype=np.int64) * 1_000_000
    is_hl = market.startswith("hyperliquid")
    order = _advancing_order(received, seq, allow_equal=is_hl)
    rr, ee = received[order], event[order]
    bb, aa, qb, qa = (values[order] for values in (bid, ask, bq, aq))
    ix, b, a, q_b, q_a, age = _sample(grid, rr, ee, bb, aa, qb, qa)
    safe = np.maximum(ix, 0)
    mid, spread = (b + a) / 2, a - b
    imbalance = (q_b - q_a) / (q_b + q_a)
    ofi = np.zeros(len(order), dtype=np.float64)
    ofi[1:] = ((bb[1:] >= bb[:-1]) * qb[1:] - (bb[1:] <= bb[:-1]) * qb[:-1]
               - (aa[1:] <= aa[:-1]) * qa[1:] + (aa[1:] >= aa[:-1]) * qa[:-1])
    cumulative = ofi.cumsum()
    prev = np.searchsorted(rr, grid - 1_000_000, side="right") - 1
    before = np.where(prev >= 0, cumulative[np.maximum(prev, 0)], 0)
    normalized = (cumulative[safe] - before) / (q_b + q_a)
    normalized[ix < 0] = np.nan
    recv_age = (grid - rr[safe]) / 1_000_000
    recv_age[ix < 0] = np.nan
    result = {
        "grid_us": grid, "bid": b, "ask": a, "bq": q_b, "aq": q_a,
        "mid": mid, "age_sec": age, "recv_age_sec": recv_age,
        "weighted_offset_bp": 5_000 * spread * imbalance / mid,
        "imbalance": imbalance, "spread_bp": 10_000 * spread / mid,
        "ofi_one_sec_norm": normalized,
        "recv_event_us": np.where(ix >= 0, ee[safe], -1),
        "recv_stamp_us": np.where(ix >= 0, rr[safe], -1),
    }
    # HL does not supply sequence numbers. Equal native timestamps can carry
    # different states; retain them in receipt order instead of dropping all
    # except the first update in that millisecond.
    truth_order = (np.lexsort((received, event)) if is_hl
                   else _advancing_order(event, seq))
    te = event[truth_order]
    tb, ta, tqb, tqa = (values[truth_order] for values in (bid, ask, bq, aq))
    for ms in OFFSETS_MS:
        query = grid + ms * 1000
        tx, xb, xa, xqb, xqa, xage = _sample(query, te, te, tb, ta, tqb, tqa)
        for label, values in (("bid", xb), ("ask", xa), ("bq", xqb), ("aq", xqa),
                              ("age", xage), ("mid", (xb + xa) / 2)):
            result[f"truth_{label}_{ms}"] = values
        result[f"truth_event_us_{ms}"] = np.where(tx >= 0, te[np.maximum(tx, 0)], -1)
    return result


def extract_l2(market: str, day: datetime):
    """Sparse native-event snapshot book for price-specific queue scenarios.

    These are infrequent snapshots, not continuously reconstructed L2. Their
    age must remain explicit when using the cached quantity as a queue proxy.
    """
    venue, contract = MARKETS[market]
    root = SOURCE / venue / contract / "l2_snapshots"
    files = sorted((root / f"date={day:%Y-%m-%d}").rglob("*.parquet"))
    files += files_in_slot(root, day - timedelta(minutes=10))
    files += files_in_slot(root, day + timedelta(days=1))
    files = sorted(set(files))
    event_name = "exchange_event_time" if venue == "binance" else "event_time"
    columns = [event_name, "received_at", "side", "level", "price", "quantity"]
    frames = [pq.ParquetFile(path).read(columns=columns).to_pandas() for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["event_us"] = frame[event_name].dt.as_unit("us").astype("int64")
    frame["recv_us"] = frame["received_at"].dt.as_unit("us").astype("int64")
    key = ["event_us", "recv_us", "side", "level"]
    frame = frame.drop_duplicates(key + ["price", "quantity"])
    if frame.duplicated(key).any():
        raise ValueError(f"Conflicting L2 snapshot levels: {market} {day}")
    frame = frame.sort_values(key, kind="stable")
    if not frame["side"].isin(["bid", "ask"]).all():
        raise ValueError("Unknown L2 side")
    groups = frame.groupby(["event_us", "recv_us"], sort=False).size()
    if not groups.eq(40).all():
        raise ValueError(f"L2 snapshots must have 40 levels: {market} {day}")
    events = groups.index.get_level_values(0).to_numpy(np.int64)
    receives = groups.index.get_level_values(1).to_numpy(np.int64)
    output = {"l2_event_us": events, "l2_recv_us": receives}
    for side in ("bid", "ask"):
        selected = frame[frame["side"].eq(side)]
        levels = selected["level"].to_numpy().reshape(-1, 20)
        if not np.all(levels == np.arange(1, 21)):
            raise ValueError("L2 side levels incomplete")
        output[f"l2_{side}_px"] = selected["price"].to_numpy(float).reshape(-1, 20)
        output[f"l2_{side}_qty"] = selected["quantity"].to_numpy(float).reshape(-1, 20)
    if not np.all(np.diff(events) >= 0):
        raise AssertionError("L2 event order broken")
    target = OUT / market / f"{day:%Y-%m-%d}.l2.npz"
    temporary = target.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **output)
    temporary.replace(target)
    return len(events)


def read_trades(market: str, day: datetime):
    venue, contract = MARKETS[market]
    kind = "agg_trades" if venue == "binance" else "trades"
    root = SOURCE / venue / contract / kind
    files = sorted((root / f"date={day:%Y-%m-%d}").rglob("*.parquet"))
    files += files_in_slot(root, day - timedelta(minutes=10))
    files += files_in_slot(root, day + timedelta(days=1))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No trades: {market} {day:%Y-%m-%d}")
    id_name = "aggregate_trade_id" if venue == "binance" else "trade_id"
    side_name = "buyer_is_maker" if venue == "binance" else "side"
    cols = ["event_time", id_name, side_name, "price", "quantity"]
    pieces = {col: [] for col in cols}
    for path in files:
        table = pq.ParquetFile(path).read(columns=cols)
        for col in cols:
            pieces[col].append(us(table[col]) if col == "event_time" else table[col].to_numpy())
    data = {name: np.concatenate(parts) for name, parts in pieces.items()}
    tt, ids = data["event_time"], data[id_name]
    price, qty = data["price"].astype(float), data["quantity"].astype(float)
    if venue == "binance":
        side = np.where(data[side_name], -1, 1).astype(np.int8)
    else:
        if not np.isin(data[side_name], ["A", "B"]).all():
            raise ValueError(f"Unknown native trade side: {market}")
        side = np.where(data[side_name] == "B", 1, -1).astype(np.int8)
    if not (np.isfinite(price) & np.isfinite(qty) & (price > 0) & (qty > 0)).all():
        raise ValueError(f"Invalid trade price/quantity: {market}")
    raw = len(tt)
    order = np.lexsort((ids, tt))
    tt, ids, side, price, qty = (x[order] for x in (tt, ids, side, price, qty))
    duplicate = np.zeros(len(tt), dtype=bool)
    duplicate[1:] = (tt[1:] == tt[:-1]) & (ids[1:] == ids[:-1])
    conflict = duplicate[1:] & ((side[1:] != side[:-1]) | (price[1:] != price[:-1])
                               | (qty[1:] != qty[:-1]))
    if conflict.any():
        raise ValueError(f"Conflicting duplicate native trades: {market}")
    start = int(day.timestamp() * 1_000_000)
    end = start + 86_400_000_000
    # Include next-day executions only through the last possible cancel query.
    # This allows current-day orders to remain alive across midnight.
    keep = ~duplicate & (tt >= start) & (tt < end + max(OFFSETS_MS) * 1000)
    return {"tt_us": tt[keep], "tside": side[keep], "tprice": price[keep],
            "tqty": qty[keep]}, {
                "trade_files_read_with_neighbours": len(files),
                "trade_rows_read_with_neighbours": raw,
                "trade_duplicates_removed_with_neighbours": int(duplicate.sum()),
                "trades_current_day": int(np.sum(keep & (tt < end))),
                "trades_next_day_tail": int(np.sum(keep & (tt >= end))),
            }


def validate(data):
    grid = data["grid_us"]
    if len(grid) != 86400 or not np.all(np.diff(grid) == 1_000_000):
        raise AssertionError("Broken one-second grid")
    known = data["recv_stamp_us"] >= 0
    if not (np.all(data["recv_stamp_us"][known] <= grid[known])
            and np.all(data["recv_event_us"][known] <= data["recv_stamp_us"][known])):
        raise AssertionError("Future data in receive-time features")
    for ms in OFFSETS_MS:
        event = data[f"truth_event_us_{ms}"]
        good = event >= 0
        if not np.all(event[good] <= grid[good] + ms * 1000):
            raise AssertionError("Future book in execution query")
    if not np.all(np.diff(data["tt_us"]) >= 0):
        raise AssertionError("Trades unsorted")


def extract_day(market: str, day: datetime, force=False):
    directory = OUT / market
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{day:%Y-%m-%d}.npz"
    if output.exists() and not force:
        with np.load(output) as prior:
            same_version = json.loads(str(prior["metadata_json"]))["version"] == VERSION
        if same_version and output.with_suffix(".l2.npz").exists():
            print(f"SKIP {market} {day:%Y-%m-%d}", flush=True)
            return
    start = time.monotonic()
    hours = [extract_hour(market, day + timedelta(hours=h)) for h in range(24)]
    data = {name: np.concatenate([hour[name] for hour in hours]) for name in hours[0]}
    del hours
    trades, trade_meta = read_trades(market, day)
    data.update(trades)
    validate(data)
    l2_snapshots = extract_l2(market, day)
    metadata = dict(version=VERSION, market=market, day=f"{day:%Y-%m-%d}",
                    rows=len(data["grid_us"]), offsets_ms=list(OFFSETS_MS),
                    feature_clock="received_at", execution_book_clock=(
                        "exchange_event_time" if market.startswith("binance") else "event_time"),
                    trade_clock="event_time", l2_snapshots_with_neighbours=l2_snapshots,
                    **trade_meta)
    metadata["feature_missing_rows"] = int(np.sum(~np.isfinite(data["mid"])))
    data["metadata_json"] = np.array(json.dumps(metadata))
    temporary = output.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"WROTE {market} {day:%Y-%m-%d} trades={len(data['tt_us']):,} "
          f"missing={metadata['feature_missing_rows']} seconds={time.monotonic()-start:.1f}", flush=True)


def selftest():
    stamps = np.array([20, 10, 30, 25, 30])
    seq = np.array([2, 1, 3, 1, 4])
    order = _advancing_order(stamps, seq)
    assert np.array_equal(seq[order], [1, 2, 3, 4])
    equal_order = _advancing_order(np.array([10, 11, 12, 13]),
                                   np.array([1, 1, 0, 2]), allow_equal=True)
    assert np.array_equal(equal_order, [0, 1, 3])
    query = np.array([9, 10, 19, 20, 30])
    time_stamps = np.array([10, 20])
    bid = np.array([99., 100.])
    ix, b, a, bq, aq, age = _sample(query, time_stamps, time_stamps, bid,
                                   bid + 2, bid * 0 + 3, bid * 0 + 4)
    assert np.isnan(b[0]) and np.array_equal(b[1:], [99, 99, 100, 100])
    assert np.all(time_stamps[np.maximum(ix, 0)][ix >= 0] <= query[ix >= 0])
    print("PASS advancing-sequence, equal-HL-timestamp and as-of causality fixtures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=MARKETS, action="append")
    parser.add_argument("--day", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    days = ([datetime.fromisoformat(day).replace(tzinfo=KST) for day in args.day]
            if args.day else [START + timedelta(days=d) for d in range((END-START).days)])
    for day in days:
        for market in args.market or MARKETS:
            extract_day(market, day, force=args.force)


if __name__ == "__main__":
    main()
