"""Causal 1-second BBO features and walk-forward midpoint forecasts.

All time-based design choices in this file were confirmed by the user:
two common-share perps, 1/5/30/60-second horizons, 1-second sampling,
expanding daily fit, and 1/5-second quote-age sensitivity analyses.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "perp_data/trading_data/processed"
OUT = ROOT / "work/orderbook_fair_value"
KST = ZoneInfo("Asia/Seoul")
START = datetime(2026, 10, 15, tzinfo=KST)
END = datetime(2026, 11, 2, tzinfo=KST)
HORIZONS = (1, 5, 30, 60)
AGES = (1, 5)
MARKETS = {
    "binance_SKHYNIXUSDT": {
        "path": SOURCE / "binance/SKHYNIXUSDT/bbo",
        "event": "exchange_event_time",
        "sequence": "update_id",
    },
    "hyperliquid_SKHX": {
        "path": SOURCE / "hyperliquid/xyz_SKHX/bbo",
        "event": "event_time",
        "sequence": None,
    },
}


def as_us(column):
    return column.to_numpy().astype("datetime64[us]").astype("int64")


def path_for(root: Path, hour: datetime, minute: int):
    return root / f"date={hour:%Y-%m-%d}" / f"hour={hour:%H}" / f"minute={minute:02d}" / "data.parquet"


def read_hour(market: str, hour: datetime):
    cfg = MARKETS[market]
    previous = hour - timedelta(hours=1)
    files = [path_for(cfg["path"], previous, 50)] + [path_for(cfg["path"], hour, minute) for minute in range(0, 60, 10)]
    files = [file for file in files if file.exists()]
    if len(files) < 6:
        raise FileNotFoundError(f"Missing BBO partition near {hour}: {files}")
    columns = [cfg["event"], "received_at", "best_bid_price", "best_ask_price", "best_bid_quantity", "best_ask_quantity"]
    if cfg["sequence"]:
        columns.append(cfg["sequence"])
    pieces = {name: [] for name in columns}
    for file in files:
        table = pq.read_table(file, columns=columns)
        for name in columns:
            pieces[name].append(as_us(table[name]) if name in (cfg["event"], "received_at") else table[name].to_numpy())
    data = {name: np.concatenate(parts) for name, parts in pieces.items()}
    event_us = data[cfg["event"]]
    recv_us = data["received_at"]
    seq = data[cfg["sequence"]].astype("int64") if cfg["sequence"] else event_us
    bid = data["best_bid_price"].astype(float)
    ask = data["best_ask_price"].astype(float)
    bq = data["best_bid_quantity"].astype(float)
    aq = data["best_ask_quantity"].astype(float)
    good = (bid > 0) & (ask > bid) & (bq > 0) & (aq > 0) & (recv_us >= event_us)
    event_us, recv_us, seq, bid, ask, bq, aq = (array[good] for array in (event_us, recv_us, seq, bid, ask, bq, aq))
    return event_us, recv_us, seq, bid, ask, bq, aq


def sample_state(grid_us, stamp_us, event_us, bid, ask, bq, aq):
    index = np.searchsorted(stamp_us, grid_us, side="right") - 1
    present = index >= 0
    safe = np.maximum(index, 0)
    mid = (bid[safe] + ask[safe]) / 2
    spread = ask[safe] - bid[safe]
    imbalance = (bq[safe] - aq[safe]) / (bq[safe] + aq[safe])
    age_sec = (grid_us - event_us[safe]) / 1_000_000
    for arr in (mid, spread, imbalance, age_sec):
        arr[~present] = np.nan
    return index, mid, spread, imbalance, age_sec


def extract_hour(market: str, hour: datetime):
    event_us, recv_us, seq, bid, ask, bq, aq = read_hour(market, hour)
    grid_start = int(hour.timestamp() * 1_000_000)
    grid = np.arange(grid_start, grid_start + 3_600_000_000, 1_000_000, dtype="int64")

    # Offline exchange-book truth: exchange event clock, tie-broken by update ID.
    event_order = np.lexsort((seq, event_us))
    ei, truth_mid, _, _, truth_age = sample_state(
        grid, event_us[event_order], event_us[event_order],
        bid[event_order], ask[event_order], bq[event_order], aq[event_order],
    )

    # Causal observation: only messages ingested by the collector by time t.
    recv_order = np.lexsort((seq, recv_us))
    sorted_seq = seq[recv_order]
    keep = np.empty(len(sorted_seq), dtype=bool)
    keep[0] = True
    keep[1:] = sorted_seq[1:] > np.maximum.accumulate(sorted_seq[:-1])
    order = recv_order[keep]
    ri, mid, spread, imbalance, age = sample_state(
        grid, recv_us[order], event_us[order], bid[order], ask[order], bq[order], aq[order],
    )
    safe = np.maximum(ri, 0)
    depth = bq[order][safe] + aq[order][safe]
    weighted_offset_bp = 5_000 * spread * imbalance / mid
    spread_bp = 10_000 * spread / mid

    # Cont-style event OFI, accumulated only over messages already received.
    bb, ba = bid[order], ask[order]
    qb, qa = bq[order], aq[order]
    ofi = np.zeros(len(order), dtype=float)
    ofi[1:] = (
        (bb[1:] >= bb[:-1]) * qb[1:] - (bb[1:] <= bb[:-1]) * qb[:-1]
        - (ba[1:] <= ba[:-1]) * qa[1:] + (ba[1:] >= ba[:-1]) * qa[:-1]
    )
    cum_ofi = np.cumsum(ofi)
    previous_index = np.searchsorted(recv_us[order], grid - 1_000_000, side="right") - 1
    prev_cum = np.where(previous_index >= 0, cum_ofi[np.maximum(previous_index, 0)], 0)
    ofi_one_sec = (cum_ofi[safe] - prev_cum) / depth
    ofi_one_sec[ri < 0] = np.nan

    # Receive-time future midpoint is retained as a separate latency diagnostic.
    result = pd.DataFrame({
        "grid_us": grid,
        "recv_mid": mid,
        "recv_age_sec": age,
        "exchange_mid": truth_mid,
        "exchange_age_sec": truth_age,
        "imbalance": imbalance,
        "spread_bp": spread_bp,
        "weighted_offset_bp": weighted_offset_bp,
        "ofi_one_sec_norm": ofi_one_sec,
        "recv_event_us": np.where(ri >= 0, event_us[order][safe], -1),
        "truth_event_us": np.where(ei >= 0, event_us[event_order][np.maximum(ei, 0)], -1),
    })
    return result


def extract(market: str | None = None):
    selected = [market] if market else list(MARKETS)
    for name in selected:
        folder = OUT / "grid" / name
        folder.mkdir(parents=True, exist_ok=True)
        day = START
        while day < END:
            output = folder / f"{day:%Y-%m-%d}.parquet"
            if output.exists():
                print(f"SKIP {output}", flush=True)
            else:
                frames = [extract_hour(name, day + timedelta(hours=hour)) for hour in range(24)]
                frame = pd.concat(frames, ignore_index=True)
                if len(frame) != 86_400 or not np.all(np.diff(frame["grid_us"].to_numpy()) == 1_000_000):
                    raise AssertionError(f"Grid broken for {name} {day:%Y-%m-%d}")
                temporary = output.with_suffix(".part")
                frame.to_parquet(temporary, index=False)
                temporary.replace(output)
                print(f"WROTE {output} rows={len(frame)}", flush=True)
            day += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["extract"])
    parser.add_argument("--market", choices=list(MARKETS))
    parser.add_argument("--only-day", help="Restrict to one KST date for diagnostics")
    args = parser.parse_args()
    if args.only_day:
        day = datetime.fromisoformat(args.only_day).replace(tzinfo=KST)
        for name in ([args.market] if args.market else MARKETS):
            frame = pd.concat([extract_hour(name, day + timedelta(hours=h)) for h in range(24)], ignore_index=True)
            print(name, day.date(), len(frame), frame.isna().mean().to_dict())
        return
    extract(args.market)


if __name__ == "__main__":
    main()
