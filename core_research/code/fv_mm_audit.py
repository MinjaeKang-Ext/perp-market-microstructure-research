"""Metadata coverage and bounded samples for four perpetual contracts.

This is an input audit, not a full-row validity or execution audit.
"""
from pathlib import Path
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "perp_data/trading_data/processed"
OUT = ROOT / "out_fv_mm/audit.json"
CONTRACTS = [("binance", "SKHYNIXUSDT"), ("binance", "SKHYUSDT"),
             ("hyperliquid", "xyz_SKHX"), ("hyperliquid", "xyz_SKHY")]


def inspect_type(folder):
    files = sorted(folder.rglob("*.parquet"))
    daily = {}
    late = []
    schemas = Counter()
    for file in files:
        pf = pq.ParquetFile(file)
        date = next(p[5:] for p in file.parts if p.startswith("date="))
        item = daily.setdefault(date, {"files": 0, "rows": 0})
        item["files"] += 1
        item["rows"] += pf.metadata.num_rows
        schemas[str(pf.schema_arrow.remove_metadata())] += 1
        if "late_parts" in file.parts:
            late.append(str(file.relative_to(SOURCE)))
    missing = []
    for date in daily:
        for hour in range(24):
            for minute in range(0, 60, 10):
                part = folder / f"date={date}" / f"hour={hour:02d}" / f"minute={minute:02d}" / "data.parquet"
                if not part.exists():
                    missing.append(str(part.relative_to(SOURCE)))
    return dict(files=len(files), rows=sum(x["rows"] for x in daily.values()),
                daily=daily, schema_variants=dict(schemas), late_parts=late,
                missing_nominal_data_partitions=missing)


def sample_bbo(folder):
    """Two deterministic partitions/day; minimum observed increment is a proxy.

    Samples do not prove an exchange price filter. HL prices around 1000 must
    use the correct five-significant-digit grid; a whole-day tick is unsafe.
    """
    result = []
    for directory in sorted(folder.glob("date=*")):
        files = sorted(directory.rglob("*.parquet"))
        selected = list(dict.fromkeys([files[0], files[len(files) // 2]]))
        cols = ["event_time", "received_at", "best_bid_price", "best_ask_price"]
        schema = pq.ParquetFile(files[0]).schema_arrow
        if "exchange_event_time" in schema.names:
            cols.append("exchange_event_time")
        frames = [pq.ParquetFile(file).read(columns=cols) for file in selected]
        values = {name: np.concatenate([t[name].to_numpy() for t in frames]) for name in cols}
        bid = values["best_bid_price"].astype(float)
        ask = values["best_ask_price"].astype(float)
        pixels = np.unique(np.round(np.concatenate([bid, ask]), 8))
        diffs = np.diff(pixels)
        diffs = diffs[diffs > 1e-8]
        lag_ms = (values["received_at"].astype("datetime64[us]").astype(np.int64)
                  - values["event_time"].astype("datetime64[us]").astype(np.int64)) / 1000
        item = dict(date=directory.name[5:], sample_rows=len(bid),
                    sampled_partitions=[str(f.relative_to(SOURCE)) for f in selected],
                    bid_min=float(bid.min()), ask_max=float(ask.max()),
                    observed_min_price_increment=float(np.round(diffs.min(), 8)) if len(diffs) else None,
                    recv_minus_event_ms_p50=float(np.median(lag_ms)),
                    recv_minus_event_ms_p99=float(np.quantile(lag_ms, .99)))
        if "exchange_event_time" in values:
            delta = (values["exchange_event_time"].astype("datetime64[us]").astype(np.int64)
                     - values["event_time"].astype("datetime64[us]").astype(np.int64)) / 1000
            item["exchange_event_minus_event_ms_p50"] = float(np.median(delta))
            item["exchange_event_minus_event_ms_max"] = float(delta.max())
        result.append(item)
    return result


def main():
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    record = dict(scope="Four perpetual contracts only; spot excluded", contracts={},
                  caveat="Parquet metadata counts include late_parts; BBO quality/ticks are bounded samples.",
                  inherited_settings={
                      "sampling_seconds": 1, "horizons_seconds": [1, 5, 30, 60],
                      "age_limits_seconds": [1, 5], "fit": "daily expanding using matured past labels",
                      "maker_fee_bp": {"binance": -.5, "hyperliquid": -.03},
                      "taker_fee_bp": {"binance": .6273, "hyperliquid": .288},
                      "historical_order_size_usd": 2000,
                      "binance_total_reaction_assumption_ms": 100,
                      "binance_total_cancel_assumption_ms": 170,
                      "hyperliquid_decision_to_ack_measured_median_ms": 440,
                      "hyperliquid_cancel_measured_median_ms": 750,
                      "hyperliquid_feed_reference_median_ms": 195,
                      "latency_caveat": "ACK latency is not measured one-way activation. Add fixed feed delay only for event-clock signals; received_at signals already contain observed feed delay.",
                      "evidence": ["mm_stage1/config.py FEES_BP", "REPORT_stage8_mm_replan.md sections 6-E-0, 6-G, 6-H", "REPORT_stage8_mm_replan.md line 1557; mm_stage1/stage9_qr.py SIZE_USD"]})
    record["tick_rule"] = {
        "source": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size",
        "perp_max_decimals": 6, "max_significant_figures": 5,
        "integer_prices_always_allowed": True,
        "boundary_caveat": "Grid depends on each candidate price; do not infer tick from a full future day or hardcode an October 28 switch.",
    }
    if "live_metadata" in previous:
        record["live_metadata"] = previous["live_metadata"]
    for venue, symbol in CONTRACTS:
        folder = SOURCE / venue / symbol
        sub = "agg_trades" if venue == "binance" else "trades"
        data = {kind: inspect_type(folder / kind) for kind in ["bbo", sub, "l2_snapshots"]}
        data["bbo_samples"] = sample_bbo(folder / "bbo")
        data["aggressor_mapping"] = "buyer_is_maker True=sell, False=buy" if venue == "binance" else "side B=buy, A=sell; inherited empirical interpretation"
        record["contracts"][f"{venue}_{symbol}"] = data
        print(venue, symbol, {kind: (x["files"], x["rows"]) for kind, x in data.items() if isinstance(x, dict) and "files" in x}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
