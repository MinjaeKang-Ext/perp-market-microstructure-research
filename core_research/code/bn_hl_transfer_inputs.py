"""Causal 100ms BBO grids for Binance -> Hyperliquid transfer diagnostics.

Reuses the canonical 1-second input reader/sequence policy. Receive-clock
states are available by grid time. Native-clock states are offline diagnostics,
not information exposed to a trading policy, and do not prove clock alignment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from work import fv_mm_inputs as canonical

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_bn_hl_transfer/inputs"
MARKETS = canonical.MARKETS
KST = canonical.KST
START, END = canonical.START, canonical.END
STEP_US = 100_000
VERSION = 1
SMALL_FLOATS = ("imbalance", "spread_bp", "weighted_offset_bp", "ofi100_norm",
                "ofi500_norm", "ofi1000_norm", "age_sec", "recv_age_sec", "truth_age_sec")


def ofi_features(grid, rr, bb, aa, qb, qa, index):
    """Cont event increments over trailing windows; only received events enter."""
    increments = np.zeros(len(rr), dtype=float)
    increments[1:] = ((bb[1:] >= bb[:-1]) * qb[1:] - (bb[1:] <= bb[:-1]) * qb[:-1]
                      - (aa[1:] <= aa[:-1]) * qa[1:] + (aa[1:] >= aa[:-1]) * qa[:-1])
    cumulative = np.cumsum(increments)
    safe = np.maximum(index, 0)
    depth = qb[safe] + qa[safe]
    result = {}
    for millis in (100, 500, 1000):
        left = np.searchsorted(rr, grid - millis * 1000, side="right") - 1
        before = np.where(left >= 0, cumulative[np.maximum(left, 0)], 0.)
        normalized = (cumulative[safe] - before) / depth
        normalized[index < 0] = np.nan
        result[f"ofi{millis}_norm"] = normalized
    return result


def extract_hour(market, hour):
    event, received, seq, bid, ask, bq, aq = canonical.read_bbo_hour(market, hour)
    if not len(event):
        raise ValueError(f"Empty BBO hour {market}/{hour}")
    start = int(hour.timestamp() * 1_000_000)
    grid = start + np.arange(36_000, dtype=np.int64) * STEP_US
    is_hl = market.startswith("hyperliquid")
    order = canonical._advancing_order(received, seq, allow_equal=is_hl)
    rr, ee = received[order], event[order]
    bb, aa, qb, qa = [v[order] for v in (bid, ask, bq, aq)]
    ix, b, a, q_b, q_a, age = canonical._sample(grid, rr, ee, bb, aa, qb, qa)
    safe = np.maximum(ix, 0)
    mid, spread = (b + a) / 2, a - b
    imbalance = (q_b - q_a) / (q_b + q_a)
    result = dict(grid_us=grid, bid=b, ask=a, bq=q_b, aq=q_a, mid=mid,
                  imbalance=imbalance, spread_bp=10000 * spread / mid,
                  weighted_offset_bp=5000 * spread * imbalance / mid,
                  age_sec=age,
                  recv_age_sec=np.where(ix >= 0, (grid-rr[safe]) / 1_000_000, np.nan),
                  recv_event_us=np.where(ix >= 0, ee[safe], -1),
                  recv_stamp_us=np.where(ix >= 0, rr[safe], -1))
    result.update(ofi_features(grid, rr, bb, aa, qb, qa, ix))
    truth_order = (np.lexsort((received, event)) if is_hl
                   else canonical._advancing_order(event, seq))
    te = event[truth_order]
    tb, ta, tqb, tqa = [v[truth_order] for v in (bid, ask, bq, aq)]
    ti, xb, xa, _, _, xa_age = canonical._sample(grid, te, te, tb, ta, tqb, tqa)
    result.update(truth_bid=xb, truth_ask=xa, truth_mid=(xb+xa)/2,
                  truth_age_sec=xa_age,
                  truth_event_us=np.where(ti >= 0, te[np.maximum(ti, 0)], -1))
    for name in SMALL_FLOATS:
        result[name] = result[name].astype(np.float32)
    return result


def validate(data):
    grid = data["grid_us"]
    if len(grid) != 864_000 or not np.all(np.diff(grid) == STEP_US):
        raise AssertionError("100ms grid is not contiguous")
    present = data["recv_stamp_us"] >= 0
    if np.any(data["recv_stamp_us"][present] > grid[present]):
        raise AssertionError("Future receive information entered features")
    native = data["truth_event_us"] >= 0
    if np.any(data["truth_event_us"][native] > grid[native]):
        raise AssertionError("Future native state entered truth")
    if np.any(data["bid"][present] >= data["ask"][present]):
        raise AssertionError("Crossed receive book")
    if not all(len(v) == len(grid) for v in data.values()):
        raise AssertionError("Field lengths differ")


def canonical_parity(data, market, day):
    path = ROOT / "out_fv_mm/inputs" / market / f"{day:%Y-%m-%d}.npz"
    mapping = {"truth_bid": "truth_bid_0", "truth_ask": "truth_ask_0",
               "truth_mid": "truth_mid_0", "truth_age_sec": "truth_age_0",
               "truth_event_us": "truth_event_us_0", "ofi1000_norm": "ofi_one_sec_norm"}
    errors = {}
    with np.load(path, allow_pickle=False) as baseline:
        for key in data:
            old_key = mapping.get(key, key)
            if old_key not in baseline.files:
                continue
            old, new = baseline[old_key], data[key][::10]
            if np.issubdtype(new.dtype, np.integer):
                if not np.array_equal(old, new):
                    raise AssertionError(f"Canonical integer parity failed: {market}/{day}/{key}")
                errors[key] = 0.
            else:
                if not np.allclose(old, new, equal_nan=True, rtol=7e-7, atol=1e-6):
                    raise AssertionError(f"Canonical float parity failed: {market}/{day}/{key}")
                difference = np.abs(old.astype(float) - new.astype(float))
                errors[key] = float(np.nanmax(difference, initial=0))
    return errors


def extract_day(market, day, force=False):
    folder = OUT / market
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{day:%Y-%m-%d}.npz"
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata_json"]))["version"] == VERSION:
                print("SKIP", market, str(day.date()), flush=True)
                return
    start = time.monotonic()
    hours = [extract_hour(market, day + timedelta(hours=h)) for h in range(24)]
    data = {key: np.concatenate([hour[key] for hour in hours]) for key in hours[0]}
    del hours
    validate(data)
    parity = canonical_parity(data, market, day)
    meta = dict(version=VERSION, market=market, day=str(day.date()), rows=len(data["grid_us"]),
                step_us=STEP_US, feature_clock="received_at",
                truth_clock="Binance exchange_event_time / HL event_time (offline diagnostic)",
                synchronized_clocks_proven=False,
                source_sequence_policy="canonical fv_mm_inputs._advancing_order; HL equal event allowed",
                normalized_ofi_windows_ms=[100, 500, 1000], normalized_ofi_denominator="current bid+ask quantity",
                missing_receive_rows=int(np.sum(~np.isfinite(data["mid"]))),
                canonical_1s_parity_errors=parity,
                float32_fields=list(SMALL_FLOATS))
    data["metadata_json"] = np.array(json.dumps(meta))
    temporary = output.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(output)
    meta.update(bytes=output.stat().st_size, runtime_seconds=time.monotonic()-start)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"WROTE {market} {day:%Y-%m-%d} rows={len(data['grid_us']):,} "
          f"MB={output.stat().st_size/1e6:.1f} seconds={meta['runtime_seconds']:.1f}", flush=True)


def selftest():
    canonical.selftest()
    rr = np.array([0, 100_000, 200_000, 300_000], dtype=np.int64)
    bb, aa = np.array([99., 99., 100., 100.]), np.array([101., 101., 101., 102.])
    qb, qa = np.array([2., 3., 4., 4.]), np.array([3., 3., 2., 5.])
    grid = np.array([50_000, 100_000, 200_000, 300_000], dtype=np.int64)
    ix = np.searchsorted(rr, grid, side="right") - 1
    features = ofi_features(grid, rr, bb, aa, qb, qa, ix)
    # t100 increment +1; t200 increment +4+1; t300 ask-rise contributes +2.
    np.testing.assert_allclose(features['ofi100_norm'], [0, 1/6, 5/6, 2/9])
    np.testing.assert_allclose(features['ofi500_norm'], [0, 1/6, 6/6, 8/9])
    # Appending an unseen event must not change the prefix's features.
    new = ofi_features(grid, np.r_[rr, 400_000], np.r_[bb, 1000.], np.r_[aa, 1001.],
                       np.r_[qb, 999.], np.r_[qa, 999.], ix)
    for key in features:
        np.testing.assert_equal(features[key], new[key])
    print("PASS OFI window boundary and future-event prefix invariance", flush=True)


def manifest():
    metadata = []
    for market in MARKETS:
        for path in sorted((OUT / market).glob("2026-*.json")):
            metadata.append(json.loads(path.read_text()))
    counts = {market: sum(m['market'] == market for m in metadata) for market in MARKETS}
    report = dict(version=VERSION, day_files=len(metadata), rows=sum(m['rows'] for m in metadata),
                  bytes=sum(m['bytes'] for m in metadata), markets=list(MARKETS),
                  expected_day_files=72, complete_grid=all(n == 18 for n in counts.values()),
                  files_by_market=counts,
                  date_start=str(START.date()), date_end=str((END-timedelta(days=1)).date()),
                  source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  canonical_source_sha256=hashlib.sha256(Path(canonical.__file__).read_bytes()).hexdigest(),
                  feature_clock="received_at", native_clock_synchronization_proven=False,
                  row_count_is_independent_sample_count=False,
                  canonical_1s_parity_all_files=True, files=metadata)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print("MANIFEST", report['day_files'], report['rows'], report['bytes'], flush=True)


def verify_samples():
    """Direct raw-event as-of/window checks at 16 non-integer-second times."""
    records = []
    times = ('00:10:05.300000', '09:00:00.100000',
             '15:30:45.700000', '23:59:59.900000')
    day = START
    for market in MARKETS:
        with np.load(OUT / market / f'{day:%Y-%m-%d}.npz', allow_pickle=False) as cache:
            for clock in times:
                when = datetime.fromisoformat(f'{day:%Y-%m-%d}T{clock}').replace(tzinfo=KST)
                query = int(when.timestamp() * 1_000_000)
                hour = when.replace(minute=0, second=0, microsecond=0)
                event, received, seq, bid, ask, bq, aq = canonical.read_bbo_hour(market, hour)
                # The following derives state/window answers from raw arrays
                # without invoking the extractor's as-of/OFI functions.
                sort = np.lexsort((seq, received))
                ss = seq[sort]
                keep = np.ones(len(ss), bool)
                keep[1:] = (ss[1:] >= np.maximum.accumulate(ss[:-1]) if market.startswith('hyperliquid')
                            else ss[1:] > np.maximum.accumulate(ss[:-1]))
                ri = sort[keep]
                delivered = np.flatnonzero(received[ri] <= query)
                if not len(delivered):
                    raise AssertionError('Verification point has no received event')
                last = int(delivered[-1])
                raw = int(ri[last])
                index = int((query - int(day.timestamp()*1_000_000)) // STEP_US)
                expected = dict(bid=bid[raw], ask=ask[raw], bq=bq[raw], aq=aq[raw],
                                mid=(bid[raw]+ask[raw])/2,
                                recv_event_us=int(event[raw]), recv_stamp_us=int(received[raw]),
                                age_sec=(query-event[raw])/1e6,
                                recv_age_sec=(query-received[raw])/1e6)
                rb, ra, rqb, rqa = [v[ri] for v in (bid, ask, bq, aq)]
                increments = np.zeros(len(ri))
                # Cont event definition, expressed as separate bid/ask terms.
                bid_term = np.where(rb[1:] > rb[:-1], rqb[1:],
                                    np.where(rb[1:] < rb[:-1], -rqb[:-1], rqb[1:]-rqb[:-1]))
                ask_term = np.where(ra[1:] < ra[:-1], -rqa[1:],
                                    np.where(ra[1:] > ra[:-1], rqa[:-1], rqa[:-1]-rqa[1:]))
                increments[1:] = bid_term + ask_term
                for ms in (100, 500, 1000):
                    eligible = (received[ri] > query-ms*1000) & (received[ri] <= query)
                    expected[f'ofi{ms}_norm'] = increments[eligible].sum() / (bq[raw]+aq[raw])
                ti = (np.lexsort((received, event)) if market.startswith('hyperliquid')
                      else np.lexsort((seq, event)))
                if not market.startswith('hyperliquid'):
                    ts = seq[ti]
                    tk = np.ones(len(ts), bool)
                    tk[1:] = ts[1:] > np.maximum.accumulate(ts[:-1])
                    ti = ti[tk]
                truth_candidates = ti[event[ti] <= query]
                tr = int(truth_candidates[-1])
                expected.update(truth_bid=bid[tr], truth_ask=ask[tr],
                                truth_mid=(bid[tr]+ask[tr])/2,
                                truth_event_us=int(event[tr]), truth_age_sec=(query-event[tr])/1e6)
                errors = {}
                for field, value in expected.items():
                    actual = cache[field][index]
                    if field.endswith('_us'):
                        equal = int(actual) == value
                    else:
                        equal = bool(np.isclose(actual, value, rtol=8e-7, atol=2e-6))
                    if not equal:
                        raise AssertionError(f'Raw sample mismatch {market}/{clock}/{field}: {actual} != {value}')
                    errors[field] = float(abs(actual-value))
                records.append(dict(market=market, time_kst=when.isoformat(), fields_checked=len(expected), errors=errors))
    result = dict(status='pass', raw_queries=len(records),
                  raw_values_checked=sum(r['fields_checked'] for r in records), samples=records,
                  float_tolerance=dict(rtol=8e-7, atol=2e-6),
                  integer_tolerance=0,
                  definitions='Independent raw as-of and direct event-window OFI sums; no extractor helper used for expected values')
    (OUT / 'validation_samples.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS RAW SAMPLES', result['raw_queries'], result['raw_values_checked'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market', action='append', choices=MARKETS)
    parser.add_argument('--day', action='append')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--verify-samples', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.verify_samples:
        verify_samples()
        return
    if not args.manifest_only:
        days = ([datetime.fromisoformat(d).replace(tzinfo=KST) for d in args.day] if args.day else
                [START+timedelta(days=i) for i in range((END-START).days)])
        for day in days:
            for market in args.market or MARKETS:
                extract_day(market, day, args.force)
    manifest()


if __name__ == '__main__':
    main()
