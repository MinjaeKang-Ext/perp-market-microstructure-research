"""Daily expanding out-of-sample evaluation of causal BBO fair-value forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work/orderbook_fair_value/grid"
OUT = ROOT / "work/orderbook_fair_value"
MARKETS = ("binance_SKHYNIXUSDT", "hyperliquid_SKHX")
HORIZONS = (1, 5, 30, 60)
AGE_LIMITS = (1, 5)
TARGETS = ("exchange", "receive")


def load_grid(market):
    files = sorted((DATA / market).glob("*.parquet"))
    if len(files) != 18:
        raise ValueError(f"Expected 18 days, found {len(files)} for {market}")
    frame = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
    grid = frame["grid_us"].to_numpy()
    if not np.all(np.diff(grid) == 1_000_000):
        raise ValueError(f"Non-contiguous 1-second grid: {market}")
    if len(frame) != 18 * 86_400:
        raise ValueError(f"Unexpected grid rows: {market}")
    return frame


def day_bootstrap_interval(values, counts, seed=20261105):
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(5000, len(values)))
    resampled = values[choices].sum(axis=1) / counts[choices].sum(axis=1)
    return np.quantile(resampled, [0.025, 0.975]).tolist()


def lagged(values, seconds):
    values = np.asarray(values, dtype=float)
    if seconds == 0:
        return values
    result = np.full(len(values), np.nan)
    result[seconds:] = values[:-seconds]
    return result


def features(frame, other, cross_lag_seconds=0):
    mid = frame["recv_mid"].to_numpy(dtype=float)
    prev = np.roll(mid, 1)
    prev[0] = np.nan
    ret = 10_000 * (mid / prev - 1)
    cross_mid = other["recv_mid"].to_numpy(dtype=float)
    cross_prev = np.roll(cross_mid, 1)
    cross_prev[0] = np.nan
    cross_ret = 10_000 * (cross_mid / cross_prev - 1)
    own = np.column_stack((
        np.ones(len(frame)),
        frame["weighted_offset_bp"].to_numpy(dtype=float),
        frame["imbalance"].to_numpy(dtype=float),
        frame["spread_bp"].to_numpy(dtype=float),
        frame["ofi_one_sec_norm"].to_numpy(dtype=float),
        ret,
    ))
    cross = np.column_stack((
        own,
        lagged(other["weighted_offset_bp"].to_numpy(dtype=float), cross_lag_seconds),
        lagged(other["imbalance"].to_numpy(dtype=float), cross_lag_seconds),
        lagged(other["ofi_one_sec_norm"].to_numpy(dtype=float), cross_lag_seconds),
        lagged(cross_ret, cross_lag_seconds),
    ))
    return own, cross


def evaluate_one(market, frame, other, horizon, age_limit, target, feature_set, cross_lag_seconds=0):
    n = len(frame)
    own_x, cross_x = features(frame, other, cross_lag_seconds)
    x = own_x if feature_set == "own" else cross_x
    base_mid = frame["recv_mid"].to_numpy(dtype=float)
    base_age = frame["recv_age_sec"].to_numpy(dtype=float)
    previous_age = np.roll(base_age, 1)
    previous_age[0] = np.nan
    target_column = "exchange_mid" if target == "exchange" else "recv_mid"
    target_age_column = "exchange_age_sec" if target == "exchange" else "recv_age_sec"
    future_mid = np.full(n, np.nan)
    future_age = np.full(n, np.nan)
    future_mid[:-horizon] = frame[target_column].to_numpy(dtype=float)[horizon:]
    future_age[:-horizon] = frame[target_age_column].to_numpy(dtype=float)[horizon:]
    y = 10_000 * (future_mid / base_mid - 1)
    # The last horizon seconds of each KST day are purged before the next day fit.
    day = np.arange(n) // 86_400
    same_day_target = np.zeros(n, dtype=bool)
    same_day_target[:-horizon] = day[:-horizon] == day[horizon:]
    valid = (
        same_day_target & np.isfinite(y) & np.isfinite(x).all(axis=1)
        & (base_age >= 0) & (base_age <= age_limit)
        & (previous_age >= 0) & (previous_age <= age_limit)
        & (future_age >= 0) & (future_age <= age_limit)
    )
    if feature_set == "cross":
        other_age = lagged(other["recv_age_sec"].to_numpy(dtype=float), cross_lag_seconds)
        other_previous_age = lagged(other["recv_age_sec"].to_numpy(dtype=float), cross_lag_seconds + 1)
        valid &= (other_age >= 0) & (other_age <= age_limit)
        valid &= (other_previous_age >= 0) & (other_previous_age <= age_limit)

    daily = []
    xx = np.zeros((x.shape[1], x.shape[1]))
    xy = np.zeros(x.shape[1])
    for d in range(18):
        selection = valid & (day == d)
        xd, yd = x[selection], y[selection]
        if len(yd) == 0:
            daily.append({"day_index": d, "n": 0})
            continue
        if d > 0 and np.any(xx):
            coefficient = np.linalg.lstsq(xx, xy, rcond=None)[0]
            predictions = {
                "current_mid": np.zeros(len(yd)),
                "weighted_mid": xd[:, 1],
                "calibrated": xd @ coefficient,
            }
            record = {"day_index": d, "n": len(yd), "nonzero": int(np.sum(yd != 0))}
            # A clock-time split, not a claim that KRX was open on every date.
            kst_seconds = np.flatnonzero(selection) % 86_400
            regular_clock = (kst_seconds >= 9 * 3600) & (kst_seconds < 15 * 3600 + 20 * 60)
            record["clock_sessions"] = {}
            for label, mask in (("09:00-15:20_KST", regular_clock), ("other_hours", ~regular_clock)):
                record["clock_sessions"][label] = {
                    "n": int(np.sum(mask)),
                    "nonzero": int(np.sum((yd != 0) & mask)),
                    "models": {
                        name: {
                            "sum_abs_error_bp": float(np.sum(np.abs((yd - pred)[mask]))),
                            "direction_hits": int(np.sum((np.sign(pred[mask]) == np.sign(yd[mask])) & (yd[mask] != 0))),
                        }
                        for name, pred in predictions.items()
                    },
                }
            for name, pred in predictions.items():
                err = np.abs(yd - pred)
                record[name] = {
                    "sum_abs_error_bp": float(np.sum(err)),
                    "sum_sq_error_bp2": float(np.sum((yd - pred)**2)),
                    "direction_hits": int(np.sum((np.sign(pred) == np.sign(yd)) & (yd != 0))),
                }
            daily.append(record)
        else:
            daily.append({"day_index": d, "n": 0, "train_n": len(yd)})
        xx += xd.T @ xd
        xy += xd.T @ yd

    usable = [row for row in daily if "calibrated" in row]
    if not usable:
        raise ValueError(f"No OOS evaluation: {market}/{horizon}/{age_limit}/{target}/{feature_set}")
    total_n = sum(row["n"] for row in usable)
    nonzero = sum(row["nonzero"] for row in usable)
    summary = {
        "market": market, "horizon_seconds": horizon, "max_age_seconds": age_limit,
        "target_clock": target, "feature_set": feature_set,
        "cross_signal_lag_seconds": cross_lag_seconds,
        "n_oos": total_n, "nonzero_moves": nonzero,
        "days_oos": len(usable), "share_of_possible_oos": total_n / (17 * (86_400 - horizon)),
    }
    for name in ("current_mid", "weighted_mid", "calibrated"):
        error_sums = np.array([row[name]["sum_abs_error_bp"] for row in usable])
        squared_error_sums = np.array([row[name]["sum_sq_error_bp2"] for row in usable])
        counts = np.array([row["n"] for row in usable])
        hits = sum(row[name]["direction_hits"] for row in usable)
        summary[name] = {
            "mae_bp": float(error_sums.sum() / counts.sum()),
            "rmse_bp": float(np.sqrt(squared_error_sums.sum() / counts.sum())),
            "direction_hit_rate_nonzero": hits / nonzero if nonzero else None,
        }
    for base_name in ("current_mid", "weighted_mid"):
        delta_sums = np.array([row[base_name]["sum_abs_error_bp"] - row["calibrated"]["sum_abs_error_bp"] for row in usable])
        counts = np.array([row["n"] for row in usable])
        delta_hits = sum(row["calibrated"]["direction_hits"] - row[base_name]["direction_hits"] for row in usable)
        daily_hit_delta = np.array([row["calibrated"]["direction_hits"] - row[base_name]["direction_hits"] for row in usable])
        daily_nonzero = np.array([row["nonzero"] for row in usable])
        summary[f"calibrated_vs_{base_name}"] = {
            "mae_improvement_bp": float(delta_sums.sum() / counts.sum()),
            "mae_improvement_day_bootstrap_95": day_bootstrap_interval(delta_sums, counts),
            "direction_hit_rate_improvement": delta_hits / nonzero if nonzero else None,
            "direction_hit_rate_improvement_day_bootstrap_95": day_bootstrap_interval(daily_hit_delta, daily_nonzero) if nonzero else None,
            "days_with_mae_improvement": int(np.sum(delta_sums > 0)),
        }
    summary["clock_sessions"] = {}
    for label in ("09:00-15:20_KST", "other_hours"):
        blocks = [row["clock_sessions"][label] for row in usable]
        session_n = sum(block["n"] for block in blocks)
        session_nonzero = sum(block["nonzero"] for block in blocks)
        if not session_n:
            continue
        summary["clock_sessions"][label] = {
            "n": session_n,
            "mae_improvement_bp": sum(block["models"]["current_mid"]["sum_abs_error_bp"] - block["models"]["calibrated"]["sum_abs_error_bp"] for block in blocks) / session_n,
            "direction_hit_rate_improvement_vs_weighted_mid": sum(block["models"]["calibrated"]["direction_hits"] - block["models"]["weighted_mid"]["direction_hits"] for block in blocks) / session_nonzero if session_nonzero else None,
        }
    return summary, daily


def main():
    frames = {market: load_grid(market) for market in MARKETS}
    if not np.array_equal(frames[MARKETS[0]]["grid_us"].to_numpy(), frames[MARKETS[1]]["grid_us"].to_numpy()):
        raise ValueError("Two market grids are not aligned")
    summaries = []
    daily_records = []
    for market, other_market in ((MARKETS[0], MARKETS[1]), (MARKETS[1], MARKETS[0])):
        for h in HORIZONS:
            for age in AGE_LIMITS:
                for target in TARGETS:
                    for feature_set in ("own", "cross"):
                        summary, daily = evaluate_one(market, frames[market], frames[other_market], h, age, target, feature_set)
                        summaries.append(summary)
                        daily_records.append({"key": [market, h, age, target, feature_set], "daily": daily})
                        print(market, h, age, target, feature_set,
                              "n", summary["n_oos"],
                              "mid_mae", round(summary["current_mid"]["mae_bp"], 4),
                              "weighted_mae", round(summary["weighted_mid"]["mae_bp"], 4),
                              "model_mae", round(summary["calibrated"]["mae_bp"], 4), flush=True)
    # Delaying the other venue's signals diagnoses apparent lead from feed timing.
    for h in HORIZONS:
        for age in AGE_LIMITS:
            for target in TARGETS:
                for lag in (1, 5):
                    summary, daily = evaluate_one(MARKETS[1], frames[MARKETS[1]], frames[MARKETS[0]], h, age, target, "cross", lag)
                    summaries.append(summary)
                    daily_records.append({"key": [MARKETS[1], h, age, target, "cross", lag], "daily": daily})
                    print(MARKETS[1], h, age, target, "cross", "lag", lag,
                          "n", summary["n_oos"],
                          "mid_mae", round(summary["current_mid"]["mae_bp"], 4),
                          "model_mae", round(summary["calibrated"]["mae_bp"], 4), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (OUT / "daily.json").write_text(json.dumps(daily_records, indent=2) + "\n")


if __name__ == "__main__":
    main()
