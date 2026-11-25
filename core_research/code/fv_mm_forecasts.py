"""Causal four-perp FV forecasts with past-day expanding OLS fits.

Run with a single BLAS thread for predictable resource use:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 work/fv_mm_forecasts.py

All forecasts are bp changes relative to the observed midpoint at decision time.
Decision masks never inspect future labels; scoring/training masks may do so.
The first KST day trains only. Each day's last H seconds are purged separately.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "out_fv_mm/inputs"
OUT = ROOT / "out_fv_mm/forecasts"
MARKETS = ("binance_SKHYNIXUSDT", "hyperliquid_SKHX",
           "binance_SKHYUSDT", "hyperliquid_SKHY")
OTHER = dict(zip(MARKETS, (MARKETS[1], MARKETS[0], MARKETS[3], MARKETS[2])))
ARRIVAL_MS = {market: (100 if market.startswith("binance_") else 440) for market in MARKETS}
HORIZONS = (1, 5, 30, 60)
AGES = (1, 5)
MODELS = ("own", "cross", "crosslag1")
TARGETS = ("exchange", "receive")
DAYS = tuple(str(date(2026, 10, 15) + timedelta(days=i)) for i in range(18))
FIELDS = ("grid_us", "mid", "age_sec", "recv_age_sec", "weighted_offset_bp",
          "imbalance", "spread_bp", "ofi_one_sec_norm", "truth_mid_0", "truth_age_0")
OWN_FEATURES = ("weighted_offset_bp", "imbalance", "spread_bp",
                "ofi_one_sec_norm", "mid_return_bp_1s")
CROSS_FEATURES = OWN_FEATURES + ("other_weighted_offset_bp", "other_imbalance",
                               "other_ofi_one_sec_norm", "other_mid_return_bp_1s")
VERSION = 1


def lag(values, seconds=1):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    if seconds == 0:
        return values.copy()
    result[seconds:] = values[:-seconds]
    return result


def prepend_tail(current, previous):
    """Keep the two prior seconds needed for a lagged other-venue return."""
    result = {}
    for name, values in current.items():
        if name == "grid_us":
            continue
        prefix = previous[name][-2:] if previous is not None else np.full(2, np.nan)
        result[name] = np.concatenate((prefix, values))
    return result


def features_for_day(own, other, previous_own=None, previous_other=None):
    a, b = prepend_tail(own, previous_own), prepend_tail(other, previous_other)
    ret_a = 10_000 * (a["mid"] / lag(a["mid"]) - 1)
    ret_b = 10_000 * (b["mid"] / lag(b["mid"]) - 1)
    own_x = np.column_stack((a["weighted_offset_bp"], a["imbalance"],
                             a["spread_bp"], a["ofi_one_sec_norm"], ret_a))[2:]
    extra = np.column_stack((b["weighted_offset_bp"], b["imbalance"],
                             b["ofi_one_sec_norm"], ret_b))
    xs = {"own": own_x, "cross": np.column_stack((own_x, extra[2:])),
          "crosslag1": np.column_stack((own_x, extra[1:-1]))}
    own_ages = np.column_stack((a["age_sec"], lag(a["age_sec"])))[2:]
    other_ages = np.column_stack((b["age_sec"], lag(b["age_sec"])))
    masks = {}
    for model in MODELS:
        relevant = own_ages
        if model == "cross":
            relevant = np.column_stack((relevant, other_ages[2:]))
        elif model == "crosslag1":
            relevant = np.column_stack((relevant, other_ages[1:-1]))
        for age in AGES:
            masks[model, age] = (
                np.isfinite(xs[model]).all(axis=1)
                & np.isfinite(relevant).all(axis=1)
                & (relevant >= 0).all(axis=1) & (relevant <= age).all(axis=1)
            )
    return xs, masks


def label_for_day(frame, horizon, target, age, arrival_ms=0):
    price_key, age_key = ((f"truth_mid_{arrival_ms}", f"truth_age_{arrival_ms}") if target == "exchange"
                          else ("mid", "age_sec"))
    future_mid, future_age = np.full(len(frame["mid"]), np.nan), np.full(len(frame["mid"]), np.nan)
    future_mid[:-horizon] = frame[price_key][horizon:]
    future_age[:-horizon] = frame[age_key][horizon:]
    y = 10_000 * (future_mid / frame["mid"] - 1)
    valid = np.isfinite(y) & (future_age >= 0) & (future_age <= age)
    return y, valid


@dataclass
class Moments:
    n: int
    sx: np.ndarray
    sy: float
    sxx: np.ndarray
    sxy: np.ndarray

    @classmethod
    def empty(cls, dimension):
        return cls(0, np.zeros(dimension), 0., np.zeros((dimension, dimension)), np.zeros(dimension))

    def update(self, x, y):
        if len(y):
            self.n += len(y)
            self.sx += x.sum(axis=0)
            self.sy += float(y.sum())
            self.sxx += x.T @ x
            self.sxy += x.T @ y

    def solve(self):
        """Centered, scaled moment OLS; rank threshold is numerical, not tuned.

        Scaling prevents basis units from deciding which collinear direction is
        dropped. The relative 1e-10 eigenvalue threshold is documented in output.
        """
        if self.n <= len(self.sx) + 1:
            return None
        mean_x, mean_y = self.sx / self.n, self.sy / self.n
        cov = self.sxx / self.n - np.outer(mean_x, mean_x)
        cov = (cov + cov.T) / 2
        cross = self.sxy / self.n - mean_x * mean_y
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        sd = np.where(sd > 1e-15, sd, 1.)
        correlation = cov / np.outer(sd, sd)
        evals, evecs = np.linalg.eigh(correlation)
        largest = float(np.max(evals))
        keep = evals > max(largest * 1e-10, 1e-14)
        if np.any(keep):
            beta_scaled = evecs[:, keep] @ ((evecs[:, keep].T @ (cross / sd)) / evals[keep])
            beta = beta_scaled / sd
            condition = float(evals[keep].max() / evals[keep].min())
        else:
            beta, condition = np.zeros(len(sd)), np.nan
        intercept = float(mean_y - mean_x @ beta)
        return intercept, beta, int(keep.sum()), condition


def score(y, pred, weighted):
    actual_moves = y != 0
    record = {"n": len(y), "nonzero": int(actual_moves.sum())}
    for name, p in (("mid", np.zeros(len(y))), ("weighted", weighted), ("model", pred)):
        error = y - p
        record[name + "_abs_sum"] = float(np.abs(error).sum())
        record[name + "_sq_sum"] = float((error * error).sum())
        record[name + "_hits"] = int(((np.sign(p) == np.sign(y)) & actual_moves).sum())
    return record


def bootstrap_interval(values, denominator, seed=20261107):
    values, denominator = np.asarray(values), np.asarray(denominator)
    if not len(values) or denominator.sum() <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(values), (5000, len(values)))
    totals = denominator[pick].sum(axis=1)
    good = totals > 0
    samples = values[pick].sum(axis=1)[good] / totals[good]
    return tuple(np.quantile(samples, (.025, .975)))


def aggregate(daily):
    frame = pd.DataFrame(daily)
    if frame.empty:
        return pd.DataFrame()
    keys = ["market", "target", "model", "horizon", "max_age"]
    records = []
    for key, group in frame[frame.n > 0].groupby(keys, sort=True):
        record = dict(zip(keys, key))
        n, nonzero = int(group.n.sum()), int(group.nonzero.sum())
        record.update(n_oos=n, nonzero_moves=nonzero, days_oos=len(group))
        for name in ("mid", "weighted", "model"):
            record[name + "_mae_bp"] = group[name + "_abs_sum"].sum() / n
            record[name + "_rmse_bp"] = np.sqrt(group[name + "_sq_sum"].sum() / n)
            record[name + "_direction_hit_rate"] = group[name + "_hits"].sum() / nonzero if nonzero else np.nan
        for baseline in ("mid", "weighted"):
            difference = group[baseline + "_abs_sum"] - group.model_abs_sum
            record["mae_improvement_vs_" + baseline + "_bp"] = difference.sum() / n
            lo, hi = bootstrap_interval(difference.to_numpy(), group.n.to_numpy())
            record["mae_improvement_vs_" + baseline + "_ci_low"] = lo
            record["mae_improvement_vs_" + baseline + "_ci_high"] = hi
            record["mae_improved_days_vs_" + baseline] = int((difference > 0).sum())
        difference = group.model_hits - group.weighted_hits
        record["direction_improvement_vs_weighted"] = difference.sum() / nonzero if nonzero else np.nan
        lo, hi = bootstrap_interval(difference.to_numpy(), group.nonzero.to_numpy())
        record["direction_improvement_vs_weighted_ci_low"] = lo
        record["direction_improvement_vs_weighted_ci_high"] = hi
        records.append(record)
    return pd.DataFrame(records)


def load_day(folder, market, day):
    path = folder / market / (day + ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    with np.load(path, allow_pickle=False) as data:
        frame = {name: data[name] for name in FIELDS}
        arrival = ARRIVAL_MS[market]
        frame[f"truth_mid_{arrival}"] = data[f"truth_mid_{arrival}"]
        # The generator also supplies truth_age_sec aliases; use one explicit convention.
        age_key = f"truth_age_{arrival}"
        frame[age_key] = data[age_key] if age_key in data else data[f"truth_age_sec_{arrival}"]
    if len(frame["grid_us"]) != 86400 or not np.all(np.diff(frame["grid_us"]) == 1_000_000):
        raise ValueError(f"Invalid 1-second day grid: {path}")
    return frame


def run(markets=MARKETS, input_folder=INPUT, output_folder=OUT, days=DAYS, wait_seconds=0):
    output_folder.mkdir(parents=True, exist_ok=True)
    states, previous, daily, coefficients, fingerprints = {}, {}, [], [], []
    needed = tuple(dict.fromkeys(tuple(markets) + tuple(OTHER[m] for m in markets)))
    for day_index, day in enumerate(days):
        deadline = time.monotonic() + wait_seconds
        while True:
            missing = [str(input_folder / m / (day + ".npz")) for m in needed
                       if not (input_folder / m / (day + ".npz")).exists()]
            if not missing:
                break
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Waiting limit reached; missing inputs: {missing}")
            print(f"WAIT {day} missing_markets={len(missing)}", flush=True)
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        frames = {m: load_day(input_folder, m, day) for m in needed}
        for market, frame in frames.items():
            path = input_folder / market / (day + ".npz")
            stat = path.stat()
            fingerprints.append({"market": market, "day": day, "path": str(path),
                                 "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        for market in markets:
            frame, other = frames[market], frames[OTHER[market]]
            if not np.array_equal(frame["grid_us"], other["grid_us"]):
                raise ValueError("Paired venue grids differ")
            xs, masks = features_for_day(frame, other, previous.get(market), previous.get(OTHER[market]))
            payload = {"grid_us": frame["grid_us"], "mid": frame["mid"]}
            for model in MODELS:
                for age in AGES:
                    payload[f"decision_valid_{model}_a{age}"] = masks[model, age]
            for target in TARGETS:
                for horizon in HORIZONS:
                    for age in AGES:
                        y, label_valid = label_for_day(frame, horizon, target, age, ARRIVAL_MS[market])
                        for model in MODELS:
                            key = (market, target, horizon, age, model)
                            x = xs[model]
                            state = states.setdefault(key, Moments.empty(x.shape[1]))
                            fitted = state.solve()
                            prediction = np.full(len(y), np.nan)
                            record = dict(market=market, day=day, day_index=day_index,
                                          target=target, model=model, horizon=horizon,
                                          max_age=age, train_n=state.n)
                            if fitted is not None:
                                intercept, beta, rank, condition = fitted
                                use = masks[model, age]
                                prediction[use] = intercept + x[use] @ beta
                                names = OWN_FEATURES if model == "own" else CROSS_FEATURES
                                for name, value in zip(("intercept",) + names, (intercept,) + tuple(beta)):
                                    coefficients.append(dict(record, feature=name, coefficient=float(value),
                                                             rank=rank, condition=condition))
                                eval_use = use & label_valid & np.isfinite(prediction)
                                record.update(score(y[eval_use], prediction[eval_use],
                                                    frame["weighted_offset_bp"][eval_use]))
                            else:
                                record.update(score(np.zeros(0), np.zeros(0), np.zeros(0)))
                            daily.append(record)
                            payload[f"pred_{target}_{model}_h{horizon}_a{age}"] = prediction.astype("float32")
                            # Updating after predicting guarantees every coefficient used
                            # today was fitted on earlier KST dates only.
                            train_use = masks[model, age] & label_valid
                            state.update(x[train_use], y[train_use])
            folder = output_folder / market
            folder.mkdir(parents=True, exist_ok=True)
            final = folder / (day + ".npz")
            temporary = final.with_suffix(".part.npz")
            np.savez_compressed(temporary, **payload)
            temporary.replace(final)
            print(f"FORECAST {market} {day} train_days={day_index} keys={len(payload)}", flush=True)
        previous = frames
        # These progress files are safe to inspect while later dates run.
        pd.DataFrame(daily).to_csv(output_folder / "daily.csv", index=False)
        pd.DataFrame(coefficients).to_csv(output_folder / "coefficients.csv", index=False)
    summary = aggregate(daily)
    summary.to_csv(output_folder / "summary.csv", index=False)
    metadata = {
        "version": VERSION, "markets": list(markets), "days": list(days),
        "sampling_seconds": 1, "horizons_seconds": HORIZONS, "max_event_age_seconds": AGES,
        "training": "expanding earlier KST days only; each day's final H seconds purged",
        "models": MODELS, "targets": TARGETS, "prediction_units": "bp versus decision-time receive midpoint",
        "arrival_ms": ARRIVAL_MS,
        "target_times": {"exchange": "decision t + venue arrival L + horizon H", "receive": "decision t + horizon H; feed diagnostic only"},
        "feature_names": {"own": OWN_FEATURES, "cross": CROSS_FEATURES, "crosslag1": CROSS_FEATURES},
        "other_venue_map": OTHER, "decision_mask_uses_future_labels": False,
        "ols": "centered standardized sufficient moments; relative eigenvalue cutoff 1e-10, absolute 1e-14",
        "confidence_intervals": "5000 whole-KST-day resamples, seed 20261107; overlapping intraday labels not iid",
        "new_holdout": False, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_fingerprints": fingerprints,
        "input_fingerprint_limit": "Path/size/mtime only, not content hashes; input generator audit is separate",
        "price_policy_or_pnl_test": False,
    }
    (output_folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"DONE summaries={len(summary)} daily_records={len(daily)}", flush=True)
    return summary


class ForecastCausalityTests(unittest.TestCase):
    def fixture(self, n=200):
        rng = np.random.default_rng(7)
        mid = 100 + np.cumsum(rng.normal(0, .01, n))
        return dict(grid_us=np.arange(n) * 1_000_000, mid=mid,
                    age_sec=np.full(n, .1), recv_age_sec=np.full(n, .01),
                    weighted_offset_bp=rng.normal(0, .1, n), imbalance=rng.normal(0, .2, n),
                    spread_bp=np.full(n, .4), ofi_one_sec_norm=rng.normal(0, 1, n),
                    truth_mid_0=mid + rng.normal(0, .001, n), truth_age_0=np.full(n, .05))

    def test_scaled_moments_match_direct_ols_with_collinearity(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(1000, 3))
        x = np.column_stack((a, 1000 * a[:, 0], np.ones(1000)))
        y = 1.5 + a @ np.array([.2, -.5, 1.]) + rng.normal(0, .01, 1000)
        state = Moments.empty(5)
        state.update(x[:300], y[:300])
        state.update(x[300:], y[300:])
        intercept, beta, _, _ = state.solve()
        direct = np.linalg.lstsq(np.column_stack((np.ones(1000), x)), y, rcond=None)[0]
        np.testing.assert_allclose(intercept + x @ beta,
                                   np.column_stack((np.ones(1000), x)) @ direct, atol=1e-9)

    def test_current_day_future_truth_cannot_change_prediction_or_decision(self):
        previous, own, other = self.fixture(), self.fixture(), self.fixture()
        x0, mask0 = features_for_day(own, other, previous, previous)
        state = Moments.empty(5)
        xp, mp = features_for_day(previous, previous)
        yp, vp = label_for_day(previous, 5, "exchange", 1)
        state.update(xp["own"][mp["own", 1] & vp], yp[mp["own", 1] & vp])
        intercept, beta, _, _ = state.solve()
        predicted0 = intercept + x0["own"] @ beta
        own["truth_mid_0"] *= 2
        own["truth_age_0"][:] = 100
        x1, mask1 = features_for_day(own, other, previous, previous)
        np.testing.assert_array_equal(mask0["own", 1], mask1["own", 1])
        np.testing.assert_allclose(predicted0, intercept + x1["own"] @ beta)

    def test_tail_purge_and_crosslag_boundaries(self):
        own, other = self.fixture(), self.fixture()
        _, masks = features_for_day(own, other)
        self.assertFalse(masks["own", 1][0])
        self.assertFalse(masks["crosslag1", 1][1])
        self.assertTrue(masks["crosslag1", 1][2])
        for h in HORIZONS:
            y, valid = label_for_day(own, h, "exchange", 1)
            self.assertFalse(valid[-h:].any())
            self.assertTrue(np.isnan(y[-h:]).all())

    def test_exchange_horizon_starts_after_arrival_receive_diagnostic_does_not(self):
        frame = self.fixture()
        frame["truth_mid_440"] = frame["mid"] + 1
        frame["truth_age_440"] = np.full(len(frame["mid"]), .05)
        exchange, ex_valid = label_for_day(frame, 1, "exchange", 1, 440)
        receive, rec_valid = label_for_day(frame, 1, "receive", 1, 440)
        expected = 10_000 * ((frame["mid"][1:] + 1) / frame["mid"][:-1] - 1)
        np.testing.assert_allclose(exchange[:-1], expected)
        self.assertTrue(np.all(exchange[:-1] > receive[:-1]))
        np.testing.assert_array_equal(ex_valid, rec_valid)

    def test_stale_other_venue_only_invalidates_cross_at_correct_lag(self):
        own, other = self.fixture(), self.fixture()
        other["age_sec"][50] = 10
        _, masks = features_for_day(own, other)
        self.assertTrue(masks["own", 1][50])
        self.assertFalse(masks["cross", 1][50])
        self.assertFalse(masks["cross", 1][51])
        self.assertFalse(masks["crosslag1", 1][51])
        self.assertFalse(masks["crosslag1", 1][52])
        self.assertTrue(masks["crosslag1", 1][50])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=MARKETS, action="append")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="Per-day wait for concurrently generated inputs")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.main(argv=[__file__], exit=False, verbosity=2)
        raise SystemExit(not result.result.wasSuccessful())
    else:
        run(tuple(args.market or MARKETS), args.input, args.output, wait_seconds=args.wait_seconds)


if __name__ == "__main__":
    main()
