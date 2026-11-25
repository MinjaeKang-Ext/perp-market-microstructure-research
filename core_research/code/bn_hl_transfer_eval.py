"""Past-day Binance -> Hyperliquid best-quote transfer diagnostics.

100ms input/price/lag resolution; primary decisions remain at one-second phase
zero. These are forecast diagnostics, not fills or PnL. Native innovations use
an offline native current quote and must not be treated as executable forecasts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import time
import unittest

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'out_bn_hl_transfer/inputs'
OUT = ROOT / 'out_bn_hl_transfer/eval'
BN = ('binance_SKHYNIXUSDT', 'binance_SKHYUSDT')
HL = ('hyperliquid_SKHX', 'hyperliquid_SKHY')
PAIRS = [(b, h) for b in BN for h in HL]
DAYS = tuple(str(date(2026, 10, 15) + timedelta(days=i)) for i in range(18))
DELAYS_MS = (0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 5000, 10000, 30000, 60000)
HORIZONS_MS = (100, 500, 1000, 5000, 30000)
CLOCKS = ('receive', 'native_innovation', 'native_endpoint')
FAMILIES = ('mid', 'micro', 'obi', 'ofi', 'joint', 'mid_ofi')
TAIL = 700
VERSION = 1
OWN_NAMES = ('own_spread_bp', 'own_obi', 'own_weighted_offset_bp',
             'own_ret100', 'own_ret500', 'own_ret1000',
             'own_ofi100', 'own_ofi500', 'own_ofi1000')
SOURCE_NAMES = ('mid_gap', 'mid_ret100', 'mid_ret500', 'mid_ret1000', 'mid_ret5000',
                'micro_gap', 'micro_ret100', 'micro_ret500', 'micro_ret1000', 'micro_ret5000',
                'obi', 'obi_delta100', 'obi_delta500', 'obi_delta1000',
                'ofi100', 'ofi500', 'ofi1000')
FEATURE_NAMES = OWN_NAMES + SOURCE_NAMES
GROUPS = dict(mid=list(range(0, 5)), micro=list(range(5, 10)), obi=list(range(10, 14)),
              ofi=list(range(14, 17)), joint=list(range(17)),
              mid_ofi=list(range(5)) + list(range(14, 17)))
RAW = ('mid', 'bid', 'ask', 'imbalance', 'spread_bp', 'weighted_offset_bp',
       'ofi100_norm', 'ofi500_norm', 'ofi1000_norm', 'age_sec',
       'truth_bid', 'truth_ask', 'truth_age_sec')


def load_day(market, day):
    with np.load(INPUT / market / (day + '.npz'), allow_pickle=False) as f:
        return {k: f[k].astype(np.float64) if k != 'grid_us' else f[k]
                for k in ('grid_us',) + RAW}


def augmented(frame, previous):
    return {k: np.concatenate((previous[k][-TAIL:] if previous is not None
                               else np.full(TAIL, np.nan), frame[k])) for k in RAW}


def causal_basis(source_mid, target_mid, initial=None):
    """EWMA log basis; only as-of prices, fixed 300-second half-life."""
    values = np.log(source_mid) - np.log(target_mid)
    good = np.isfinite(values)
    if not good.any():
        return np.full(len(values), np.nan), initial
    first = int(np.flatnonzero(good)[0])
    # Forward fill invalid input; this is only a state carrier. Its independent
    # source/target freshness checks still gate every prediction.
    take = np.maximum.accumulate(np.where(good, np.arange(len(values)), first))
    filled = values[take]
    alpha = 1 - math.exp(-math.log(2) * .1 / 300.)
    start = float(values[first]) if initial is None else float(initial)
    smooth, _ = lfilter([alpha], [1, -(1-alpha)], filled,
                        zi=np.array([(1-alpha)*start]))
    if initial is None:
        smooth[:first] = np.nan
    return smooth, float(smooth[-1])


def log_return(values, at, back):
    return 10000. * (np.log(values[at]) - np.log(values[at-back]))


def own_features(a, at):
    x = np.column_stack((a['spread_bp'][at], a['imbalance'][at], a['weighted_offset_bp'][at],
                         *[log_return(a['mid'], at, b) for b in (1, 5, 10)],
                         a['ofi100_norm'][at], a['ofi500_norm'][at], a['ofi1000_norm'][at]))
    ages = np.column_stack([a['age_sec'][at-b] for b in (0, 1, 5, 10)])
    return x, ages


def source_features(a, target_mid, basis, at):
    wm = a['mid'] * (1 + a['weighted_offset_bp'] / 10000.)
    mid_gap = 10000. * (np.log(a['mid'][at]) - basis[at] - np.log(target_mid))
    micro_gap = 10000. * (np.log(wm[at]) - basis[at] - np.log(target_mid))
    x = np.column_stack((mid_gap, *[log_return(a['mid'], at, b) for b in (1, 5, 10, 50)],
                         micro_gap, *[log_return(wm, at, b) for b in (1, 5, 10, 50)],
                         a['imbalance'][at],
                         *[a['imbalance'][at] - a['imbalance'][at-b] for b in (1, 5, 10)],
                         a['ofi100_norm'][at], a['ofi500_norm'][at], a['ofi1000_norm'][at]))
    ages = np.column_stack([a['age_sec'][at-b] for b in (0, 1, 5, 10, 50)])
    return x, ages


def labels(frame, phase=0):
    base = np.arange(phase, len(frame['mid']), 10)
    columns, masks, names = [], [], []
    for clock in CLOCKS:
        for h in HORIZONS_MS:
            ahead = base + h // 100
            present = ahead < len(frame['mid'])
            safe = np.minimum(ahead, len(frame['mid']) - 1)
            native = clock != 'receive'
            future_age = frame['truth_age_sec' if native else 'age_sec'][safe]
            current_age = frame['truth_age_sec' if clock == 'native_innovation' else 'age_sec'][base]
            for side in ('bid', 'ask'):
                future = frame[('truth_' if native else '') + side][safe]
                current = frame[('truth_' if clock == 'native_innovation' else '') + side][base]
                y = 10000. * (future/current - 1)
                y[~present] = np.nan
                columns.append(y)
                masks.append(np.column_stack((present, current_age, future_age)))
                names.append((clock, h, side))
    return np.column_stack(columns), masks, names


@dataclass
class Moments:
    n: int
    sx: np.ndarray
    sy: np.ndarray
    xx: np.ndarray
    xy: np.ndarray

    @classmethod
    def empty(cls, p, q):
        return cls(0, np.zeros(p), np.zeros(q), np.zeros((p, p)), np.zeros((p, q)))

    def update(self, x, y):
        if len(x):
            self.n += len(x)
            self.sx += x.sum(axis=0)
            self.sy += y.sum(axis=0)
            self.xx += x.T @ x
            self.xy += x.T @ y

    def solve(self, selection):
        if self.n <= len(selection)+1:
            return None
        ids = np.asarray(selection)
        mx, my = self.sx[ids]/self.n, self.sy/self.n
        cov = self.xx[np.ix_(ids, ids)]/self.n - np.outer(mx, mx)
        cov = (cov + cov.T)/2
        cross = self.xy[ids]/self.n - np.outer(mx, my)
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        sd = np.where(sd > 1e-14, sd, 1.)
        eigen, vectors = np.linalg.eigh(cov / np.outer(sd, sd))
        keep = eigen > max(1e-14, float(eigen.max())*1e-10)
        if keep.any():
            b = vectors[:, keep] @ ((vectors[:, keep].T @ (cross / sd[:, None])) / eigen[keep, None])
            b /= sd[:, None]
        else:
            b = np.zeros((len(ids), len(my)))
        full = np.zeros((len(self.sx), len(my)))
        full[ids] = b
        return my - mx @ b, full


def fit_families(moment):
    own = list(range(len(OWN_NAMES)))
    fitted = [moment.solve(own)]
    for family in FAMILIES:
        fitted.append(moment.solve(own + [len(OWN_NAMES)+j for j in GROUPS[family]]))
    return fitted


def prediction_matrix(x, fitted, outputs=None):
    if fitted[0] is None:
        return None
    intercept = np.concatenate([f[0] if outputs is None else f[0][outputs] for f in fitted])
    coefficient = np.concatenate([f[1] if outputs is None else f[1][:, outputs] for f in fitted], axis=1)
    return intercept + x @ coefficient


def score_rows(xpred, y, names, current_valid, label_masks, age, context,
               common_valid, sessions, frame, day, phase=0):
    q = len(names)
    rows = []
    # Each fitted block contains all label outputs in the same fixed order.
    for j, (clock, horizon, side) in enumerate(names):
        if age == 5 and horizon not in (1000, 5000):
            continue
        lm = label_masks[j]
        valid = current_valid & lm[:, 0].astype(bool) & (lm[:, 1] >= 0) & (lm[:, 1] <= age) & (lm[:, 2] >= 0) & (lm[:, 2] <= age) & np.isfinite(y[:, j])
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        actual = y[ids, j]
        own = xpred[ids, j]
        own_error = np.abs(actual-own)
        nonzero = actual != 0
        base_hits = np.sign(own) == np.sign(actual)
        common = common_valid[ids]
        # Clock session and tick-regime groups are evaluated on identical rows
        # within a family comparison; these are not trade-entry filters.
        regular = sessions[ids]
        high_tick = frame['mid'][phase::10][ids] >= 1000
        for f, family in enumerate(FAMILIES, start=1):
            pred = xpred[ids, f*q+j]
            error = np.abs(actual-pred)
            gain = own_error-error
            record = dict(context, day=day, clock=clock, horizon_ms=horizon, side=side,
                          family=family, max_age=age, n=len(ids), nonzero=int(nonzero.sum()),
                          abs_sum=float(error.sum()), sq_sum=float(np.square(actual-pred).sum()),
                          hits=int(((np.sign(pred)==np.sign(actual)) & nonzero).sum()),
                          own_abs_sum=float(own_error.sum()), own_sq_sum=float(np.square(actual-own).sum()),
                          own_hits=int((base_hits & nonzero).sum()),
                          nochange_abs_sum=float(np.abs(actual).sum()),
                          gain_sum=float(gain.sum()), improved_observations=int((gain > 0).sum()),
                          common_short_n=int(common.sum()), common_short_gain_sum=float(gain[common].sum()),
                          regular_n=int(regular.sum()), regular_gain_sum=float(gain[regular].sum()),
                          other_n=int((~regular).sum()), other_gain_sum=float(gain[~regular].sum()),
                          high_price_n=int(high_tick.sum()), high_price_gain_sum=float(gain[high_tick].sum()),
                          low_price_n=int((~high_tick).sum()), low_price_gain_sum=float(gain[~high_tick].sum()),
                          weekend=date.fromisoformat(day).weekday() >= 5)
            rows.append(record)
    return rows


def checkpoint_state(path, states):
    payload = {}
    for (age, d), state in states.items():
        prefix = f'a{age}_d{d}_'
        for key in ('n', 'sx', 'sy', 'xx', 'xy'):
            payload[prefix+key] = np.asarray(getattr(state, key))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def run_pair(source, target, wait_seconds=7200, phase=0):
    pair = source + '__' + target
    folder = OUT / pair
    folder.mkdir(parents=True, exist_ok=True)
    previous_b, previous_h, last_basis = None, None, None
    states, fingerprints = {}, []
    for di, day in enumerate(DAYS):
        begun = time.monotonic()
        deadline = begun + wait_seconds
        while not all((INPUT/m/(day+'.npz')).exists() for m in (source, target)):
            if time.monotonic() > deadline:
                raise FileNotFoundError(f'Inputs not ready {pair}/{day}')
            print('WAIT', pair, day, flush=True)
            time.sleep(5)
        source_frame, target_frame = load_day(source, day), load_day(target, day)
        if not np.array_equal(source_frame['grid_us'], target_frame['grid_us']):
            raise AssertionError('Paired grids differ')
        for market in (source, target):
            p = INPUT/market/(day+'.npz')
            fingerprints.append(dict(market=market, day=day, bytes=p.stat().st_size,
                                     mtime_ns=p.stat().st_mtime_ns))
        sb, th = augmented(source_frame, previous_b), augmented(target_frame, previous_h)
        basis_today, last_basis = causal_basis(source_frame['mid'], target_frame['mid'], last_basis)
        basis = np.concatenate((previous_b['_basis'][-TAIL:] if previous_b is not None else np.full(TAIL, np.nan), basis_today))
        base = np.arange(phase, len(target_frame['mid']), 10)
        at = TAIL+base
        own, own_ages = own_features(th, at)
        y, label_masks, names = labels(target_frame, phase)
        sessions = (base >= 9*36000) & (base < (15*3600+20*60)*10)
        own_good = np.isfinite(own).all(axis=1) & np.isfinite(own_ages).all(axis=1)
        age_valid = {age: own_good & (own_ages >= 0).all(axis=1) & (own_ages <= age).all(axis=1) for age in (1, 5)}
        short_common = {age: age_valid[age].copy() for age in (1, 5)}
        for d in DELAYS_MS:
            if d > 1000:
                break
            s, ages = source_features(sb, target_frame['mid'][base], basis, at-d//100)
            for age in (1, 5):
                short_common[age] &= np.isfinite(s).all(axis=1) & np.isfinite(ages).all(axis=1) & (ages >= 0).all(axis=1) & (ages <= age).all(axis=1)
        # Training uses a common historical label intersection for every
        # output, so its X'X is shared. Future validity never gates predictions.
        historical_label_mask = {}
        for age in (1, 5):
            use = np.isfinite(y).all(axis=1)
            for lm in label_masks:
                use &= lm[:, 0].astype(bool) & (lm[:, 1] >= 0) & (lm[:, 1] <= age) & (lm[:, 2] >= 0) & (lm[:, 2] <= age)
            historical_label_mask[age] = use
        rows, coeff = [], {}
        frozen = {age: fit_families(states.setdefault((age, 0), Moments.empty(len(FEATURE_NAMES), len(names)))) for age in (1, 5)}
        for d in DELAYS_MS:
            source_x, source_ages = source_features(sb, target_frame['mid'][base], basis, at-d//100)
            x = np.column_stack((own, source_x))
            finite = np.isfinite(x).all(axis=1) & np.isfinite(source_ages).all(axis=1)
            safe_x = np.where(np.isfinite(x), x, 0.)
            for age in (1, 5):
                current_valid = age_valid[age] & finite & (source_ages >= 0).all(axis=1) & (source_ages <= age).all(axis=1)
                moment = states.setdefault((age, d), Moments.empty(len(FEATURE_NAMES), len(names)))
                refit = fit_families(moment)
                for mode, fitted in (('refit', refit), ('frozen', frozen[age])):
                    if di == 0 or fitted[0] is None:
                        continue
                    outputs = np.arange(len(names)) if age == 1 else np.array([j for j, name in enumerate(names) if name[1] in (1000, 5000)])
                    prediction = prediction_matrix(safe_x, fitted, outputs)
                    context = dict(source=source, target=target,
                                   pair_type='matched' if (source.endswith('SKHYNIXUSDT') == target.endswith('SKHX')) else 'cross_underlying',
                                   delay_ms=d, mode=mode)
                    rows.extend(score_rows(prediction, y[:, outputs], [names[j] for j in outputs], current_valid, [label_masks[j] for j in outputs], age,
                                           context, short_common[age] if d <= 1000 else np.zeros(len(base), bool),
                                           sessions, target_frame, day))
                    # Compact multioutput coefficients: output names/features
                    # are globally fixed in metadata, every day is recoverable.
                    coeff[f'a{age}_d{d}_{mode}_intercept'] = np.stack([f[0] for f in fitted])
                    coeff[f'a{age}_d{d}_{mode}_coefficient'] = np.stack([f[1] for f in fitted])
                training = current_valid & historical_label_mask[age]
                moment.update(x[training], y[training])
        daily = pd.DataFrame(rows)
        if len(daily):
            daily.to_parquet(folder/(day+'.parquet'), index=False)
            np.savez_compressed(folder/(day+'.coefficients.npz'), **coeff)
        checkpoint_state(folder/(day+'.state.npz'), states)
        source_frame['_basis'] = basis_today
        previous_b, previous_h = source_frame, target_frame
        print('EVALUATED', pair, day, 'rows', len(rows), 'seconds', round(time.monotonic()-begun, 2), flush=True)
    (folder/'provenance.json').write_text(json.dumps(dict(source=source, target=target,
                                                        fingerprints=fingerprints, days=list(DAYS)), indent=2))


def aggregate(bootstrap=2000, matched_only=False):
    paths = sorted(OUT.glob('*__*/2026-*.parquet'))
    if matched_only:
        names = {BN[0]+'__'+HL[0], BN[1]+'__'+HL[1]}
        paths = [p for p in paths if p.parent.name in names]
        if len(paths) != 34:
            raise ValueError('Both matched pairs must have all 17 OOS dates')
    if not paths:
        raise ValueError('No daily evaluation files')
    daily = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    keys = ['source', 'target', 'pair_type', 'clock', 'delay_ms', 'horizon_ms', 'max_age', 'mode', 'family', 'side']
    counts = ['n', 'nonzero', 'abs_sum', 'sq_sum', 'hits', 'own_abs_sum', 'own_sq_sum', 'own_hits',
              'nochange_abs_sum', 'gain_sum', 'improved_observations', 'common_short_n', 'common_short_gain_sum',
              'regular_n', 'regular_gain_sum', 'other_n', 'other_gain_sum',
              'high_price_n', 'high_price_gain_sum', 'low_price_n', 'low_price_gain_sum']
    grouped = daily.groupby(keys, sort=True)
    summary = grouped[counts].sum()
    summary['days'] = grouped.day.nunique()
    summary['improved_days_vs_own'] = grouped.gain_sum.apply(lambda x: int((x > 0).sum()))
    summary['mae_bp'] = summary.abs_sum/summary.n
    summary['rmse_bp'] = np.sqrt(summary.sq_sum/summary.n)
    summary['hit_nonzero'] = summary.hits/summary.nonzero.replace(0, np.nan)
    summary['own_mae_bp'] = summary.own_abs_sum/summary.n
    summary['nochange_mae_bp'] = summary.nochange_abs_sum/summary.n
    summary['gain_vs_own_bp'] = summary.gain_sum/summary.n
    summary['gain_vs_nochange_bp'] = summary.nochange_mae_bp-summary.mae_bp
    summary['common_short_gain_vs_own_bp'] = summary.common_short_gain_sum/summary.common_short_n.replace(0, np.nan)
    for state in ('regular', 'other', 'high_price', 'low_price'):
        summary[state+'_gain_vs_own_bp'] = summary[state+'_gain_sum']/summary[state+'_n'].replace(0, np.nan)
    # Same date resampling for every model/lag/side gives paired pointwise CIs
    # and a family+lag+side simultaneous max-standardized-band approximation.
    index = summary.index
    gains = daily.pivot(index=keys, columns='day', values='gain_sum').reindex(index).fillna(0).reindex(columns=DAYS[1:], fill_value=0).to_numpy()
    ns = daily.pivot(index=keys, columns='day', values='n').reindex(index).fillna(0).reindex(columns=DAYS[1:], fill_value=0).to_numpy()
    rng = np.random.default_rng(20261108)
    picks = rng.integers(0, len(DAYS)-1, (bootstrap, len(DAYS)-1))
    weights = np.zeros((bootstrap, len(DAYS)-1))
    for b in range(bootstrap):
        weights[b] = np.bincount(picks[b], minlength=len(DAYS)-1)
    errors = np.zeros((len(summary), bootstrap), dtype=np.float32)
    point = summary.gain_vs_own_bp.to_numpy()
    low, high, sd = np.zeros(len(summary)), np.zeros(len(summary)), np.zeros(len(summary))
    for start in range(0, len(summary), 512):
        stop = min(start+512, len(summary))
        den = ns[start:stop] @ weights.T
        means = np.divide(gains[start:stop] @ weights.T, den,
                          out=np.full(den.shape, np.nan), where=den > 0)
        low[start:stop], high[start:stop] = np.nanquantile(means, [.025, .975], axis=1)
        sd[start:stop] = np.nanstd(means, axis=1, ddof=1)
        errors[start:stop] = means - point[start:stop, None]
    summary['gain_vs_own_ci_low'], summary['gain_vs_own_ci_high'] = low, high
    summary['bootstrap_sd_bp'] = sd
    bands = summary.reset_index()
    bandkeys = ['source', 'target', 'clock', 'horizon_ms', 'max_age', 'mode']
    critical = np.zeros(len(summary))
    for _, block in bands.groupby(bandkeys, sort=False):
        rows = block.index.to_numpy()
        student = errors[rows].astype(float) / np.maximum(sd[rows, None], 1e-15)
        max_t = np.nanmax(np.abs(student), axis=0)
        critical[rows] = float(np.nanquantile(max_t, .95))
    summary['simultaneous_critical'] = critical
    summary['gain_vs_own_simul_low'] = point-critical*sd
    summary['gain_vs_own_simul_high'] = point+critical*sd
    # Day-side worst values and weekend/generalization summaries.
    daily['daily_gain_bp'] = daily.gain_sum/daily.n
    summary['worst_day_gain_bp'] = daily.groupby(keys).daily_gain_bp.min().reindex(index)
    for label, weekend in (('weekday', False), ('weekend', True)):
        block = daily[daily.weekend == weekend].groupby(keys)[['gain_sum', 'n']].sum().reindex(index)
        summary[label+'_gain_vs_own_bp'] = block.gain_sum/block.n
    stem = 'matched_' if matched_only else ''
    daily.to_parquet(OUT/(stem+'daily.parquet'), index=False)
    daily.to_csv(OUT/(stem+'daily.csv.gz'), index=False, compression='gzip')
    summary.reset_index().to_csv(OUT/(stem+'summary.csv'), index=False)
    (OUT/(stem+'metadata.json')).write_text(json.dumps(dict(
        version=VERSION, days=list(DAYS), oos_days=list(DAYS[1:]), pairs=sorted(set(zip(daily.source,daily.target))),
        delays_ms=list(DELAYS_MS), horizons_ms=list(HORIZONS_MS), families=list(FAMILIES),
        clocks=list(CLOCKS), age5_horizons_ms=[1000, 5000], decision_interval_ms=1000,
        phase_ms=0, feature_and_label_resolution_ms=100, basis_ewma_half_life_seconds=300,
        input_fields_float64_for_computation=True, prediction_features=list(FEATURE_NAMES),
        micro_definition='L1 weighted-mid proxy, not fitted Stoikov Markov micro-price',
        sampling='one-second decisions, 100ms features/lags/future endpoints',
        signal_age='original source feed age checked at its own timestamp; intentional signal delay separately reported',
        source_freshness='current,100/500/1000/5000ms return endpoints all satisfy max_age at original timestamps',
        historical_training='expanding prior days only, common finite-label/freshness intersection across all horizons and clocks; current-day future masks are never prediction masks',
        native_innovation='truth_future / truth_current - 1; offline remaining market movement; current native price not necessarily known',
        native_endpoint='truth_future / received_current - 1; future executable quote forecast plus possible feed catch-up',
        frozen='previous-day delay-zero coefficient applied to aged BN features plus current HL features',
        basis='as-of 100ms logBN-logHL EWMA at aged-source timestamp, no current BN information in aged features',
        bootstrap=dict(samples=bootstrap, unit='whole KST dates; fixed historically fitted forecasts, not retrained bootstrap',
                       simultaneous_scope='families x delays x bid/ask per pair,clock,horizon,max_age,mode',
                       method='max absolute centered day-bootstrap deviation standardized by point bootstrap SD'),
        reused_research_period=True, daily_rows=len(daily), summary_rows=len(summary),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print('AGGREGATED', len(daily), len(summary), flush=True)


class Tests(unittest.TestCase):
    def frame(self, n=900):
        mid = 100 + np.arange(n)*.0001
        return dict(mid=mid, bid=mid-.005, ask=mid+.005, imbalance=np.full(n, .2),
                    spread_bp=10000*.01/mid, weighted_offset_bp=10000*.001/mid,
                    ofi100_norm=np.zeros(n), ofi500_norm=np.zeros(n), ofi1000_norm=np.zeros(n),
                    age_sec=np.full(n, .1), truth_bid=mid-.004, truth_ask=mid+.006,
                    truth_age_sec=np.full(n, .05))

    def test_weighted_mid_obi_identity(self):
        f = self.frame()
        wm = f['mid']*(1+f['weighted_offset_bp']/10000)
        expected = f['mid'] + .5*(f['ask']-f['bid'])*f['imbalance']
        np.testing.assert_allclose(wm, expected, atol=1e-12)

    def test_aged_source_has_no_current_source_leak(self):
        b, h = self.frame(), self.frame()
        basis, _ = causal_basis(b['mid'], h['mid'])
        at = np.array([600])
        x, _ = source_features(b, h['mid'][700:701], basis, at)
        mutated = {k: v.copy() for k, v in b.items()}
        for value in mutated.values():
            value[601:] *= 2
        basis2, _ = causal_basis(mutated['mid'], h['mid'])
        z, _ = source_features(mutated, h['mid'][700:701], basis2, at)
        np.testing.assert_array_equal(x, z)

    def test_future_label_mutation_does_not_change_forecast(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(200, 5)), rng.normal(size=(200, 2))
        past = Moments.empty(5, 2)
        past.update(x[:100], y[:100])
        before = past.solve(range(5))
        y[100:] *= -1000
        after = past.solve(range(5))
        np.testing.assert_array_equal(before[0]+x[100:]@before[1], after[0]+x[100:]@after[1])

    def test_future_masks_do_not_mutate_current_features(self):
        f = self.frame()
        a = augmented(f, None)
        at = np.arange(700, 1600, 10)
        x, age = own_features(a, at)
        f['truth_bid'][300:] *= .5
        labels(f)
        xx, aa = own_features(a, at)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(age, aa)


def phase_check(wait_seconds=7200):
    """300ms shifted decisions, original phase-zero d0 frozen coefficients.

    This is a timestamp-phase stress, not another model-selection sweep.
    """
    rows = []
    for source, target in ((BN[0], HL[0]), (BN[1], HL[1])):
        previous_b, previous_h, last_basis = None, None, None
        pair = source+'__'+target
        for di, day in enumerate(DAYS):
            deadline = time.monotonic()+wait_seconds
            coefficient_path = OUT/pair/(day+'.coefficients.npz')
            if di:
                while not coefficient_path.exists():
                    if time.monotonic() > deadline:
                        raise FileNotFoundError(coefficient_path)
                    time.sleep(5)
            b, h = load_day(source, day), load_day(target, day)
            sb, th = augmented(b, previous_b), augmented(h, previous_h)
            basis_now, last_basis = causal_basis(b['mid'], h['mid'], last_basis)
            basis = np.concatenate((previous_b['_basis'][-TAIL:] if previous_b is not None
                                    else np.full(TAIL, np.nan), basis_now))
            if di:
                with np.load(coefficient_path) as coefficients:
                    intercepts = coefficients['a1_d0_frozen_intercept']
                    coefs = coefficients['a1_d0_frozen_coefficient']
                fitted = [(a, c) for a, c in zip(intercepts, coefs)]
                base = np.arange(3, len(h['mid']), 10)
                at = TAIL+base
                own, own_ages = own_features(th, at)
                own_good = (np.isfinite(own).all(axis=1) & np.isfinite(own_ages).all(axis=1)
                            & (own_ages >= 0).all(axis=1) & (own_ages <= 1).all(axis=1))
                y, masks, names = labels(h, phase=3)
                outputs = np.array([j for j, name in enumerate(names) if name[1] in (1000, 5000)])
                sessions = (base >= 9*36000) & (base < (15*3600+20*60)*10)
                for d in (0, 500, 800, 1000, 2000, 5000):
                    s, ages = source_features(sb, h['mid'][base], basis, at-d//100)
                    x = np.column_stack((own, s))
                    valid = own_good & np.isfinite(x).all(axis=1) & np.isfinite(ages).all(axis=1) & (ages >= 0).all(axis=1) & (ages <= 1).all(axis=1)
                    pred = prediction_matrix(np.where(np.isfinite(x), x, 0.), fitted, outputs)
                    context = dict(source=source, target=target, pair_type='matched', delay_ms=d,
                                   mode='frozen', phase_ms=300)
                    rows.extend(score_rows(pred, y[:, outputs], [names[j] for j in outputs], valid,
                                           [masks[j] for j in outputs], 1, context,
                                           np.zeros(len(base), bool), sessions, h, day, phase=3))
                print('PHASE_CHECK', pair, day, flush=True)
            b['_basis'] = basis_now
            previous_b, previous_h = b, h
    daily = pd.DataFrame(rows)
    daily.to_parquet(OUT/'phase300_daily.parquet', index=False)
    keys = ['source', 'target', 'clock', 'horizon_ms', 'delay_ms', 'family', 'side', 'mode', 'max_age', 'phase_ms']
    summary = daily.groupby(keys)[['n', 'gain_sum', 'abs_sum', 'own_abs_sum']].sum()
    summary['gain_vs_own_bp'] = summary.gain_sum/summary.n
    summary['mae_bp'], summary['own_mae_bp'] = summary.abs_sum/summary.n, summary.own_abs_sum/summary.n
    summary['days'] = daily.groupby(keys).day.nunique()
    summary['improved_days_vs_own'] = daily.groupby(keys).gain_sum.apply(lambda x: int((x > 0).sum()))
    gains = daily.pivot(index=keys, columns='day', values='gain_sum').reindex(summary.index).reindex(columns=DAYS[1:]).fillna(0).to_numpy()
    ns = daily.pivot(index=keys, columns='day', values='n').reindex(summary.index).reindex(columns=DAYS[1:]).fillna(0).to_numpy()
    rng = np.random.default_rng(20261108)
    pick = rng.integers(0, len(DAYS)-1, (2000, len(DAYS)-1))
    weights = np.array([np.bincount(row, minlength=len(DAYS)-1) for row in pick])
    ratio = (gains @ weights.T)/(ns @ weights.T)
    summary['gain_vs_own_ci_low'], summary['gain_vs_own_ci_high'] = np.quantile(ratio, [.025, .975], axis=1)
    summary.reset_index().to_csv(OUT/'phase300_summary.csv', index=False)
    (OUT/'phase300_metadata.json').write_text(json.dumps(dict(
        phase_ms=300, training='phase0 previous-day d0 coefficients reused, no refitting',
        pairs=[(BN[0], HL[0]), (BN[1], HL[1])], horizons_ms=[1000, 5000],
        delays_ms=[0, 500, 800, 1000, 2000, 5000], max_age=1,
        summary_rows=len(summary), daily_rows=len(daily), bootstrap='2000 whole dates, pointwise 95%'), indent=2))


def fit_guard_families(moment, own_count):
    own = list(range(own_count))
    return [moment.solve(own)] + [moment.solve(own+[own_count+j for j in GROUPS[f]]) for f in FAMILIES]


def compact_summary(daily, keys, output):
    """Guard summaries with paired whole-date pointwise and simultaneous bands."""
    sums = ['n', 'nonzero', 'abs_sum', 'sq_sum', 'hits', 'own_abs_sum', 'own_sq_sum', 'own_hits',
            'nochange_abs_sum', 'gain_sum']
    g = daily.groupby(keys)
    s = g[sums].sum()
    s['days'], s['improved_days_vs_own'] = g.day.nunique(), g.gain_sum.apply(lambda x: int((x > 0).sum()))
    s['mae_bp'], s['own_mae_bp'] = s.abs_sum/s.n, s.own_abs_sum/s.n
    s['gain_vs_own_bp'] = s.gain_sum/s.n
    s['gain_vs_nochange_bp'] = (s.nochange_abs_sum-s.abs_sum)/s.n
    gains = daily.pivot(index=keys, columns='day', values='gain_sum').reindex(s.index).reindex(columns=DAYS[1:]).fillna(0).to_numpy()
    ns = daily.pivot(index=keys, columns='day', values='n').reindex(s.index).reindex(columns=DAYS[1:]).fillna(0).to_numpy()
    rng = np.random.default_rng(20261108)
    pick = rng.integers(0, 17, (2000, 17))
    weights = np.array([np.bincount(row, minlength=17) for row in pick])
    ratio = (gains @ weights.T)/(ns @ weights.T)
    s['gain_vs_own_ci_low'], s['gain_vs_own_ci_high'] = np.quantile(ratio, [.025, .975], axis=1)
    sd = ratio.std(axis=1, ddof=1)
    point = s.gain_vs_own_bp.to_numpy()
    frame = s.reset_index()
    critical = np.zeros(len(frame))
    groupkeys = ['source', 'target', 'clock', 'horizon_ms', 'mode', 'guard']
    for _, block in frame.groupby(groupkeys):
        ix = block.index.to_numpy()
        errors = (ratio[ix]-point[ix, None])/np.maximum(sd[ix, None], 1e-15)
        critical[ix] = np.quantile(np.abs(errors).max(axis=0), .95)
    s['gain_vs_own_simul_low'], s['gain_vs_own_simul_high'] = point-critical*sd, point+critical*sd
    s.reset_index().to_csv(output, index=False)


def extended_guard():
    """Long HL-memory controls guard transport-gap attribution.

    Frozen models add current HL 5/30/60s returns, its 300s EMA residual and
    event age. Refit models additionally control the exact aged-HL component
    carried inside the transported cross-market level gap.
    """
    delays = (0, 500, 800, 1000, 2000, 5000, 10000, 30000, 60000)
    all_rows = []
    for source, target in ((BN[0], HL[0]), (BN[1], HL[1])):
        folder = OUT/'extended_guard'/(source+'__'+target)
        folder.mkdir(parents=True, exist_ok=True)
        previous_b, previous_h, last_basis, last_h_ema = None, None, None, None
        states = {}
        for di, day in enumerate(DAYS):
            started = time.monotonic()
            while not all((INPUT/m/(day+'.npz')).exists() for m in (source, target)):
                time.sleep(5)
            b, h = load_day(source, day), load_day(target, day)
            sb, th = augmented(b, previous_b), augmented(h, previous_h)
            basis_now, last_basis = causal_basis(b['mid'], h['mid'], last_basis)
            ema_now, last_h_ema = causal_basis(h['mid'], np.ones(len(h['mid'])), last_h_ema)
            basis = np.concatenate((previous_b['_basis'][-TAIL:] if previous_b is not None else np.full(TAIL, np.nan), basis_now))
            ema_h = np.concatenate((previous_h['_ema'][-TAIL:] if previous_h is not None else np.full(TAIL, np.nan), ema_now))
            base = np.arange(0, len(h['mid']), 10)
            at = TAIL+base
            own, own_ages = own_features(th, at)
            extra = np.column_stack((*[log_return(th['mid'], at, back) for back in (50, 300, 600)],
                                     10000*(np.log(th['mid'][at])-ema_h[at]), th['age_sec'][at]))
            extended = np.column_stack((own, extra))
            own_count = extended.shape[1]
            long_ages = np.column_stack([th['age_sec'][at-back] for back in (50, 300, 600)])
            own_valid = np.isfinite(extended).all(axis=1) & np.isfinite(own_ages).all(axis=1) & np.isfinite(long_ages).all(axis=1)
            own_valid &= (own_ages >= 0).all(axis=1) & (own_ages <= 1).all(axis=1) & (long_ages >= 0).all(axis=1) & (long_ages <= 1).all(axis=1)
            y, masks, names = labels(h)
            outputs = np.array([j for j, name in enumerate(names) if name[1] in (1000, 5000)])
            historical = np.isfinite(y).all(axis=1)
            for lm in masks:
                historical &= lm[:, 0].astype(bool) & (lm[:, 1] >= 0) & (lm[:, 1] <= 1) & (lm[:, 2] >= 0) & (lm[:, 2] <= 1)
            sessions = (base >= 9*36000) & (base < (15*3600+20*60)*10)
            frozen_state = states.setdefault(('extended', 0), Moments.empty(own_count+len(SOURCE_NAMES), len(names)))
            frozen = fit_guard_families(frozen_state, own_count)
            rows = []
            for d in delays:
                source_x, source_ages = source_features(sb, h['mid'][base], basis, at-d//100)
                x_frozen = np.column_stack((extended, source_x))
                # Linear EWMA decomposition: this is the known HL term inside
                # log BN(t-d) - basis(t-d) - log HL(t), preventing its history
                # from being credited to Binance.
                target_term = 10000*(ema_h[at-d//100]-np.log(th['mid'][at]))
                x_refit = np.column_stack((extended, target_term, source_x))
                refit_state = states.setdefault(('exact', d), Moments.empty(own_count+1+len(SOURCE_NAMES), len(names)))
                fitted = fit_guard_families(refit_state, own_count+1)
                valid = own_valid & np.isfinite(x_refit).all(axis=1) & np.isfinite(source_ages).all(axis=1) & (source_ages >= 0).all(axis=1) & (source_ages <= 1).all(axis=1)
                if di:
                    for mode, guard, features, coeff in (
                        ('frozen', 'extended_current', x_frozen, frozen),
                        ('refit', 'extended_exact_transport', x_refit, fitted)):
                        pred = prediction_matrix(np.where(np.isfinite(features), features, 0), coeff, outputs)
                        if pred is None:
                            continue
                        context = dict(source=source, target=target, pair_type='matched', delay_ms=d,
                                       mode=mode, guard=guard)
                        rows.extend(score_rows(pred, y[:, outputs], [names[j] for j in outputs], valid,
                                               [masks[j] for j in outputs], 1, context,
                                               np.zeros(len(base), bool), sessions, h, day))
                training = valid & historical
                refit_state.update(x_refit[training], y[training])
                if d == 0:
                    frozen_state.update(x_frozen[training], y[training])
            if rows:
                pd.DataFrame(rows).to_parquet(folder/(day+'.parquet'), index=False)
                all_rows.extend(rows)
            b['_basis'], h['_ema'] = basis_now, ema_now
            previous_b, previous_h = b, h
            print('EXTENDED_GUARD', source, target, day, 'seconds', round(time.monotonic()-started, 2), flush=True)
    daily = pd.DataFrame(all_rows)
    daily.to_parquet(OUT/'extended_guard_daily.parquet', index=False)
    keys = ['source', 'target', 'pair_type', 'clock', 'horizon_ms', 'delay_ms', 'max_age', 'mode', 'guard', 'family', 'side']
    compact_summary(daily, keys, OUT/'extended_guard_summary.csv')
    (OUT/'extended_guard_metadata.json').write_text(json.dumps(dict(
        own_added=['HL return 5s', 'HL return 30s', 'HL return 60s', 'HL log-mid minus 300s log EMA', 'HL event age'],
        refit_addition='exact HL EMA(t-delay)-HL logmid(t) term carried by source level-gap basis',
        frozen_control='current extended HL features remain current; phase0 d0 past coefficients only',
        refit_control='age-conditioned exact HL transport component also included, no d0 frozen collinear extrapolation',
        delays_ms=list(delays), horizons_ms=[1000, 5000], max_age=1,
        bootstrap='whole date 2000 draws; max-standardized family/delay/side simultaneous bands',
        daily_rows=len(daily)), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=BN)
    parser.add_argument('--target', choices=HL)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--no-aggregate', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--phase-only', action='store_true')
    parser.add_argument('--extended-only', action='store_true')
    parser.add_argument('--matched-aggregate-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.matched_aggregate_only:
        aggregate(args.bootstrap, matched_only=True)
        return
    if args.extended_only:
        extended_guard()
        return
    if args.phase_only:
        phase_check()
        return
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        (OUT/'tests.json').write_text(json.dumps(dict(tests=result.testsRun, passed=result.wasSuccessful(),
                                                    failures=len(result.failures), errors=len(result.errors)), indent=2))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    if not args.aggregate_only:
        pairs = [(args.source, args.target)] if args.source and args.target else PAIRS
        for source, target in pairs:
            run_pair(source, target)
    if not args.no_aggregate:
        aggregate(args.bootstrap)


if __name__ == '__main__':
    main()
