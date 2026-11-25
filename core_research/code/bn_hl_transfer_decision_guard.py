"""Residual native price prediction after waiting, with the whole decision frozen.

Reuses prior-day delay-zero coefficients. It never substitutes current HL
information after the hypothetical order was sent. Differencing two endpoint
forecasts cancels feed catch-up completed before the window starts.
"""
from __future__ import annotations

import json
import time
from zipfile import BadZipFile
from pathlib import Path
import numpy as np
import pandas as pd

from work import bn_hl_transfer_eval as E

OUT = E.ROOT / 'out_bn_hl_transfer/decision_guard'
PAIRS = [(E.BN[0], E.HL[0]), (E.BN[1], E.HL[1])]
WINDOWS = ((100, 500), (500, 1000), (500, 5000), (1000, 5000), (5000, 30000))


def frozen_difference(predicted_end, predicted_start):
    return predicted_end - predicted_start


def run_pair(source, target):
    pair = source + '__' + target
    folder = OUT / pair
    folder.mkdir(parents=True, exist_ok=True)
    previous_b, previous_h, last_basis = None, None, None
    for day_index, day in enumerate(E.DAYS):
        source_frame, target_frame = E.load_day(source, day), E.load_day(target, day)
        sb, th = E.augmented(source_frame, previous_b), E.augmented(target_frame, previous_h)
        basis_today, last_basis = E.causal_basis(source_frame['mid'], target_frame['mid'], last_basis)
        basis = np.r_[previous_b['_basis'][-E.TAIL:] if previous_b is not None else np.full(E.TAIL, np.nan), basis_today]
        if day_index:
            output = folder / (day + '.parquet')
            if not output.exists():
                coeff_path = E.OUT / pair / (day + '.coefficients.npz')
                while not coeff_path.exists():
                    print('WAIT_GUARD', pair, day, flush=True)
                    time.sleep(5)
                while True:
                    try:
                        with np.load(coeff_path) as f:
                            intercept = f['a1_d0_frozen_intercept'].copy()
                            coefficient = f['a1_d0_frozen_coefficient'].copy()
                        break
                    except (OSError, ValueError, EOFError, BadZipFile):
                        time.sleep(1)
                names = [(clock, h, side) for clock in E.CLOCKS
                         for h in E.HORIZONS_MS for side in ('bid', 'ask')]
                selected = [i for i, key in enumerate(names) if key[0] == 'native_endpoint']
                compact_names = [names[i] for i in selected]
                ints = intercept[:, selected]
                coef = coefficient[:, :, selected]
                records = []
                for phase in (0, 3):
                    base = np.arange(phase, len(target_frame['mid']), 10)
                    at = E.TAIL + base
                    own, own_ages = E.own_features(th, at)
                    source_x, source_ages = E.source_features(sb, target_frame['mid'][base], basis, at)
                    x = np.c_[own, source_x]
                    valid = (np.isfinite(x).all(axis=1)
                             & (own_ages >= 0).all(axis=1) & (own_ages <= 1).all(axis=1)
                             & (source_ages >= 0).all(axis=1) & (source_ages <= 1).all(axis=1))
                    safe_x = np.where(np.isfinite(x), x, 0.)
                    predictions = (ints.reshape(-1) + safe_x @ coef.transpose(1, 0, 2).reshape(len(E.FEATURE_NAMES), -1)).reshape(len(base), len(E.FAMILIES)+1, len(selected))
                    for start, end in WINDOWS:
                        start_at, end_at = base + start//100, base + end//100
                        present = end_at < len(target_frame['mid'])
                        a, b = np.minimum(start_at, len(target_frame['mid'])-1), np.minimum(end_at, len(target_frame['mid'])-1)
                        for side in ('bid', 'ask'):
                            js = compact_names.index(('native_endpoint', start, side))
                            je = compact_names.index(('native_endpoint', end, side))
                            current = target_frame[side][base]
                            actual = 10000 * (target_frame['truth_'+side][b]-target_frame['truth_'+side][a])/current
                            use = (valid & present & np.isfinite(actual)
                                   & (target_frame['truth_age_sec'][a] >= 0)
                                   & (target_frame['truth_age_sec'][a] <= 1)
                                   & (target_frame['truth_age_sec'][b] >= 0)
                                   & (target_frame['truth_age_sec'][b] <= 1))
                            actual = actual[use]
                            own_pred = frozen_difference(predictions[:, 0, je], predictions[:, 0, js])[use]
                            own_error = np.abs(actual-own_pred)
                            for fi, family in enumerate(E.FAMILIES, 1):
                                predicted = frozen_difference(predictions[:, fi, je], predictions[:, fi, js])[use]
                                error = np.abs(actual-predicted)
                                records.append(dict(source=source, target=target, day=day,
                                                    family=family, side=side, phase_ms=phase*100,
                                                    start_ms=start, end_ms=end, n=len(actual),
                                                    nonzero=int(np.sum(actual != 0)),
                                                    abs_sum=float(error.sum()), own_abs_sum=float(own_error.sum()),
                                                    nochange_abs_sum=float(np.abs(actual).sum()),
                                                    gain_sum=float((own_error-error).sum()),
                                                    hits=int(np.sum((np.sign(predicted)==np.sign(actual)) & (actual != 0)))))
                pd.DataFrame(records).to_parquet(output, index=False)
                print('DECISION_GUARD', pair, day, len(records), flush=True)
        source_frame['_basis'] = basis_today
        previous_b, previous_h = source_frame, target_frame


def aggregate():
    daily = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob('*__*/2026-*.parquet'))], ignore_index=True)
    keys = ['source', 'target', 'family', 'side', 'phase_ms', 'start_ms', 'end_ms']
    sums = daily.groupby(keys)[['n', 'nonzero', 'abs_sum', 'own_abs_sum', 'nochange_abs_sum', 'gain_sum', 'hits']].sum()
    sums['gain_vs_own_bp'] = sums.gain_sum/sums.n
    sums['model_mae_bp'] = sums.abs_sum/sums.n
    sums['own_mae_bp'] = sums.own_abs_sum/sums.n
    sums['gain_vs_nochange_bp'] = (sums.nochange_abs_sum-sums.abs_sum)/sums.n
    sums['hit_nonzero'] = sums.hits/sums.nonzero.replace(0, np.nan)
    sums['days'] = daily.groupby(keys).day.nunique()
    sums['improved_days'] = daily.groupby(keys).gain_sum.apply(lambda a: int((a > 0).sum()))
    rng = np.random.default_rng(20261108)
    pick = rng.integers(0, 17, size=(2000, 17))
    weights = np.stack([np.bincount(a, minlength=17) for a in pick])
    rows = sums.index
    g = daily.pivot(index=keys, columns='day', values='gain_sum').reindex(rows).reindex(columns=E.DAYS[1:]).fillna(0).to_numpy()
    n = daily.pivot(index=keys, columns='day', values='n').reindex(rows).reindex(columns=E.DAYS[1:]).fillna(0).to_numpy()
    den = n @ weights.T
    means = np.divide(g @ weights.T, den, out=np.full(den.shape, np.nan), where=den > 0)
    center = sums.gain_vs_own_bp.to_numpy()
    sd = np.nanstd(means, axis=1, ddof=1)
    lo, hi = np.nanquantile(means, [.025, .975], axis=1)
    sums['ci_low'], sums['ci_high'] = lo, hi
    reset = sums.reset_index()
    critical = np.zeros(len(reset))
    for _, block in reset.groupby(['source', 'target', 'phase_ms']):
        ids = block.index.to_numpy()
        max_t = np.nanmax(np.abs((means[ids]-center[ids, None])/np.maximum(sd[ids, None], 1e-15)), axis=0)
        critical[ids] = np.nanquantile(max_t, .95)
    reset['simul_low'] = center-critical*sd
    reset['simul_high'] = center+critical*sd
    reset.to_csv(OUT/'summary.csv', index=False)
    daily.to_csv(OUT/'daily.csv.gz', index=False, compression='gzip')
    metadata = dict(rows=len(reset), daily_rows=len(daily), windows_ms=list(WINDOWS),
                    phases_ms=[0, 300], reused_coefficients='each decision day prior-day fitted delay-zero mapping',
                    frozen_information='all BN and HL inputs held at original decision; no refreshed HL input',
                    target='native best-quote movement after start_ms until end_ms, normalized by received current same-side quote',
                    purpose='detect apparent signals exhausted before hypothetical order activation; not fills or PnL',
                    interval='whole-date block bootstrap; simultaneous family/side/window band within pair and phase')
    (OUT/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('DECISION_GUARD_SUMMARY', len(reset), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for source, target in PAIRS:
        run_pair(source, target)
    aggregate()


if __name__ == '__main__':
    main()
