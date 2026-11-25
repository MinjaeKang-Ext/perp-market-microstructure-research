"""Descriptive cross-market best-quote reaction curves, not forecast results.

Source states are sampled every second from a 100ms grid. For lag L, the target
    is its best quote change over [t+L, t+L+100ms]. CSV lag_ms is this window's
start offset; end_lag_ms=L+100ms aligns both source/target return-bin endpoints.
Native-clock curves are offline diagnostics and
do not establish synchronized clocks, causality or executable opportunities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'out_bn_hl_transfer/inputs'
OUT = ROOT / 'out_bn_hl_transfer/response'
BN = ('binance_SKHYNIXUSDT', 'binance_SKHYUSDT')
HL = ('hyperliquid_SKHX', 'hyperliquid_SKHY')
PAIRS = tuple((a, b) for a in BN for b in HL) + tuple((b, a) for a in BN for b in HL)
DAYS = tuple(pd.date_range('2026-10-15', '2026-11-01').strftime('%Y-%m-%d'))
LAGS = np.arange(-50, 51)
MOMENTS = ('n', 'sx', 'sy', 'sxx', 'syy', 'sxy', 'sz', 'szz', 'sxz', 'syz',
           'source_nonzero', 'target_nonzero', 'signed_response_sum')
VERSION = 1


def load(path):
    with np.load(path, allow_pickle=False) as file:
        return {k: file[k] for k in file.files if k != 'metadata_json'}


def changes(price):
    price = np.asarray(price, float)
    result = np.full(len(price), np.nan)
    result[1:] = 10000 * (price[1:] / price[:-1] - 1)
    return result


def features(g, clock):
    if clock == 'native':
        return {'mid': changes(g['truth_mid'])}
    micro = g['mid'] * (1 + np.asarray(g['weighted_offset_bp'], float) / 10000)
    obi = np.asarray(g['imbalance'], float)
    delta_obi = np.r_[np.nan, np.diff(obi)]
    return {'mid': changes(g['mid']), 'micro_proxy': changes(micro),
            'obi_change': delta_obi, 'ofi100': np.asarray(g['ofi100_norm'], float)}


def verify_feature_cohorts():
    """Check that scalar and batch feature-finite cohorts coincide in data."""
    records = []
    names = ('mid', 'weighted_offset_bp', 'imbalance', 'ofi100_norm', 'age_sec')
    for market in BN + HL:
        for day in DAYS:
            with np.load(INPUT / market / (day + '.npz'), allow_pickle=False) as file:
                g = {name: file[name] for name in names}
            age = g['age_sec']
            valid = ((age >= 0) & (age <= 1) & (np.roll(age, 1) >= 0) & (np.roll(age, 1) <= 1))
            valid[0] = False
            invalid = {name: int(np.sum(valid & ~np.isfinite(values)))
                       for name, values in features(g, 'receive').items()}
            records.append(dict(market=market, day=day, valid_source_rows=int(valid.sum()), invalid=invalid))
    total_invalid = sum(sum(row['invalid'].values()) for row in records)
    payload = dict(status='pass' if total_invalid == 0 else 'fail',
                   market_days=len(records), valid_source_rows=sum(row['valid_source_rows'] for row in records),
                   invalid_feature_entries=total_invalid, rows=records)
    (OUT / 'source_finite_verification.json').write_text(json.dumps(payload, indent=2))
    if total_invalid:
        raise AssertionError('Batch feature-finite cohort differs from scalar cohort')
    return payload


def totals(x, y, z, mask):
    x, y, z = x[mask], y[mask], z[mask]
    return np.array([len(x), x.sum(), y.sum(), x @ x, y @ y, x @ y,
                     z.sum(), z @ z, x @ z, y @ z,
                     np.count_nonzero(x), np.count_nonzero(y),
                     np.sum(np.sign(x) * y)], float)


def matrix_totals(x, y, z, mask):
    """Identical moments for multiple source features in one memory pass."""
    x, y, z = x[mask], y[mask], z[mask]
    out = np.empty((x.shape[1], len(MOMENTS)), float)
    out[:, 0] = len(y)
    out[:, 1] = x.sum(axis=0)
    out[:, 2] = y.sum()
    out[:, 3] = np.einsum('ij,ij->j', x, x)
    out[:, 4] = y @ y
    out[:, 5] = x.T @ y
    out[:, 6] = z.sum()
    out[:, 7] = z @ z
    out[:, 8] = x.T @ z
    out[:, 9] = y @ z
    out[:, 10] = np.count_nonzero(x, axis=0)
    out[:, 11] = np.count_nonzero(y)
    out[:, 12] = (np.sign(x) * y[:, None]).sum(axis=0)
    return out


def statistics(t):
    """Accept an arbitrary leading shape with sufficient moments last."""
    n = t[..., 0]
    safe = np.maximum(n, 1)
    vx = t[..., 3] - t[..., 1] ** 2 / safe
    vy = t[..., 4] - t[..., 2] ** 2 / safe
    vz = t[..., 7] - t[..., 6] ** 2 / safe
    xy = t[..., 5] - t[..., 1] * t[..., 2] / safe
    xz = t[..., 8] - t[..., 1] * t[..., 6] / safe
    yz = t[..., 9] - t[..., 2] * t[..., 6] / safe
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b > 1e-20)
    corr = divide(xy, np.sqrt(np.maximum(0, vx * vy)))
    slope = divide(xy, vx)
    vcondx = vx - np.nan_to_num(divide(xz ** 2, vz))
    vcondy = vy - np.nan_to_num(divide(yz ** 2, vz))
    condxy = xy - np.nan_to_num(divide(xz * yz, vz))
    partial = divide(condxy, np.sqrt(np.maximum(0, vcondx * vcondy)))
    identifiable = ((vcondx > 1e-10 * np.maximum(vx, 1e-20))
                    & (vcondy > 1e-10 * np.maximum(vy, 1e-20)))
    partial = np.where(identifiable, partial, np.nan)
    response = divide(t[..., 12], t[..., 10])
    return dict(correlation=corr, slope=slope, partial_correlation=partial,
                signed_response_bp=response,
                source_nonzero_fraction=divide(t[..., 10], safe),
                target_change_fraction=divide(t[..., 11], safe))


def curve(source, target, clock='receive', phase=0):
    n = len(source['mid'])
    # No day-boundary label or source-window ambiguity: omit five seconds at
    # each edge rather than borrowing future/past date data inconsistently.
    ix = np.arange(50 + phase, n - 51, 10)
    source_age = source['age_sec'] if clock == 'receive' else source['truth_age_sec']
    target_age = target['age_sec'] if clock == 'receive' else target['truth_age_sec']
    src_valid = ((source_age[ix] >= 0) & (source_age[ix] <= 1)
                 & (source_age[ix-1] >= 0) & (source_age[ix-1] <= 1))
    prefix = '' if clock == 'receive' else 'truth_'
    control = changes(target[prefix + 'mid'])[ix]
    # Control is the target's own immediately preceding 100ms midpoint move.
    control_valid = ((target_age[ix] >= 0) & (target_age[ix] <= 1)
                     & (target_age[ix-1] >= 0) & (target_age[ix-1] <= 1))
    seconds = ix / 10
    regular = (seconds >= 9*3600) & (seconds < 15*3600+20*60)
    rows = []
    source_features = features(source, clock)
    names = list(source_features)
    x = np.column_stack([source_features[name][ix] for name in names])
    # Valid BBO samples make every source feature finite. Use one explicitly
    # common cohort across feature families instead of repeated array copies.
    feature_valid = np.isfinite(x).all(axis=1)
    for side in ('bid', 'ask'):
        p = target[prefix + side]
        for lag in LAGS:
            j = ix + lag
            y = 10000 * (p[j+1] / p[j] - 1)
            valid = (src_valid & control_valid & feature_valid & np.isfinite(y)
                     & np.isfinite(control) & (target_age[j] >= 0)
                     & (target_age[j] <= 1) & (target_age[j+1] >= 0)
                     & (target_age[j+1] <= 1))
            if phase:
                sessions = [('all', matrix_totals(x, y, control, valid))]
            else:
                normal = matrix_totals(x, y, control, valid & regular)
                other = matrix_totals(x, y, control, valid & ~regular)
                sessions = [('all', normal+other), ('09:00-15:20_KST', normal), ('other_hours', other)]
            for label, moments in sessions:
                for f, feature in enumerate(names):
                    row = dict(clock=clock, feature=feature, side=side,
                               lag_ms=int(lag * 100), session=label, phase_ms=phase*100)
                    row.update(zip(MOMENTS, moments[f]))
                    rows.append(row)
    return rows


def write_states(day, grids):
    state_rows = []
    def price_changes(values):
        return int(np.sum(np.isfinite(values[1:]) & np.isfinite(values[:-1])
                          & (np.diff(values) != 0)))
    for market, grid in grids.items():
        receipt = grid['recv_stamp_us']
        native = grid['truth_event_us']
        one_second = receipt[::10]
        state_rows.append(dict(
            market=market, day=day, grid_rows=len(receipt),
            receive_updates_100ms=int(np.sum((np.diff(receipt) != 0) & (receipt[1:] >= 0))),
            native_updates_100ms=int(np.sum((np.diff(native) != 0) & (native[1:] >= 0))),
            receive_distinct_states_1s=int(np.sum((np.diff(one_second) != 0) & (one_second[1:] >= 0))),
            midpoint_changes_100ms=price_changes(grid['mid']),
            bid_changes_100ms=price_changes(grid['bid']),
            ask_changes_100ms=price_changes(grid['ask'])))
    (OUT / 'states').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state_rows).to_csv(OUT / 'states' / (day + '.csv'), index=False)


def process_day(day):
    path = OUT / 'daily' / (day + '.csv')
    if path.exists():
        if not (OUT / 'states' / (day + '.csv')).exists():
            write_states(day, {market: load(INPUT / market / (day + '.npz')) for market in BN + HL})
        print('SKIP_RESPONSE', day, flush=True)
        return
    deadline = time.monotonic() + 3600
    while not all((INPUT / market / (day + '.npz')).exists() for market in BN + HL):
        if time.monotonic() > deadline:
            raise FileNotFoundError(f'Inputs absent for {day}')
        print('WAIT_RESPONSE', day, flush=True)
        time.sleep(5)
    start = time.monotonic()
    grids = {market: load(INPUT / market / (day + '.npz')) for market in BN + HL}
    write_states(day, grids)
    records = []
    for source, target in PAIRS:
        s, t = grids[source], grids[target]
        if not np.array_equal(s['grid_us'], t['grid_us']):
            raise AssertionError('Unaligned source and target grid')
        for clock in ('receive', 'native'):
            phases = (0, 3) if {source, target} == {BN[0], HL[0]} else (0,)
            for phase in phases:
                for row in curve(s, t, clock, phase):
                    row.update(source=source, target=target, day=day,
                               matching_underlying=((source.endswith('SKHX') or source.endswith('SKHYNIXUSDT'))
                                                    == (target.endswith('SKHX') or target.endswith('SKHYNIXUSDT'))))
                    records.append(row)
    frame = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.part')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    print('RESPONSE_DAY', day, len(frame), 'seconds', round(time.monotonic()-start, 2), flush=True)


def aggregate(bootstrap=1000):
    files = sorted((OUT / 'daily').glob('*.csv'))
    daily = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    keys = ['source', 'target', 'clock', 'feature', 'side', 'session', 'phase_ms', 'lag_ms']
    if sorted(daily.day.unique()) != list(DAYS):
        raise AssertionError('All 18 dates must be complete before final aggregation')
    if daily.duplicated(['day'] + keys).any():
        raise AssertionError('Duplicate response-moment keys')
    canonical_index = None
    for day, block in daily.groupby('day', sort=True):
        if len(block) != 26260:
            raise AssertionError(f'Partial moment file: {day} rows={len(block)}')
        if block.groupby('phase_ms').size().to_dict() != {0: 24240, 300: 2020}:
            raise AssertionError(f'Wrong phase cohorts: {day}')
        index = pd.MultiIndex.from_frame(block[keys]).sort_values()
        if canonical_index is None:
            canonical_index = index
        elif not index.equals(canonical_index):
            raise AssertionError(f'Inconsistent source/target/clock/lag groups: {day}')
        counts = block.groupby(keys[:-1]).size()
        if not counts.eq(101).all() or sorted(block.lag_ms.unique()) != list(LAGS * 100):
            raise AssertionError(f'Missing lag bins: {day}')
        if not ((block.source_nonzero <= block.n) & (block.target_nonzero <= block.n) & (block.n >= 0)).all():
            raise AssertionError(f'Invalid episode/sample counts: {day}')
    daily.to_csv(OUT / 'daily.csv', index=False)
    state_files = sorted((OUT / 'states').glob('*.csv'))
    if state_files:
        pd.concat((pd.read_csv(path) for path in state_files), ignore_index=True).to_csv(OUT / 'state_episodes.csv', index=False)
    group_keys = ['source', 'target', 'clock', 'feature', 'side', 'session', 'phase_ms']
    records = []
    daily_peak_values = {}
    rng = np.random.default_rng(20261108)
    for key, group in daily.groupby(group_keys, sort=True):
        days = sorted(group.day.unique())
        order = pd.MultiIndex.from_product([days, sorted(group.lag_ms.unique())], names=['day', 'lag_ms'])
        matrix = group.set_index(['day', 'lag_ms']).reindex(order)[list(MOMENTS)].to_numpy().reshape(len(days), -1, len(MOMENTS))
        total = np.nansum(matrix, axis=0)
        point = statistics(total)
        picks = rng.multinomial(len(days), np.ones(len(days))/len(days), bootstrap)
        boot_total = (picks @ np.nan_to_num(matrix).reshape(len(days), -1)).reshape(bootstrap, -1, len(MOMENTS))
        boot = statistics(boot_total)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ci = {name: np.nanquantile(boot[name], [.025, .975], axis=0)
                  for name in ('correlation', 'partial_correlation', 'signed_response_bp')}
        day_point = statistics(matrix)
        day_peaks = []
        group_lags = sorted(group.lag_ms.unique())
        for values in day_point['correlation']:
            if np.isfinite(values).any():
                day_peaks.append(int(group_lags[np.nanargmax(np.abs(values))]))
        daily_peak_values[key] = day_peaks
        for j, lag in enumerate(sorted(group.lag_ms.unique())):
            row = dict(zip(group_keys, key))
            row.update(lag_ms=int(lag), end_lag_ms=int(lag)+100, days=len(days), n=int(total[j, 0]),
                       source_nonzero=int(total[j, 10]), target_nonzero=int(total[j, 11]))
            for name, values in point.items():
                row[name] = float(values[j])
            for name, limits in ci.items():
                row[name + '_ci_low'] = float(limits[0, j])
                row[name + '_ci_high'] = float(limits[1, j])
            row['days_positive_correlation'] = int(np.nansum(day_point['correlation'][:, j] > 0))
            records.append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / 'summary.csv', index=False)
    peaks = []
    for key, group in summary.groupby(group_keys, sort=True):
        row = dict(zip(group_keys, key))
        for label, valid in (('all', group), ('nonnegative', group[group.end_lag_ms >= 0]), ('negative', group[group.end_lag_ms < 0])):
            if valid.correlation.notna().any():
                best = valid.loc[valid.correlation.abs().idxmax()]
                row[label + '_abs_peak_lag_ms'] = int(best.lag_ms)
                row[label + '_abs_peak_end_lag_ms'] = int(best.end_lag_ms)
                row[label + '_abs_peak_correlation'] = float(best.correlation)
                row[label + '_peak_partial_correlation'] = float(best.partial_correlation)
        day_peaks = daily_peak_values[key]
        row['daily_abs_peak_lag_median_ms'] = float(np.median(day_peaks)) if day_peaks else np.nan
        row['daily_abs_peak_end_lag_median_ms'] = float(np.median(day_peaks)+100) if day_peaks else np.nan
        row['daily_abs_peak_lag_min_ms'] = min(day_peaks) if day_peaks else np.nan
        row['daily_abs_peak_lag_max_ms'] = max(day_peaks) if day_peaks else np.nan
        peaks.append(row)
    peak_frame = pd.DataFrame(peaks)
    peak_frame.to_csv(OUT / 'peaks.csv', index=False)
    matched_forward = (((peak_frame.source == BN[0]) & (peak_frame.target == HL[0]))
                       | ((peak_frame.source == BN[1]) & (peak_frame.target == HL[1])))
    matched_reverse = (((peak_frame.source == HL[0]) & (peak_frame.target == BN[0]))
                       | ((peak_frame.source == HL[1]) & (peak_frame.target == BN[1])))
    primary = (peak_frame.session == 'all') & (peak_frame.phase_ms == 0)
    peak_frame[matched_forward & primary].to_csv(OUT / 'headline_matched.csv', index=False)
    peak_frame[matched_reverse & primary].to_csv(OUT / 'headline_reverse.csv', index=False)
    metadata = dict(version=VERSION, dates=sorted(daily.day.unique()),
                    input_grid_ms=100, source_sample_period_ms=1000,
                    primary_phase_ms=0, alternate_phase_ms=300,
                    lag_definition='lag_ms is target-window START: [t+lag,t+lag+100ms], compared with source [t-100ms,t].',
                    end_lag_definition='end_lag_ms=lag_ms+100; aligns return-bin endpoints, so a physical 300ms delayed copy peaks at end_lag=300ms.',
                    start_lag_range_ms=[-5000, 5000], end_aligned_lag_range_ms=[-4900, 5100],
                    source_features_receive=['mid return100ms', 'weighted-mid proxy return100ms', 'OBI change100ms', 'OFI100ms normalized'],
                    source_features_native=['mid return100ms only; native quantity unavailable'],
                    freshness_max_seconds=1, bootstrap_dates=bootstrap,
                    latest_measured_reference_ms=dict(HL_order_ack=440, HL_cancel_ack=750,
                                                      BN_order_assumption=100, BN_cancel_assumption=100,
                                                      BN_feed=5, HL_feed=240),
                    feed_delay_added_to_received_at=False,
                    n_is_not_independent_sample_count=True,
                    limitations=['Descriptive full-period curves, not out-of-sample forecasts.',
                                 'Pointwise bootstrap intervals do not correct selection of maximal lag.',
                                 'Overlapping returns and repeated states are clustered by whole date.',
                                 'Collection latency asymmetry and unsynchronized clocks can mimic a market lead.',
                                 'A reaction peak does not measure maximum usable execution delay.'],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    definition_parity=json.loads((OUT / 'batch_parity.json').read_text()),
                    daily_rows=len(daily), summary_rows=len(summary), peak_rows=len(peaks))
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (OUT / 'integrity.json').write_text(json.dumps(dict(
        status='pass', dates=len(DAYS), daily_rows=len(daily), rows_per_day=26260,
        unique_moment_keys=True, lag_bins_per_group=101,
        phase_rows_per_day={'0': 24240, '300': 2020},
        identical_profile_keys_across_dates=True,
        source_sha256=metadata['source_sha256'],
        definition_parity=metadata['definition_parity']), indent=2))
    return summary


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, (source, target) in enumerate(zip(BN, HL)):
        for row, side in enumerate(('bid', 'ask')):
            ax = axes[row, column]
            subset = summary[(summary.source == source) & (summary.target == target)
                             & (summary.side == side) & (summary.session == 'all')
                             & (summary.phase_ms == 0)]
            for name in ('mid', 'micro_proxy', 'obi_change', 'ofi100'):
                block = subset[(subset.clock == 'receive') & (subset.feature == name)].sort_values('lag_ms')
                ax.plot(block.end_lag_ms, block.correlation, label=name)
            native = subset[(subset.clock == 'native') & (subset.feature == 'mid')].sort_values('lag_ms')
            ax.plot(native.end_lag_ms, native.correlation, '--', color='black', label='native mid')
            ax.axvline(0, color='grey', linewidth=.8)
            ax.axhline(0, color='grey', linewidth=.8)
            ax.axvspan(400, 500, color='grey', alpha=.12)
            ax.set_title(f'{source.split("_")[-1]} -> {target.split("_")[-1]} best {side}')
            ax.set_xlabel('End-aligned target return lag (ms)')
            ax.set_ylabel('Correlation')
            ax.grid(alpha=.2)
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Descriptive quote-response curves | 1s source samples, 100ms target changes')
    fig.tight_layout()
    fig.savefig(OUT / 'response_curves.png', dpi=170)
    plt.close(fig)


class ResponseTests(unittest.TestCase):
    def test_positive_lag_and_reverse_sign(self):
        rng = np.random.default_rng(7)
        source_returns = rng.normal(size=3000)
        target_returns = np.r_[np.zeros(3), source_returns[:-3]]
        correlations = []
        for lag in range(-10, 11):
            ix = np.arange(11, 2989)
            correlations.append(np.corrcoef(source_returns[ix], target_returns[ix+lag])[0, 1])
        self.assertEqual(np.argmax(correlations)-10, 3)
        reverse = [np.corrcoef(target_returns[ix], source_returns[ix+lag])[0, 1] for lag in range(-10, 11)]
        self.assertEqual(np.argmax(reverse)-10, -3)

    def test_partial_control_removes_shared_component(self):
        rng = np.random.default_rng(9)
        z = rng.normal(size=20000)
        x, y = z + rng.normal(size=len(z)), 2*z + rng.normal(size=len(z))
        result = statistics(totals(x, y, z, np.ones(len(z), bool)))
        self.assertGreater(float(result['correlation']), .6)
        self.assertLess(abs(float(result['partial_correlation'])), .03)

    def test_moment_statistics_match_direct(self):
        rng = np.random.default_rng(4)
        x, y, z = rng.normal(size=(3, 1000))
        result = statistics(totals(x, y, z, np.ones(len(z), bool)))
        self.assertAlmostEqual(float(result['correlation']), np.corrcoef(x, y)[0, 1])
        self.assertAlmostEqual(float(result['slope']), np.cov(x, y, ddof=0)[0, 1] / x.var())

    def test_partial_is_undefined_when_control_is_target(self):
        rng = np.random.default_rng(12)
        x, y = rng.normal(size=(2, 1000))
        result = statistics(totals(x, y, y, np.ones(len(y), bool)))
        self.assertTrue(np.isnan(result['partial_correlation']))

    def test_complete_price_path_end_alignment(self):
        rng = np.random.default_rng(17)
        price = 100 + np.cumsum(rng.normal(scale=.001, size=3000))
        delayed = np.r_[np.repeat(price[0], 3), price[:-3]]
        def grid(p):
            return dict(mid=p, truth_mid=p, bid=p-.01, ask=p+.01,
                        truth_bid=p-.01, truth_ask=p+.01,
                        age_sec=np.zeros(len(p)), truth_age_sec=np.zeros(len(p)))
        def peak(a, b):
            rows = [r for r in curve(grid(a), grid(b), 'native')
                    if r['feature']=='mid' and r['side']=='bid' and r['session']=='all']
            stat = statistics(np.array([[r[name] for name in MOMENTS] for r in rows]))
            return rows[int(np.nanargmax(stat['correlation']))]['lag_ms'] + 100
        self.assertEqual(peak(price, delayed), 300)
        self.assertEqual(peak(delayed, price), -300)

    def test_matrix_moments_equal_scalar_features(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(3000, 4))
        y, z = rng.normal(size=(2, 3000))
        mask = rng.random(3000) > .2
        fast = matrix_totals(x, y, z, mask)
        slow = np.array([totals(x[:, i], y, z, mask) for i in range(4)])
        np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--day')
    parser.add_argument('--from-day')
    parser.add_argument('--verify-features', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify_features:
        result = verify_feature_cohorts()
        print('SOURCE_FEATURE_COHORTS', result['status'], result['market_days'], result['invalid_feature_entries'])
        return
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ResponseTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        (OUT / 'tests.json').write_text(json.dumps(dict(tests=result.testsRun, successful=result.wasSuccessful())))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    if not args.aggregate_only:
        days = (args.day,) if args.day else tuple(day for day in DAYS if not args.from_day or day >= args.from_day)
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(process_day, days))
        else:
            for day in days:
                process_day(day)
    summary = aggregate()
    plots(summary)
    print('RESPONSE_DONE', len(summary), flush=True)


if __name__ == '__main__':
    main()
