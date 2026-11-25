"""Exploratory ranking of cross-market price forecasts by residual signal life.

An isolated significant old lag does not establish a usable signal lifetime.
Both bid/ask and both physical-innovation/endpoint diagnostics must pass along
the whole observed delay prefix. This is research screening, not live approval.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out_bn_hl_transfer'
REQUIRED = ('native_innovation', 'native_endpoint')


def contiguous_life(delays, lower):
    """Return NaN if delay zero fails; stop at the first failed grid point."""
    order = np.argsort(delays)
    d, lo = np.asarray(delays)[order], np.asarray(lower)[order]
    if not len(d) or d[0] != 0:
        return np.nan
    last = np.nan
    for delay, bound in zip(d, lo):
        if not np.isfinite(bound) or bound <= 0:
            break
        last = float(delay)
    return last


def joint_curve(block, clocks=REQUIRED):
    records = []
    for delay, group in block.groupby('delay_ms', sort=True):
        selected = group[group.clock.isin(clocks)]
        combinations = set(zip(selected.clock, selected.side))
        expected = {(clock, side) for clock in clocks for side in ('bid', 'ask')}
        if combinations != expected or len(selected) != len(expected):
            records.append(dict(delay_ms=delay, min_gain_bp=np.nan, lower_bp=np.nan,
                                upper_bp=np.nan, min_improved_days=0, n_min=0,
                                complete=False))
            continue
        records.append(dict(delay_ms=int(delay), min_gain_bp=float(selected.gain_vs_own_bp.min()),
                            lower_bp=float(selected.gain_vs_own_simul_low.min()),
                            upper_bp=float(selected.gain_vs_own_simul_high.min()),
                            min_improved_days=int(selected.improved_days_vs_own.min()),
                            n_min=int(selected.n.min()), complete=True))
    return pd.DataFrame(records)


def rank(summary):
    keys = ['source', 'target', 'pair_type', 'family', 'horizon_ms', 'max_age', 'mode']
    all_curves, records = [], []
    for key, block in summary.groupby(keys, sort=True):
        label = dict(zip(keys, key))
        curve = joint_curve(block)
        receive = joint_curve(block, ('receive',))
        life = contiguous_life(curve.delay_ms.to_numpy(), curve.lower_bp.to_numpy())
        recv_life = contiguous_life(receive.delay_ms.to_numpy(), receive.lower_bp.to_numpy())
        passing = curve[curve.lower_bp > 0]
        record = dict(label, contiguous_life_ms=life,
                      receive_contiguous_life_ms=recv_life,
                      longest_isolated_pass_ms=float(passing.delay_ms.max()) if len(passing) else np.nan,
                      right_censored=bool(np.isfinite(life) and life == curve.delay_ms.max()),
                      native_screen_pass=bool(np.isfinite(life)),
                      status='native_both_sides_prefix_pass' if np.isfinite(life)
                      else ('receive_only_or_feed_catchup' if np.isfinite(recv_life) else 'no_qualified_delay_prefix'))
        for anchor in (0, 400, 500, 700, 800, 1000, 2000, 5000):
            row = curve[curve.delay_ms == anchor]
            suffix = str(anchor)
            for name, column in (('gain', 'min_gain_bp'), ('lower', 'lower_bp'),
                                 ('days', 'min_improved_days'), ('n', 'n_min')):
                record[f'{name}_{suffix}'] = float(row.iloc[0][column]) if len(row) else np.nan
        records.append(record)
        all_curves.append(curve.assign(**label))
    frame = pd.DataFrame(records)
    frame['_life'] = frame.contiguous_life_ms.fillna(-1)
    # Prespecified execution reference bracket; no weights fitted to outcomes.
    frame = frame.sort_values(['source', 'target', 'horizon_ms', 'max_age', 'mode',
                               '_life', 'lower_800', 'lower_500', 'days_500', 'gain_500'],
                              ascending=[True, True, True, True, True, False, False, False, False, False])
    frame['diagnostic_rank'] = frame.groupby(['source', 'target', 'horizon_ms', 'max_age', 'mode']).cumcount() + 1
    frame = frame.drop(columns='_life')
    return frame, pd.concat(all_curves, ignore_index=True)


def common_short_intervals():
    """Matched 0..1s source-age support prevents delay-specific sample shifts."""
    daily = pd.read_parquet(OUT/'eval/daily.parquet')
    block = daily[(daily['mode'] == 'frozen') & (daily.max_age == 1)
                  & daily.horizon_ms.isin([1000, 5000]) & (daily.delay_ms <= 1000)
                  & daily.clock.isin(REQUIRED) & (daily.pair_type == 'matched')].copy()
    keys = ['source', 'target', 'family', 'horizon_ms', 'max_age', 'mode', 'clock', 'side', 'delay_ms']
    dates = sorted(block.day.unique())
    gains = block.pivot(index=keys, columns='day', values='common_short_gain_sum').reindex(columns=dates).fillna(0)
    counts = block.pivot(index=keys, columns='day', values='common_short_n').reindex(gains.index).reindex(columns=dates).fillna(0)
    # For a given target/day/clock/horizon/side, every family/lag must have the
    # exact same common-support count, not merely a similar coverage fraction.
    count_keys = ['source', 'target', 'day', 'clock', 'horizon_ms', 'side']
    if (block.groupby(count_keys).common_short_n.nunique() > 1).any():
        raise AssertionError('Short-delay matched sample differs across families/lags')
    rng = np.random.default_rng(20261108)
    pick = rng.integers(0, len(dates), size=(2000, len(dates)))
    weights = np.stack([np.bincount(row, minlength=len(dates)) for row in pick])
    den = counts.to_numpy() @ weights.T
    means = np.divide(gains.to_numpy() @ weights.T, den,
                      out=np.full(den.shape, np.nan), where=den > 0)
    n = counts.sum(axis=1).to_numpy()
    point = np.divide(gains.sum(axis=1).to_numpy(), n, out=np.full(len(n), np.nan), where=n > 0)
    sd = np.nanstd(means, axis=1, ddof=1)
    result = gains.index.to_frame(index=False)
    critical = np.zeros(len(result))
    for _, group in result.groupby(['source', 'target', 'clock', 'horizon_ms']):
        ids = group.index.to_numpy()
        max_t = np.nanmax(np.abs((means[ids]-point[ids, None])/np.maximum(sd[ids, None], 1e-15)), axis=0)
        critical[ids] = np.nanquantile(max_t, .95)
    result['common_gain_bp'] = point
    result['common_simul_low'] = point-critical*sd
    result['common_simul_high'] = point+critical*sd
    result['common_n'] = n
    result.to_csv(OUT/'ranking/short_delay_common_support.csv', index=False)
    return result


def family_contrasts():
    columns = ['source', 'target', 'pair_type', 'clock', 'delay_ms', 'horizon_ms',
               'max_age', 'mode', 'family', 'side', 'day', 'n', 'abs_sum']
    daily = pd.read_parquet(OUT/'eval/daily.parquet', columns=columns)
    use = daily[(daily.pair_type == 'matched') & (daily.max_age == 1)
                & (daily['mode'] == 'frozen') & daily.horizon_ms.isin([1000, 5000])
                & daily.delay_ms.isin([0, 100, 500, 800])]
    keys = ['source', 'target', 'clock', 'delay_ms', 'horizon_ms', 'side', 'day']
    errors = use.pivot(index=keys, columns='family', values='abs_sum')
    counts = use.groupby(keys).n.agg(['first', 'nunique'])
    if (counts['nunique'] != 1).any():
        raise AssertionError('Family pair comparisons lack identical evaluation samples')
    candidates = []
    for left, right in combinations(sorted(errors.columns), 2):
        result = errors.index.to_frame(index=False)
        result['left'], result['right'] = left, right
        result['gain_sum'] = (errors[right]-errors[left]).to_numpy()
        result['n'] = counts['first'].reindex(errors.index).to_numpy()
        candidates.append(result)
    data = pd.concat(candidates, ignore_index=True)
    grouping = ['source', 'target', 'clock', 'delay_ms', 'horizon_ms', 'side', 'left', 'right']
    dates = sorted(data.day.unique())
    gain = data.pivot(index=grouping, columns='day', values='gain_sum').reindex(columns=dates).fillna(0)
    n = data.pivot(index=grouping, columns='day', values='n').reindex(gain.index).reindex(columns=dates).fillna(0)
    rng = np.random.default_rng(20261108)
    picks = rng.integers(0, len(dates), size=(2000, len(dates)))
    weights = np.stack([np.bincount(row, minlength=len(dates)) for row in picks])
    den = n.to_numpy() @ weights.T
    resamples = np.divide(gain.to_numpy() @ weights.T, den,
                          out=np.full(den.shape, np.nan), where=den > 0)
    point = gain.sum(axis=1).to_numpy()/n.sum(axis=1).to_numpy()
    sd = np.nanstd(resamples, axis=1, ddof=1)
    result = gain.index.to_frame(index=False)
    critical = np.zeros(len(result))
    for _, block in result.groupby(['source', 'target', 'clock', 'horizon_ms']):
        ids = block.index.to_numpy()
        max_t = np.nanmax(np.abs((resamples[ids]-point[ids, None])/np.maximum(sd[ids, None], 1e-15)), axis=0)
        critical[ids] = np.nanquantile(max_t, .95)
    result['left_gain_vs_right_bp'] = point
    result['simul_low'] = point-critical*sd
    result['simul_high'] = point+critical*sd
    result['n'] = n.sum(axis=1).to_numpy()
    result.to_csv(OUT/'ranking/family_pairwise_contrasts.csv', index=False)
    return result


def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs = [('binance_SKHYNIXUSDT', 'hyperliquid_SKHX'),
             ('binance_SKHYUSDT', 'hyperliquid_SKHY')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = dict(mid='#2065b0', micro='#00a6a6', obi='#8c5ab5', ofi='#da7843',
                  joint='#2e8b57', mid_ofi='#8c5e3c')
    for ri, (source, target) in enumerate(pairs):
        for ci, horizon in enumerate((1000, 5000)):
            ax = axes[ri, ci]
            selected = curves[(curves.source == source) & (curves.target == target)
                              & (curves.horizon_ms == horizon) & (curves.max_age == 1)
                              & (curves['mode'] == 'frozen') & (curves.delay_ms <= 5000)]
            for family, block in selected.groupby('family'):
                block = block.sort_values('delay_ms')
                x = block.delay_ms.to_numpy() / 1000
                y = block.min_gain_bp.to_numpy()
                ax.plot(x, y, '.-', label=family, color=colors.get(family), lw=1.4)
            ax.axhline(0, color='black', lw=.8)
            ax.axvline(.44, color='#888888', ls=':', lw=1)
            ax.axvline(.75, color='#888888', ls='--', lw=1)
            ax.set_title(f"{target.replace('hyperliquid_', '')}: forward H={horizon/1000:g}s")
            ax.set_ylabel('Worst bid/ask and native-clock MAE gain (bp)')
            ax.grid(alpha=.2)
            if ri == 1:
                ax.set_xlabel('Age added to Binance information (s)')
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Residual cross-market predictability: fixed coefficients, aged Binance input')
    fig.text(.5, .008, 'Vertical references: HL ACK 440ms and cancellation ACK 750ms; ACK is not measured one-way activation.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .95))
    folder = OUT / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'residual_signal_life.png', dpi=160)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT / 'eval/summary.csv')
    folder = OUT / 'ranking'
    folder.mkdir(parents=True, exist_ok=True)
    common = common_short_intervals()
    family_contrasts()
    join_keys = ['source', 'target', 'family', 'horizon_ms', 'max_age', 'mode', 'clock', 'side', 'delay_ms']
    summary = summary.merge(common, on=join_keys, how='left', validate='one_to_one')
    supported = summary.common_simul_low.notna()
    summary.loc[supported, 'gain_vs_own_simul_low'] = np.minimum(
        summary.loc[supported, 'gain_vs_own_simul_low'], summary.loc[supported, 'common_simul_low'])
    ranks, curves = rank(summary)
    ranks.to_csv(folder / 'ranking.csv', index=False)
    curves.to_csv(folder / 'joint_clock_curves.csv', index=False)
    main_rows = ranks[(ranks.max_age == 1) & (ranks['mode'] == 'frozen')
                      & ranks.horizon_ms.isin([1000, 5000])]
    main_rows.to_csv(folder / 'primary_ranking.csv', index=False)
    metadata = dict(
        scope='Exploratory rank on reused historical dates; not a confirmed live trading selection',
        required_clocks=list(REQUIRED), required_sides=['bid', 'ask'],
        primary_method='daily past-only coefficients fitted at delay 0, Binance input aged, current HL controls retained',
        lifetime='largest consecutive tested delay from zero with all required simultaneous lower confidence bounds positive',
        tie_break=['800ms lower bound', '500ms lower bound', '500ms positive days', '500ms worst-side mean gain'],
        no_prefix_rule='report no qualified signal; numeric diagnostic rank does not imply usability',
        right_censor_rule='positive at final tested delay is a lower bound on lifetime, not an estimated maximum',
        latencies_ms=dict(hl_order_ack=440, hl_cancel_ack=750, bn_order_assumption=100,
                          bn_cancel_assumption=100, bn_data_call=5, hl_data_call=240),
        interpretation='Added signal age, forward forecast horizon, native propagation lag and order/cancellation ACK are distinct quantities',
        rows=len(ranks), native_prefix_pass_rows=int(ranks.native_screen_pass.sum()))
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    figures(curves)
    print('RANKED', len(ranks), 'curves', len(curves), flush=True)


if __name__ == '__main__':
    main()
