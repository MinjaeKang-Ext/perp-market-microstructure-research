"""Independent artifact/causality checks for BN -> HL transfer diagnostics.

Only verification.json is written. Numerical results and models are untouched.
The 100ms input manifest already records full canonical one-second parity and
272 direct raw as-of checks. This verifier inspects selected dates and compact
training checkpoints rather than rescanning all input fields for all dates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_transfer_eval as E
from work import bn_hl_transfer_decision_guard as D
from work import bn_hl_transfer_inputs as I

ROOT = E.ROOT
OUT = ROOT / 'out_bn_hl_transfer'
KEYS = ['source', 'target', 'pair_type', 'clock', 'delay_ms', 'horizon_ms',
        'max_age', 'mode', 'family', 'side']
SAMPLE_DAYS = ('2026-10-16', '2026-10-23', '2026-11-01')
MATCHED = ((E.BN[0], E.HL[0]), (E.BN[1], E.HL[1]))


def read_npz(path, fields=None):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in (fields if fields is not None else f.files)}


def assert_close(actual, expected, context, atol=1e-9, rtol=1e-10):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        error = float(np.nanmax(np.abs(np.asarray(actual)-np.asarray(expected))))
        raise AssertionError(f'{context}; max error {error}')
    difference = np.abs(np.asarray(actual)-np.asarray(expected))
    return float(np.nanmax(difference, initial=0))


def full_tables(allow_partial=False):
    daily = pd.read_parquet(E.OUT/'daily.parquet')
    summary = pd.read_csv(E.OUT/'summary.csv')
    if not allow_partial and (len(daily) != 582624 or len(summary) != 34272):
        raise AssertionError(f'Incomplete core grid {len(daily)}/{len(summary)}')
    if daily.duplicated(KEYS+['day']).any() or summary.duplicated(KEYS).any():
        raise AssertionError('Duplicate model/profile/date rows')
    grouped = daily.groupby(KEYS, sort=True)
    summed = grouped[[c for c in summary.columns if c in daily.columns
                      and c not in KEYS and c != 'weekend' and c != 'day']].sum()
    s = summary.set_index(KEYS).sort_index()
    if not summed.index.equals(s.index):
        raise AssertionError('Daily and summary profile indices differ')
    max_error = 0.
    for key in ('n', 'nonzero', 'abs_sum', 'sq_sum', 'hits', 'own_abs_sum',
                'own_sq_sum', 'own_hits', 'nochange_abs_sum', 'gain_sum'):
        max_error = max(max_error, assert_close(s[key], summed[key], 'daily aggregate '+key,
                                                atol=1e-7, rtol=1e-11))
    if not allow_partial and not grouped.day.nunique().eq(17).all():
        raise AssertionError('A core profile lacks 17 evaluation dates')
    for metric, expected in (
        ('gain_vs_own_bp', s.gain_sum/s.n),
        ('mae_bp', s.abs_sum/s.n),
        ('own_mae_bp', s.own_abs_sum/s.n),
        ('gain_vs_nochange_bp', (s.nochange_abs_sum-s.abs_sum)/s.n)):
        assert_close(s[metric], expected, 'summary identity '+metric)
    matched = [k for k in KEYS if k != 'family']+['day']
    for field in ('n', 'nonzero', 'own_abs_sum', 'own_sq_sum', 'own_hits', 'nochange_abs_sum'):
        spread = daily.groupby(matched)[field].agg(['min', 'max'])
        assert_close(spread['min'], spread['max'], 'family matched baseline '+field,
                     atol=1e-8, rtol=1e-11)
    d0 = daily[daily.delay_ms == 0]
    refit = d0[d0['mode']=='refit'].set_index([k for k in matched if k != 'mode']+['family']).sort_index()
    frozen = d0[d0['mode']=='frozen'].set_index([k for k in matched if k != 'mode']+['family']).sort_index()
    for field in ('n', 'abs_sum', 'sq_sum', 'hits', 'own_abs_sum', 'gain_sum'):
        assert_close(refit[field], frozen[field], 'delay zero frozen/refit '+field)
    return daily, summary, dict(daily_rows=len(daily), summary_rows=len(summary),
                                all_groups_17_dates=bool(grouped.day.nunique().eq(17).all()),
                                matched_family_baselines=True, delay0_frozen_refit_identical=True,
                                aggregate_max_error=max_error)


def checkpoint_validation():
    state_count, coefficient_count = 0, 0
    max_error = 0.
    for source, target in E.PAIRS:
        folder = E.OUT/(source+'__'+target)
        last_counts = {}
        previous = None
        for di, day in enumerate(E.DAYS):
            path = folder/(day+'.state.npz')
            if not path.exists():
                raise AssertionError('Missing checkpoint '+str(path))
            state = read_npz(path)
            state_count += 1
            for age in (1, 5):
                for delay in E.DELAYS_MS:
                    prefix = f'a{age}_d{delay}_'
                    n = int(state[prefix+'n'])
                    old = last_counts.get((age, delay), 0)
                    if not old <= n <= old+86400-30:
                        raise AssertionError('Training count violates chronological daily purge')
                    last_counts[age, delay] = n
            if di == 0:
                if (folder/(day+'.coefficients.npz')).exists():
                    raise AssertionError('First training day unexpectedly has OOS coefficients')
            else:
                stored = read_npz(folder/(day+'.coefficients.npz'))
                coefficient_count += 1
                # Check compact past-only state at selected dates and delays.
                if day in SAMPLE_DAYS:
                    for age in (1, 5):
                        for delay in (0, 500, 1000, 60000):
                            for mode in ('frozen', 'refit'):
                                old_delay = 0 if mode == 'frozen' else delay
                                prefix = f'a{age}_d{old_delay}_'
                                moment = E.Moments(*(previous[prefix+k].item() if k == 'n'
                                                     else previous[prefix+k] for k in ('n','sx','sy','xx','xy')))
                                fits = E.fit_families(moment)
                                ints = np.stack([fit[0] for fit in fits])
                                coef = np.stack([fit[1] for fit in fits])
                                key = f'a{age}_d{delay}_{mode}_'
                                max_error = max(max_error,
                                                assert_close(stored[key+'intercept'], ints,
                                                             'coefficient previous-day intercept', atol=1e-8, rtol=1e-9),
                                                assert_close(stored[key+'coefficient'], coef,
                                                             'coefficient previous-day slopes', atol=1e-8, rtol=1e-9))
            previous = state
    return dict(state_files=state_count, coefficient_files=coefficient_count,
                first_day_oos_coefficients_absent=True, training_n_monotone=True,
                selected_saved_coefficients_match_previous_state=True,
                prior_state_coefficient_max_error=max_error)


def sample_frames(source, target, day):
    day_index = E.DAYS.index(day)
    previous_day = E.DAYS[day_index-1]
    b, h = E.load_day(source, day), E.load_day(target, day)
    bp, hp = {}, {}
    for market, destination in ((source, bp), (target, hp)):
        with np.load(E.INPUT/market/(previous_day+'.npz'), allow_pickle=False) as f:
            for name in E.RAW:
                destination[name] = f[name][-E.TAIL:].astype(float)
            destination['_burn_mid'] = f['mid'][-216000:].astype(float)
    # Six hours is 72 half-lives. Unknown earlier initialization contributes
    # less than 2^-72 and is below machine precision at these log-price levels.
    burn, previous_basis = E.causal_basis(bp['_burn_mid'], hp['_burn_mid'])
    current_basis, _ = E.causal_basis(b['mid'], h['mid'], previous_basis)
    basis = np.r_[burn[-E.TAIL:], current_basis]
    return b, h, E.augmented(b, bp), E.augmented(h, hp), basis


def manual_features(sb, th, basis, base, delay=0):
    at, source_at = E.TAIL+base, E.TAIL+base-delay//100
    ret = lambda values, indices, back: 10000*(np.log(values[indices])-np.log(values[indices-back]))
    own = np.array([
        th['spread_bp'][at], th['imbalance'][at], th['weighted_offset_bp'][at],
        ret(th['mid'], at, 1), ret(th['mid'], at, 5), ret(th['mid'], at, 10),
        th['ofi100_norm'][at], th['ofi500_norm'][at], th['ofi1000_norm'][at]]).T
    wm = sb['mid'] + sb['mid']*sb['weighted_offset_bp']/10000
    target_mid = th['mid'][at]
    source = np.array([
        10000*(np.log(sb['mid'][source_at])-basis[source_at]-np.log(target_mid)),
        *[ret(sb['mid'], source_at, back) for back in (1,5,10,50)],
        10000*(np.log(wm[source_at])-basis[source_at]-np.log(target_mid)),
        *[ret(wm, source_at, back) for back in (1,5,10,50)],
        sb['imbalance'][source_at],
        *[sb['imbalance'][source_at]-sb['imbalance'][source_at-back] for back in (1,5,10)],
        sb['ofi100_norm'][source_at], sb['ofi500_norm'][source_at], sb['ofi1000_norm'][source_at]]).T
    return np.c_[own, source]


def manual_prediction(x, intercept, coefficient, selected):
    out = np.empty((len(x), len(intercept)*len(selected)))
    for fi in range(len(intercept)):
        for oj, column in enumerate(selected):
            value = np.full(len(x), intercept[fi, column])
            for feature in range(x.shape[1]):
                value += x[:, feature]*coefficient[fi, feature, column]
            out[:, fi*len(selected)+oj] = value
    return out


def sampled_predictions():
    rng = np.random.default_rng(20261108)
    names = [(clock,h,side) for clock in E.CLOCKS for h in E.HORIZONS_MS for side in ('bid','ask')]
    outputs = [j for j,key in enumerate(names) if key[0] == 'native_endpoint' and key[1] in (1000,5000)]
    feature_error = prediction_error = identity_error = mutation_error = 0.
    prediction_values, sampled_decisions = 0, 0
    raw_queries = []
    for source,target in MATCHED:
        for day in SAMPLE_DAYS:
            b,h,sb,th,basis = sample_frames(source,target,day)
            base = np.sort(rng.choice(np.arange(700, len(b['mid'])-1000, 10), size=100, replace=False))
            with np.load(E.OUT/(source+'__'+target)/(day+'.coefficients.npz')) as stored:
                for delay in (0,500,1000,60000):
                    x = manual_features(sb,th,basis,base,delay)
                    own,_ = E.own_features(th,E.TAIL+base)
                    sources,_ = E.source_features(sb,h['mid'][base],basis,E.TAIL+base-delay//100)
                    ref_x = np.c_[own,sources]
                    feature_error = max(feature_error,assert_close(x,ref_x,'independent features',atol=1e-8,rtol=1e-10))
                    x = np.where(np.isfinite(x),x,0.)
                    for mode in ('frozen','refit'):
                        ints=stored[f'a1_d{delay}_{mode}_intercept']
                        coef=stored[f'a1_d{delay}_{mode}_coefficient']
                        direct=manual_prediction(x,ints,coef,outputs)
                        fits=list(zip(ints,coef))
                        ref=E.prediction_matrix(x,fits,np.array(outputs))
                        prediction_error=max(prediction_error,assert_close(direct,ref,'independent prediction accumulation',atol=2e-8,rtol=1e-10))
                        prediction_values+=direct.size
            for ahead in (10,50):
                for side in ('bid','ask'):
                    now=h[side][base]
                    current_truth=h['truth_'+side][base]
                    future_truth=h['truth_'+side][base+ahead]
                    endpoint=10000*(future_truth/now-1)
                    innovation=10000*(future_truth/current_truth-1)
                    catchup=10000*(current_truth/now-1)
                    identity_error=max(identity_error,assert_close(1+endpoint/10000,
                        (1+innovation/10000)*(1+catchup/10000),'endpoint innovation identity',atol=1e-14))
            # Mutation at one actual decision cutoff: all later BN/HL fields
            # including native labels change, while the frozen feature row does not.
            cutoff=int(E.TAIL+base[0])
            short_b={k:v[:cutoff+100].copy() for k,v in sb.items()}
            short_h={k:v[:cutoff+100].copy() for k,v in th.items()}
            short_basis=basis[:cutoff+100]
            one=np.array([base[0]])
            original=manual_features(short_b,short_h,short_basis,one,0)
            for values in short_b.values():
                values[cutoff+1:]*=1.3
            for values in short_h.values():
                values[cutoff+1:]*=.7
            altered_basis,_=E.causal_basis(short_b['mid'],short_h['mid'],float(short_basis[0]))
            # Rebuild only the future basis portion; the prefix is the original
            # known as-of carrier and cannot be influenced by future changes.
            mutated_basis=short_basis.copy()
            mutated_basis[cutoff+1:]=altered_basis[cutoff+1:]
            changed=manual_features(short_b,short_h,mutated_basis,one,0)
            mutation_error=max(mutation_error,assert_close(original,changed,'future BN/HL feature mutation',atol=0,rtol=0))
            # All sampled cache timestamps must be at or before their query.
            with np.load(E.INPUT/target/(day+'.npz')) as cache:
                for indices in (base,base+10,base+50):
                    q=cache['grid_us'][indices]
                    if np.any(cache['recv_stamp_us'][indices]>q) or np.any(cache['truth_event_us'][indices]>q):
                        raise AssertionError('Sample cache contains future BBO information')
            sampled_decisions+=len(base)
            # Five source decisions in each of four fixed clock-hours gives
            # compact raw-hours verification without a full-period reread.
            selected_hours=(0,9,15,23)
            for hour in selected_hours:
                sec=hour*3600+int(rng.integers(10,3500))
                raw_queries.append((target,day,sec*10))
    return dict(sample_days=list(SAMPLE_DAYS),matched_pairs=2,source_delays_ms=[0,500,1000,60000],
                families=6,endpoint_horizons_ms=[1000,5000],decisions=sampled_decisions,
                prediction_values=prediction_values,feature_max_error=feature_error,
                prediction_max_error=prediction_error,endpoint_identity_max_error=identity_error,
                future_feature_mutation_max_error=mutation_error,
                basis_reconstruction='previous 6h burn-in, 72 half-lives; prior initialization below machine precision'),raw_queries


def raw_label_samples(queries):
    checked=0
    max_error=0.
    for market,day,base in queries:
        dt=pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(milliseconds=base*100)
        hour=dt.to_pydatetime().replace(minute=0,second=0,microsecond=0)
        event,received,seq,bid,ask,bq,aq=I.canonical.read_bbo_hour(market,hour)
        receive_order=np.lexsort((seq,received))
        sorted_seq=seq[receive_order]
        keep=np.ones(len(sorted_seq),bool)
        keep[1:]=sorted_seq[1:]>=np.maximum.accumulate(sorted_seq[:-1])
        receive_order=receive_order[keep]
        native_order=np.lexsort((received,event))
        with np.load(E.INPUT/market/(day+'.npz')) as cache:
            for ahead in (0,1,5,10,50):
                query=int(cache['grid_us'][base+ahead])
                ri=receive_order[np.flatnonzero(received[receive_order]<=query)[-1]]
                ni=native_order[np.flatnonzero(event[native_order]<=query)[-1]]
                for side,raw in (('bid',bid),('ask',ask)):
                    max_error=max(max_error,assert_close(cache[side][base+ahead],raw[ri],'raw received quote',atol=0,rtol=0),
                                  assert_close(cache['truth_'+side][base+ahead],raw[ni],'raw native quote',atol=0,rtol=0))
                    checked+=2
    return dict(raw_hour_queries=len(queries),native_and_received_price_values=checked,
                forward_offsets_ms=[0,100,500,1000,5000],max_error=max_error)


def decision_guard_tables():
    daily=pd.read_csv(D.OUT/'daily.csv.gz')
    summary=pd.read_csv(D.OUT/'summary.csv')
    if len(daily)!=4080 or len(summary)!=240:
        raise AssertionError(f'Incomplete frozen-decision grid {len(daily)}/{len(summary)}')
    keys=['source','target','family','side','phase_ms','start_ms','end_ms']
    grouped=daily.groupby(keys)
    if not grouped.day.nunique().eq(17).all():
        raise AssertionError('Decision guard profile lacks 17 dates')
    s=summary.set_index(keys).sort_index()
    totals=grouped[['n','gain_sum','abs_sum','own_abs_sum','nonzero']].sum()
    for key in totals:
        assert_close(s[key],totals[key],'decision guard aggregation '+key,atol=1e-7)
    assert_close(s.gain_vs_own_bp,s.gain_sum/s.n,'decision guard gain identity')
    baseline=[k for k in keys if k!='family']+['day']
    for field in ('n','nonzero','own_abs_sum'):
        block=daily.groupby(baseline)[field].agg(['min','max'])
        assert_close(block['min'],block['max'],'guard family matched baseline '+field)
    return dict(daily_rows=len(daily),summary_rows=len(summary),all_groups_17_dates=True,
                frozen_information='all features at original t; endpoint prediction differences use same denominator')


def ranking_tables(summary):
    path=OUT/'ranking/ranking.csv'
    rank=pd.read_csv(path)
    keys=['source','target','pair_type','family','horizon_ms','max_age','mode']
    if len(rank)!=336 or rank.duplicated(keys).any():
        raise AssertionError('Incomplete or duplicated ranking profiles')
    calculations=0
    for _,row in rank.iterrows():
        mask=np.ones(len(summary),bool)
        for k in keys:
            mask &= summary[k].eq(row[k]).to_numpy()
        selected=summary[mask & summary.clock.isin(['native_innovation','native_endpoint']).to_numpy()]
        expected=np.nan
        for delay,group in selected.groupby('delay_ms',sort=True):
            if set(zip(group.clock,group.side))!={(c,s) for c in ('native_innovation','native_endpoint') for s in ('bid','ask')}:
                break
            if not np.isfinite(group.gain_vs_own_simul_low).all() or group.gain_vs_own_simul_low.min()<=0:
                break
            expected=float(delay)
        assert_close(row.contiguous_life_ms,expected,'ranking prefix lifetime',atol=0,rtol=0)
        calculations+=1
    return dict(profiles=len(rank),independent_prefix_lifetimes=calculations,
                no_isolated_late_lag_counted_as_lifetime=True)


def verify(allow_partial=False):
    input_meta=json.loads((E.INPUT/'manifest.json').read_text())
    raw_meta=json.loads((E.INPUT/'validation_samples.json').read_text())
    if not input_meta['complete_grid'] or input_meta['rows']!=62208000 or raw_meta['status']!='pass':
        raise AssertionError('100ms input evidence incomplete')
    daily,summary,tables=full_tables(allow_partial)
    states=checkpoint_validation()
    predictions,queries=sampled_predictions()
    raw=raw_label_samples(queries)
    guard=decision_guard_tables()
    ranks=ranking_tables(summary)
    return dict(status='pass',core=tables,training_states=states,
                sampled_prediction_reconstruction=predictions,raw_label_checks=raw,
                frozen_decision_guard=guard,ranking=ranks,
                input_evidence=dict(rows=input_meta['rows'],day_files=input_meta['day_files'],
                                    prior_raw_values_checked=raw_meta['raw_values_checked']),
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),Path(E.__file__),Path(D.__file__),Path(I.__file__)]},
                limitations=['Diagnostics on previously examined dates, not untouched holdout.',
                             'Raw as-of checks cover selected hours and do not prove exchange/collector clock synchronization.',
                             'Independent prediction accumulation reuses saved coefficients; previous-state fitting check uses the documented solver.',
                             'Only matched-pair selected-date forecasts are reconstructed in full detail; all profiles receive aggregate checks.',
                             'The covariance/model family search and inferred signal lifetimes do not establish executable profit.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    output=OUT/'verification.json'
    try:
        result=verify(args.allow_partial)
    except Exception as exc:
        output.write_text(json.dumps(dict(status='fail',error=repr(exc)),indent=2)+'\n')
        raise
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','limitations')},indent=2))


if __name__=='__main__':
    main()
