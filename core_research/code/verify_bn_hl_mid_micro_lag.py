"""Independent checks for matched mid+micro 100ms lag study."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_mid_micro_lag as M
from work import bn_hl_cadence as C


def score_reconstruction(source_name=None,target_name=None,output=M.OUT):
    if source_name is None or target_name is None:
        source_name,target_name=C.PAIRS[0]
    cadence=1000; phase=0; lag=5
    frames=[]
    for day in C.DAYS[:2]:
        source,target=C.load(source_name,day),C.load(target_name,day)
        at,x,valid,_=M.design(source,target,cadence,phase)
        ret,ret_valid=M.returns100(target)
        end=at+lag
        within=end<len(ret_valid)
        safe=np.minimum(end,len(ret_valid)-1)
        y=np.column_stack((ret['bid'][safe],ret['ask'][safe]))
        use=valid&within&ret_valid[safe]&np.isfinite(y).all(axis=1)
        frames.append((x[use],y[use]))
    train_x,train_y=frames[0]
    test_x,test_y=frames[1]
    pred={}
    for name,columns in [('own',list(range(5)))]+list(M.SELECTION.items()):
        design=np.column_stack((np.ones(len(train_x)),train_x[:,columns]))
        coeff=np.linalg.lstsq(design,train_y,rcond=1e-10)[0]
        pred[name]=np.column_stack((np.ones(len(test_x)),test_x[:,columns]))@coeff
    daily=pd.read_parquet(output/'forecast_daily.parquet')
    block=daily[(daily.day==C.DAYS[1])&(daily.source==source_name)
                &(daily.target==target_name)&(daily.cadence_ms==cadence)
                &(daily.phase_ms==0)&(daily.end_lag_ms==lag*100)]
    max_error=0.
    for name in M.FAMILIES:
        actual=np.abs(test_y-pred[name]).sum(axis=0)
        own=np.abs(test_y-pred['own']).sum(axis=0)
        mid=np.abs(test_y-pred['mid']).sum(axis=0)
        for j,side in enumerate(('bid','ask')):
            row=block[(block.family==name)&(block.side==side)].iloc[0]
            if row.n!=len(test_x):
                raise AssertionError('Independent train/test sample size differs')
            for computed,saved in ((actual[j],row.model_abs_sum),
                                   (own[j]-actual[j],row.gain_vs_own_sum),
                                   (mid[j]-actual[j],row.gain_vs_mid_sum)):
                max_error=max(max_error,abs(computed-saved))
    if max_error>2e-5:
        raise AssertionError(f'Independent OLS reconstruction mismatch {max_error}')
    return dict(train_rows=len(train_x),test_rows=len(test_x),
                independently_recomputed_models=['own','mid','micro','both'],
                max_absolute_sum_error=max_error)


def complete(output=M.OUT,pairs=C.PAIRS):
    manifest=json.loads((output/'manifest.json').read_text())
    rd=pd.read_parquet(output/'response_daily.parquet')
    fd=pd.read_parquet(output/'forecast_daily.parquet')
    rs=pd.read_csv(output/'response_summary.csv')
    fs=pd.read_csv(output/'forecast_summary.csv')
    rp=pd.read_csv(output/'response_peaks.csv')
    fp=pd.read_csv(output/'forecast_joint_peaks.csv')
    p=len(pairs)
    if tuple(map(tuple,manifest['pairs']))!=tuple(pairs):
        raise AssertionError('Manifest pair mapping mismatch')
    expected=(18*p*8*2*101*2*2,
              17*p*8*2*len(M.FORECAST_LAGS)*3*2,
              p*8*2*101*2*2,
              p*8*2*len(M.FORECAST_LAGS)*3*2,
              p*8*2*2,p*8)
    observed=tuple(map(len,(rd,fd,rs,fs,rp,fp)))
    if expected!=observed:
        raise AssertionError(f'Output row counts {observed} vs {expected}')
    if set(zip(rd.source,rd.target))!=set(pairs) or set(zip(fd.source,fd.target))!=set(pairs):
        raise AssertionError('Unexpected pair was included')
    response_keys=['source','target','cadence_ms','phase_ms','feature','side','end_lag_ms']
    forecast_keys=['source','target','cadence_ms','phase_ms','end_lag_ms','family','side']
    if rd.duplicated(['day']+response_keys).any() or fd.duplicated(['day']+forecast_keys).any():
        raise AssertionError('Duplicate result key')
    if not rd.groupby(response_keys).day.nunique().eq(18).all():
        raise AssertionError('Response date missing')
    if not fd.groupby(forecast_keys).day.nunique().eq(17).all():
        raise AssertionError('Forecast date missing')
    matched=fd.groupby(['day','source','target','cadence_ms','phase_ms','end_lag_ms','side']).n.nunique()
    if not matched.eq(1).all():
        raise AssertionError('Family models use different samples')
    error=np.max(np.abs(fd.own_abs_sum-fd.model_abs_sum-fd.gain_vs_own_sum))
    extra_error=np.max(np.abs(fd.mid_abs_sum-fd.model_abs_sum-fd.gain_vs_mid_sum))
    if error>1e-6 or extra_error>1e-6:
        raise AssertionError('MAE identities fail')
    rmatched=rd.groupby(['day','source','target','cadence_ms','phase_ms','side','end_lag_ms']).n.nunique()
    if not rmatched.eq(1).all():
        raise AssertionError('Mid/micro response samples differ')
    if not ((rs.end_lag_ms==rs.lag_samples*100)).all():
        raise AssertionError('End-aligned lag units wrong')
    return dict(row_counts=dict(zip(('response_daily','forecast_daily','response_summary',
         'forecast_summary','response_peaks','forecast_joint_peaks'),observed)),
         intended_pairs_only=True,dates_response=18,dates_forecast=17,
         family_sample_matched=True,response_feature_sample_matched=True,
         mae_identity_max_error=float(error),increment_identity_max_error=float(extra_error))


def main():
    prior=json.loads((C.OUT/'verification.json').read_text())
    if prior.get('status')!='pass':
        raise AssertionError('Cadence input/raw verification did not pass')
    result=dict(status='pass',completeness=complete(),independent_ols=score_reconstruction(),
                source_cadence_validation=dict(status=prior['status'],
                    raw_events_checked=prior['raw_spot']['raw_events_checked']))
    (M.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
