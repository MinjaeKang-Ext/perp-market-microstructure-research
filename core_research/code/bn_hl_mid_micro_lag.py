"""100ms physical response to cadence-bin BN mid+micro, matched BN→HL only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E

OUT=C.ROOT/'out_bn_hl_mid_micro_lag'
WITHIN_EXCHANGE_PAIRS=((C.PAIRS[0][0],C.PAIRS[1][0]),
                       (C.PAIRS[1][0],C.PAIRS[0][0]),
                       (C.PAIRS[0][1],C.PAIRS[1][1]),
                       (C.PAIRS[1][1],C.PAIRS[0][1]))
ADR_TO_DOMESTIC_PAIRS=tuple((source,target)
    for source in (C.PAIRS[1][0],C.PAIRS[1][1])
    for target in (C.PAIRS[0][0],C.PAIRS[0][1]))
DOMESTIC_TO_ADR_PAIRS=tuple((source,target)
    for source in (C.PAIRS[0][0],C.PAIRS[0][1])
    for target in (C.PAIRS[1][0],C.PAIRS[1][1]))
CROSS_TYPE_BN_HL_PAIRS=((C.PAIRS[0][0],C.PAIRS[1][1]),
                        (C.PAIRS[1][0],C.PAIRS[0][1]))
LAGS=np.arange(-50,51,dtype=int)  # end-aligned 100ms returns, −5s ... +5s
FORECAST_LAGS=(1,2,3,4,5,6,7,8,9,10,15,20,30,50)
FAMILIES=('mid','micro','both')
SELECTION={'mid':list(range(5))+[5],
           'micro':list(range(5))+[6],
           'both':list(range(5))+[5,7]}


def returns100(target):
    result={}
    for side in ('bid','ask'):
        price=target[side]
        r=np.full(len(price),np.nan)
        r[1:]=10000*(np.log(price[1:])-np.log(price[:-1]))
        result[side]=r
    age=C.valid_age(target['age_sec'])
    valid=age & np.r_[False,age[:-1]]
    return result,valid


def design(source,target,cadence_ms,phase_samples):
    at,before,x,base_valid,source_valid=C.assemble(source,target,cadence_ms,phase_samples)
    # Mid + micro spans mid + (micro-mid); use the latter for stable fitting.
    x=np.column_stack((x[:,:5],x[:,5],x[:,6],x[:,6]-x[:,5]))
    return at,x,base_valid,source_valid


def moment_rows(day,source_name,target_name,cadence_ms,phase_ms,at,x,source_valid,ret,ret_valid):
    source_two=x[:,5:7]
    rows=[]
    for lag in LAGS:
        end=at+lag
        within=(end>=1)&(end<len(ret_valid))
        safe=np.clip(end,1,len(ret_valid)-1)
        y=np.column_stack((ret['bid'][safe],ret['ask'][safe]))
        keep=(source_valid&np.isfinite(source_two).all(axis=1)
              &within&ret_valid[safe]&np.isfinite(y).all(axis=1))
        a,b=source_two[keep],y[keep]
        if not len(a):
            continue
        sx=a.sum(axis=0); sy=b.sum(axis=0)
        sxx=np.einsum('ij,ij->j',a,a); syy=np.einsum('ij,ij->j',b,b)
        sxy=a.T@b
        for i,feature in enumerate(('mid','micro')):
            for j,side in enumerate(('bid','ask')):
                rows.append(dict(day=day,source=source_name,target=target_name,
                    cadence_ms=cadence_ms,phase_ms=phase_ms,feature=feature,
                    side=side,end_lag_ms=int(lag*100),lag_samples=int(lag),
                    n=int(len(a)),sx=float(sx[i]),sy=float(sy[j]),
                    sxx=float(sxx[i]),syy=float(syy[j]),sxy=float(sxy[i,j]),
                    source_nonzero=int(np.count_nonzero(a[:,i])),
                    target_nonzero=int(np.count_nonzero(b[:,j]))))
    return rows


def fitted(moment):
    return [('own',moment.solve(list(range(5))))]+[
        (name,moment.solve(SELECTION[name])) for name in FAMILIES]


def score(day,source_name,target_name,cadence_ms,phase_ms,end_lag_ms,x,y,valid,models):
    if models[0][1] is None:
        return []
    ids=np.flatnonzero(valid)
    if not len(ids):
        return []
    xs,actual=x[ids],y[ids]
    predicted={name:intercept+xs@coef for name,(intercept,coef) in models}
    own_abs=np.abs(actual-predicted['own'])
    mid_abs=np.abs(actual-predicted['mid'])
    rows=[]
    for name in FAMILIES:
        error=np.abs(actual-predicted[name])
        for j,side in enumerate(('bid','ask')):
            rows.append(dict(day=day,source=source_name,target=target_name,
                cadence_ms=cadence_ms,phase_ms=phase_ms,end_lag_ms=end_lag_ms,
                family=name,side=side,n=len(ids),
                gain_vs_own_sum=float(np.sum(own_abs[:,j]-error[:,j])),
                gain_vs_mid_sum=float(np.sum(mid_abs[:,j]-error[:,j])),
                own_abs_sum=float(np.sum(own_abs[:,j])),
                mid_abs_sum=float(np.sum(mid_abs[:,j])),
                model_abs_sum=float(np.sum(error[:,j])),
                nonzero=int(np.count_nonzero(actual[:,j]))))
    return rows


def process(pairs=C.PAIRS,output=OUT):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    response_rows=[]; forecast_rows=[]; coverage_rows=[]
    moment={ (pi,cadence,ph,lag):E.Moments.empty(8,2)
        for pi in range(len(pairs)) for cadence in C.CADENCES_MS for ph in (0,1)
        for lag in FORECAST_LAGS}
    for di,day in enumerate(C.DAYS):
        for pi,(source_name,target_name) in enumerate(pairs):
            source,target=C.load(source_name,day),C.load(target_name,day)
            ret,ret_valid=returns100(target)
            for cadence_ms in C.CADENCES_MS:
                k=cadence_ms//100
                for ph,phase_samples in enumerate((0,k//2)):
                    phase_ms=phase_samples*100
                    at,x,base_valid,source_valid=design(source,target,cadence_ms,phase_samples)
                    response_rows.extend(moment_rows(day,source_name,target_name,cadence_ms,
                        phase_ms,at,x,source_valid,ret,ret_valid))
                    for lag in FORECAST_LAGS:
                        end=at+lag
                        within=end<len(ret_valid)
                        safe=np.minimum(end,len(ret_valid)-1)
                        y=np.column_stack((ret['bid'][safe],ret['ask'][safe]))
                        valid=base_valid&within&ret_valid[safe]&np.isfinite(y).all(axis=1)
                        coverage_rows.append(dict(day=day,source=source_name,target=target_name,
                            cadence_ms=cadence_ms,phase_ms=phase_ms,end_lag_ms=lag*100,
                            grid_rows=len(at),valid_rows=int(np.count_nonzero(valid))))
                        key=(pi,cadence_ms,ph,lag)
                        if di:
                            forecast_rows.extend(score(day,source_name,target_name,
                                cadence_ms,phase_ms,lag*100,x,y,valid,fitted(moment[key])))
                        moment[key].update(x[valid],y[valid])
            print('PROCESSED_JOINT_LAG',day,source_name,target_name,flush=True)
            del source,target
    pd.DataFrame(response_rows).to_parquet(output/'response_daily.parquet',index=False)
    pd.DataFrame(forecast_rows).to_parquet(output/'forecast_daily.parquet',index=False)
    pd.DataFrame(coverage_rows).to_csv(output/'coverage_daily.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(dict(
        pairs=[list(v) for v in pairs],days=list(C.DAYS),
        cadences_ms=list(C.CADENCES_MS),
        response_end_lags_ms=[int(v) for v in LAGS*100],
        forecast_end_lags_ms=[int(v*100) for v in FORECAST_LAGS],
        source_features='completed cadence-bin BN mid and L1-weighted-micro log returns; both uses mid and micro-minus-mid',
        target='HL best bid/ask return over 100ms ending at t+lag',
        causal_clock='collector received_at',freshness_max_event_age_sec=1,
        first_day_training_only=True,
        response_daily_rows=len(response_rows),forecast_daily_rows=len(forecast_rows),
        coverage_daily_rows=len(coverage_rows)),indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',choices=('matched','within_exchange','adr_to_domestic',
                                            'domestic_to_adr','cross_type_bn_hl'),default='matched')
    args=parser.parse_args()
    mapping=dict(matched=(C.PAIRS,OUT),
                 within_exchange=(WITHIN_EXCHANGE_PAIRS,C.ROOT/'out_adr_domestic_within_exchange'),
                 adr_to_domestic=(ADR_TO_DOMESTIC_PAIRS,C.ROOT/'out_adr_to_domestic_exchanges'),
                 domestic_to_adr=(DOMESTIC_TO_ADR_PAIRS,C.ROOT/'out_domestic_to_adr_exchanges'),
                 cross_type_bn_hl=(CROSS_TYPE_BN_HL_PAIRS,C.ROOT/'out_bn_hl_cross_type_lag'))
    process(*mapping[args.cases])
