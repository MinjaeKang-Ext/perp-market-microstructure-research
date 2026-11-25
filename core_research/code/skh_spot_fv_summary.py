"""Summarize spot→perp transfer and a chronologically selected response kernel."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import skh_spot_fv_transfer as T
from work import skh_spot_grid as G


def enrich(frame):
    x=frame.copy()
    n=x.n.to_numpy(float)
    vx=x.sxx.to_numpy()-x.sx.to_numpy()**2/n
    vy=x.syy.to_numpy()-x.sy.to_numpy()**2/n
    cov=x.sxy.to_numpy()-x.sx.to_numpy()*x.sy.to_numpy()/n
    x['corr']=np.divide(cov,np.sqrt(np.maximum(vx*vy,0)),
        out=np.full(len(x),np.nan),where=(vx>1e-20)&(vy>1e-20))
    x['beta']=np.divide(cov,vx,out=np.full(len(x),np.nan),where=vx>1e-20)
    x['alpha_bp']=x.sy/n-x.beta*x.sx/n
    x['source_change_fraction']=x.source_nonzero/x.n
    x['target_change_fraction']=x.target_nonzero/x.n
    return x


def main():
    files=sorted(G.OUT.glob('response_2026-*.parquet'))
    if len(files)!=len(G.DAYS):
        raise AssertionError(f'Expected {len(G.DAYS)} spot/perp day files')
    daily=pd.concat((pd.read_parquet(p) for p in files),ignore_index=True)
    if daily.duplicated(['day','target','cadence_ms','phase_ms','segment',
                         'end_lag_ms','feature','side']).any():
        raise AssertionError('Duplicate response key')
    keys=['target','cadence_ms','phase_ms','segment','end_lag_ms','feature','side']
    moment=list(T.MOMENTS)
    total=enrich(daily.groupby(keys,as_index=False)[moment].sum())
    total.to_csv(G.OUT/'response_summary.csv',index=False)
    overall_keys=[k for k in keys if k!='segment']
    overall=enrich(daily.groupby(overall_keys,as_index=False)[moment].sum())
    overall['segment']='all_0900_2000'
    overall.to_csv(G.OUT/'response_overall_0900_2000.csv',index=False)
    split_at=G.DAYS[6]
    daily['split']=np.where(daily.day<split_at,'discovery','validation')
    split=enrich(daily.groupby(['split',*keys],as_index=False)[moment].sum())
    split.to_csv(G.OUT/'response_split.csv',index=False)
    peaks=[]
    for key,b in total[(total.phase_ms==0)&
        (total.segment.isin(('regular_0900_1520','late_1600_2000')))].groupby(
        ['target','cadence_ms','segment','feature','side']):
        pos=b[b.end_lag_ms>0]
        best=pos.loc[pos['corr'].idxmax()]
        absolute=b.loc[b['corr'].abs().idxmax()]
        peaks.append((*key,int(best.end_lag_ms),float(best['corr']),
                      float(best.beta),int(best.n),
                      int(absolute.end_lag_ms),float(absolute['corr'])))
    pd.DataFrame(peaks,columns=['target','cadence_ms','segment','feature','side',
        'future_peak_ms','future_peak_corr','future_peak_beta',
        'future_peak_n','absolute_peak_ms','absolute_peak_corr']).to_csv(
        G.OUT/'response_peaks.csv',index=False)
    # Pick one physical positive-lag response on the first six days only;
    # evaluate the exact setting on the next six days.
    ranked=[]
    discovery=split[(split.split=='discovery')&(split.phase_ms==0)&
        (split.side=='mid')&(split.segment.isin(('regular_0900_1520','late_1600_2000')))&
        (split.end_lag_ms>0)]
    for key,b in discovery.groupby(['target','cadence_ms','segment','feature']):
        best=b.loc[b['corr'].idxmax()]
        validation=split[(split.split=='validation')&(split.phase_ms==0)&
            (split.target==key[0])&
            (split.cadence_ms==key[1])&(split.segment==key[2])&
            (split.feature==key[3])&(split.side=='mid')&
            (split.end_lag_ms==best.end_lag_ms)]
        if len(validation)!=1:
            raise AssertionError('Missing validation response')
        v=validation.iloc[0]
        ranked.append((*key,int(best.end_lag_ms),float(best['corr']),
            float(v['corr']),float(best.beta),int(best.n),int(v.n)))
    pd.DataFrame(ranked,columns=['target','cadence_ms','segment','feature',
        'discovery_peak_ms','discovery_r','validation_r',
        'discovery_beta','discovery_n','validation_n']).to_csv(
        G.OUT/'selected_response_validation.csv',index=False)
    audit=dict(pass_=True,days=len(files),daily_rows=len(daily),
        response_rows=len(total),overall_rows=len(overall),
        split_rows=len(split),peak_rows=len(peaks),
        discovery_days=list(G.DAYS[:6]),validation_days=list(G.DAYS[6:]),
        funding_input_rows=T.FUNDING_COUNTS,
        funding_mask=pd.read_csv(G.OUT/'funding_mask_audit.csv').to_dict(orient='records'))
    (G.OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='funding_mask'},indent=2))


if __name__=='__main__':
    main()
