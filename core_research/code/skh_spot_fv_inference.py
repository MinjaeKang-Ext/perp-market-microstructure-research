"""Paired day-block uncertainty for the fixed 500ms / 1s fair-value candidates."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import skh_spot_grid as G


def main():
    daily=pd.read_parquet(G.OUT/'forecast_daily.parquet')
    q=daily[(daily.cadence_ms==500)&(daily.phase_ms==0)&
            (daily.horizon_ms==1000)&
            (daily.segment.isin(('regular_0900_1520','late_1600_2000')))]
    days=G.DAYS[1:]
    rows=[]; point=[]; draws=[]
    rng=np.random.default_rng(20261108)
    weights=rng.multinomial(len(days),np.ones(len(days))/len(days),size=5000).T
    for (target,segment),group in q.groupby(['target','segment']):
        arrays={}
        for family in ('own','mid','micro','both'):
            sub=group[group.family==family].set_index('day').reindex(days)
            if sub[['n','sse','sae','own_sse','own_sae']].isna().any().any():
                raise AssertionError(f'Missing OOS day {target} {segment} {family}')
            arrays[family]=sub
        own=arrays['own'].sse.to_numpy(float)
        mid=arrays['mid'].sse.to_numpy(float)
        micro=arrays['micro'].sse.to_numpy(float)
        both=arrays['both'].sse.to_numpy(float)
        n=arrays['micro'].n.to_numpy(float)
        mae=(arrays['own'].sae-arrays['micro'].sae).to_numpy(float)
        ratios={
            'micro_vs_own_mse':(own-micro,own),
            'mid_vs_own_mse':(own-mid,own),
            'both_vs_own_mse':(own-both,own),
            'micro_vs_mid_mse':(mid-micro,own),
            'micro_vs_own_mae_bp':(mae,n),
        }
        for metric,(numer,denom) in ratios.items():
            value=float(numer.sum()/denom.sum())
            boot=(numer@weights)/(denom@weights)
            rows.append((target,segment,metric,int(n.sum()),len(days),
                         int(np.count_nonzero(numer>0))))
            point.append(value);draws.append(boot)
    point=np.array(point);draws=np.array(draws)
    sd=np.nanstd(draws,axis=1,ddof=1)
    critical=float(np.nanquantile(np.nanmax(np.abs(
        (draws-point[:,None])/np.maximum(sd[:,None],1e-15)),axis=0),.95))
    result=pd.DataFrame(rows,columns=['target','segment','metric','n','days','positive_days'])
    result['estimate']=point
    result['simul_low']=point-critical*sd
    result['simul_high']=point+critical*sd
    result['simple_low']=np.nanquantile(draws,.025,axis=1)
    result['simple_high']=np.nanquantile(draws,.975,axis=1)
    result['simul_critical']=critical
    result.to_csv(G.OUT/'forecast_inference.csv',index=False)
    print(json.dumps(dict(rows=len(result),days=len(days),simul_critical=critical,
        micro_regular=[r for r in result[result.metric=='micro_vs_own_mse'].
                       to_dict(orient='records') if r['segment']=='regular_0900_1520']),indent=2))


if __name__=='__main__':
    main()
