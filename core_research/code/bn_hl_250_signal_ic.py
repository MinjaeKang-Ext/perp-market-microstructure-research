"""250ms raw BN-signal IC against next-lag HL quote returns (descriptive)."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from work import bn_hl_250_response as R
from work import bn_hl_cadence as C

NAMES=('mid','micro')
SIDES=('bid','ask')


def corr(x,y):
    x=x-np.mean(x);y=y-np.mean(y)
    denom=np.sqrt(np.dot(x,x)*np.dot(y,y))
    return float(np.dot(x,y)/denom) if denom>0 else float('nan')


def process(day,source,target):
    s=R.load(source,day,0);t=R.load(target,day,0)
    assert np.array_equal(s['grid_us'],t['grid_us'])
    n=len(s['grid_us']);at=np.arange(1,n-2);end=at+2
    def valid(d,ix):
        return (C.valid_age(d['age_sec'][ix])&(d['bid'][ix]<d['ask'][ix]))
    with np.errstate(invalid='ignore',divide='ignore'):
        mid=(s['bid']+s['ask'])/2
        micro=(s['ask']*s['bq']+s['bid']*s['aq'])/(s['bq']+s['aq'])
        p=np.column_stack((mid,micro));q=np.column_stack((t['bid'],t['ask']))
        x=1e4*(np.log(p[at])-np.log(p[at-1]))
        y=1e4*(np.log(q[end])-np.log(q[end-1]))
    keep=(valid(s,at)&valid(s,at-1)&valid(t,end)&valid(t,end-1)&
          np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1))
    return x[keep],y[keep]


def main():
    daily=[];summary=[];errors=[]
    for source,target in C.PAIRS:
        parts=[]
        for day in C.DAYS:
            x,y=process(day,source,target)
            ranks_x=np.column_stack([rankdata(x[:,j],method='average') for j in range(2)])
            ranks_y=np.column_stack([rankdata(y[:,j],method='average') for j in range(2)])
            for i,feature in enumerate(NAMES):
                for j,side in enumerate(SIDES):
                    daily.append((day,source,target,feature,side,len(x),
                        corr(x[:,i],y[:,j]),corr(ranks_x[:,i],ranks_y[:,j]),
                        float(np.mean(y[:,j]==0))))
            parts.append((x,y))
            print('RAW_SIGNAL_IC',day,source,flush=True)
        X=np.concatenate([p[0] for p in parts]);Y=np.concatenate([p[1] for p in parts])
        del parts
        RX=np.column_stack([rankdata(X[:,j],method='average') for j in range(2)])
        RY=np.column_stack([rankdata(Y[:,j],method='average') for j in range(2)])
        pooled=pd.read_csv(R.ROOT/'response_curve.csv')
        pooled=pooled[(pooled.source==source)&(pooled.target==target)&
             (pooled.phase_ms==0)&(pooled.regime=='all')&(~pooled.exclude_funding)&
             (pooled.end_lag_ms==500)]
        D=pd.DataFrame(daily,columns=['day','source','target','feature','side',
             'n','ic','rank_ic','target_zero_fraction'])
        for i,feature in enumerate(NAMES):
            for j,side in enumerate(SIDES):
                g=D[(D.source==source)&(D.target==target)&(D.feature==feature)&(D.side==side)]
                g2=pooled[(pooled.feature==feature)&(pooled.side==side)]
                assert len(g2)==1 and int(g2.iloc[0].n)==len(X)
                ic=corr(X[:,i],Y[:,j]);rank_ic=corr(RX[:,i],RY[:,j])
                error=abs(ic-g2.iloc[0]['corr']);errors.append(error)
                assert error<1e-10
                summary.append((source,target,feature,side,len(X),len(g),ic,rank_ic,
                    g.ic.mean(),g.ic.std(ddof=1),int((g.ic>0).sum()),
                    g.rank_ic.mean(),g.rank_ic.std(ddof=1),int((g.rank_ic>0).sum()),
                    float(np.mean(Y[:,j]==0))))
        del X,Y,RX,RY
    cols=['day','source','target','feature','side','n','ic','rank_ic','target_zero_fraction']
    pd.DataFrame(daily,columns=cols).to_csv(R.ROOT/'signal_ic_daily.csv',index=False)
    cols=['source','target','feature','side','n','days','pooled_ic','pooled_rank_ic',
          'daily_mean_ic','daily_std_ic','positive_ic_days','daily_mean_rank_ic',
          'daily_std_rank_ic','positive_rank_ic_days','target_zero_fraction']
    out=pd.DataFrame(summary,columns=cols)
    out.to_csv(R.ROOT/'signal_ic_summary.csv',index=False)
    (R.ROOT/'signal_ic_verification.json').write_text(json.dumps(dict(
        raw_signal_not_fitted_forecast=True,clock='received_at',lag_end_ms=500,
        source_interval_ms=250,target_interval_ms=250,phase_ms=0,
        sample='all hours, funding included, per previous 250ms study',
        scipy_rankdata_method='average',pearson_curve_max_abs_error=max(errors),
        pooled_and_day_are_distinct=True),indent=2)+'\n')
    print(out.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
