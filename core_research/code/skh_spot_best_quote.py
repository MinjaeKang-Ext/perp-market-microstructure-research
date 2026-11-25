"""Direct OOS prediction of perp best bid/ask 100ms changes from spot returns."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work import skh_spot_grid as G
from work import skh_spot_fv_transfer as T
from work import skh_spot_fv_forecast as F

SETTINGS=((500,100),(500,200),(1000,100),(5000,100))
SEGMENTS=F.SEGMENTS
SIDES=('bid','ask')


def design(spot,perp,cadence,lag,segment):
    n=len(spot['grid_us']);grid=spot['grid_us']
    k=cadence//100;l=lag//100
    at=np.arange(k,n-l,k);before=at-k;end=at+l;start=end-1
    labels=T.segment_labels(n);code=T.SEGMENTS.index(segment)
    keep=((labels[before]==code)&(labels[at]==code)&
          (labels[start]==code)&(labels[end]==code))
    age=perp['age_sec'][G.START:G.END]
    keep&=(C.valid_age(spot['age_sec'][before])&C.valid_age(spot['age_sec'][at])&
           C.valid_age(age[before])&C.valid_age(age[at])&
           C.valid_age(age[start])&C.valid_age(age[end])&
           (spot['bid'][before]<spot['ask'][before])&
           (spot['bid'][at]<spot['ask'][at]))
    keep&=T.funding_clear(grid[before],grid[end])
    pb=perp['bid'][G.START:G.END];pa=perp['ask'][G.START:G.END]
    full_at=G.START+at;full_before=G.START+before
    qty=perp['bq'][full_at]+perp['aq'][full_at]
    with np.errstate(invalid='ignore',divide='ignore'):
        own_bid=10000*(np.log(pb[at])-np.log(pb[before]))
        own_ask=10000*(np.log(pa[at])-np.log(pa[before]))
        own_ofi=(perp['_ofi_cum'][full_at+1]-
                 perp['_ofi_cum'][full_before+1])/qty
        sm=10000*(np.log(spot['mid'][at])-np.log(spot['mid'][before]))
        su=10000*(np.log(spot['micro'][at])-np.log(spot['micro'][before]))
        y=np.column_stack((10000*(np.log(pb[end])-np.log(pb[start])),
                           10000*(np.log(pa[end])-np.log(pa[start]))))
    x=np.column_stack((own_bid,own_ask,
        perp['spread_bp'][G.START:G.END][at],
        perp['imbalance'][G.START:G.END][at],own_ofi,sm,su,su-sm))
    keep&=np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
    return x[keep],y[keep]


def main():
    models={(target,cadence,lag,seg):E.Moments.empty(8,2)
        for target in T.TARGETS for cadence,lag in SETTINGS for seg in SEGMENTS}
    rows=[];coverage=[]
    for di,day in enumerate(G.DAYS):
        spot=G.load(day)
        for target in T.TARGETS:
            perp=C.load(target,day)
            raw=perp['ofi100_norm']*(perp['bq']+perp['aq'])
            perp['_ofi_cum']=np.r_[0.,np.cumsum(np.where(np.isfinite(raw),raw,0.))]
            for cadence,lag in SETTINGS:
                for seg in SEGMENTS:
                    key=(target,cadence,lag,seg)
                    x,y=design(spot,perp,cadence,lag,seg)
                    coverage.append((day,target,cadence,lag,seg,len(y),
                        int(np.count_nonzero(y[:,0])),int(np.count_nonzero(y[:,1]))))
                    if di and len(y):
                        fitted={name:models[key].solve(ids)
                                for name,ids in F.FAMILIES.items()}
                        if all(v is not None for v in fitted.values()):
                            a0,b0=fitted['own'];own=y-(a0+x@b0)
                            for name,(a,b) in fitted.items():
                                err=y-(a+x@b)
                                for j,side in enumerate(SIDES):
                                    rows.append((day,target,cadence,lag,seg,name,side,
                                        len(y),int(np.count_nonzero(y[:,j])),
                                        float(np.sum(err[:,j]**2)),
                                        float(np.sum(np.abs(err[:,j]))),
                                        float(np.sum(own[:,j]**2)),
                                        float(np.sum(np.abs(own[:,j]))),
                                        float(np.sum(y[:,j]**2))))
                    models[key].update(x,y)
            print('BEST_QUOTE',day,target,flush=True)
    columns=('day','target','cadence_ms','end_lag_ms','segment','family','side',
             'n','nonzero','sse','sae','own_sse','own_sae','nochange_sse')
    daily=pd.DataFrame(rows,columns=columns)
    daily.to_parquet(G.OUT/'best_quote_forecast_daily.parquet',index=False)
    pd.DataFrame(coverage,columns=('day','target','cadence_ms','end_lag_ms',
        'segment','valid','bid_nonzero','ask_nonzero')).to_csv(
        G.OUT/'best_quote_coverage.csv',index=False)
    keys=['target','cadence_ms','end_lag_ms','segment','family','side']
    summary=daily.groupby(keys,as_index=False)[['n','nonzero','sse','sae',
        'own_sse','own_sae','nochange_sse']].sum()
    summary['mse_gain_vs_own']=1-summary.sse/summary.own_sse
    summary['mae_gain_vs_own_bp']=(summary.own_sae-summary.sae)/summary.n
    summary['change_fraction']=summary.nonzero/summary.n
    summary.to_csv(G.OUT/'best_quote_forecast_summary.csv',index=False)
    (G.OUT/'best_quote_manifest.json').write_text(json.dumps(dict(
        days=list(G.DAYS),settings=[list(s) for s in SETTINGS],
        target='separate best bid and best ask 100ms log returns',
        baseline='prior-day own bid/ask Δ return, spread, imbalance, OFI',
        funding='all four contracts settlement ±30s union, entire signal-response envelope',
        rows=len(daily)),indent=2)+'\n')


if __name__=='__main__':
    main()
