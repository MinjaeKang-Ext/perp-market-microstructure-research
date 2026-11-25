"""250ms response curves, held-out lag checks and day-block uncertainty."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from work import bn_hl_cadence as C

OUT=C.ROOT/'out_bn_hl_250ms'
SUMS=['n','sx','sy','sxx','syy','sxy']
KEYS=['source','target','phase_ms','regime','exclude_funding','feature','side']


def corr(s):
    n,sx,sy,xx,yy,xy=np.moveaxis(np.asarray(s,dtype=float),-1,0)
    with np.errstate(invalid='ignore',divide='ignore'):
        a=xx-sx*sx/n; b=yy-sy*sy/n
        return np.where((n>2)&(a>0)&(b>0),(xy-sx*sy/n)/np.sqrt(a*b),np.nan)


def bootstrap(group,seed=20261108):
    arr=group.groupby('block',sort=True)[SUMS].sum().to_numpy(float)
    if len(arr)<3:return (np.nan,np.nan,len(arr))
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(arr),np.ones(len(arr))/len(arr),size=3000)
    r=corr(weights@arr);r=r[np.isfinite(r)]
    return (*np.quantile(r,[.025,.975]),len(arr)) if len(r)>30 else (np.nan,np.nan,len(arr))


def main():
    daily=pd.concat([pd.read_parquet(OUT/f'response_{day}.parquet') for day in C.DAYS],ignore_index=True)
    pooled=daily.groupby(KEYS+['end_lag_ms'],as_index=False)[SUMS].sum()
    pooled['corr']=corr(pooled[SUMS].to_numpy())
    pooled.to_csv(OUT/'response_curve.csv',index=False)
    peaks=[]; holdout=[]; lag_resampling=[]
    for key,g in pooled.groupby(KEYS,sort=False):
        actual=daily
        for col,val in zip(KEYS,key): actual=actual[actual[col]==val]
        future=g[g.end_lag_ms>0]
        if future.empty:continue
        best=future.loc[future['corr'].idxmax()]
        full=g.loc[g['corr'].abs().idxmax()]
        fixed=actual[actual.end_lag_ms==best.end_lag_ms]
        low,high,blocks=bootstrap(fixed)
        peaks.append((*key,int(best.end_lag_ms),float(best['corr']),int(best.n),
                      int(full.end_lag_ms),float(full['corr']),low,high,blocks))
        # Resample complete 07:00 blocks and re-select the peak each time.
        # This describes peak-location stability, rather than treating a
        # post-selected peak as a fixed a-priori statistical test.
        byblock=actual.groupby(['block','end_lag_ms'])[SUMS].sum()
        block_ids=sorted(actual.block.unique()); lags=np.sort(g.end_lag_ms.unique())
        ix=pd.MultiIndex.from_product((block_ids,lags),names=['block','end_lag_ms'])
        cube=byblock.reindex(ix,fill_value=0).to_numpy().reshape(len(block_ids),len(lags),len(SUMS))
        rng=np.random.default_rng(20261108)
        weights=rng.multinomial(len(block_ids),np.ones(len(block_ids))/len(block_ids),size=1000)
        curves=corr(np.einsum('db,bli->dli',weights,cube,optimize=True))
        future_lags=lags[lags>0]; future_curves=curves[:,lags>0]
        good=np.isfinite(future_curves).any(axis=1)
        choices=future_lags[np.argmax(np.where(np.isfinite(future_curves[good]),
                                               future_curves[good],-np.inf),axis=1)]
        counts={str(int(k)):int(v) for k,v in zip(*np.unique(choices,return_counts=True))}
        if len(choices):
            lag_resampling.append((*key,int(len(choices)),
                float(np.quantile(choices,.025)),float(np.quantile(choices,.975)),
                json.dumps(counts,sort_keys=True)))
        # Keep 07:00 blocks intact across the discovery / validation boundary.
        discovery=actual[(actual.block>='2026-10-15')&(actual.block<='2026-10-23')]
        check=actual[(actual.block>='2026-10-24')&(actual.block<='2026-11-01')]
        d=discovery.groupby('end_lag_ms',as_index=False)[SUMS].sum()
        if d.empty or check.empty:continue
        d['corr']=corr(d[SUMS].to_numpy());d=d[d.end_lag_ms>0]
        pick=d.loc[d['corr'].idxmax()]
        v=check[check.end_lag_ms==pick.end_lag_ms]
        if v.empty:continue
        r=float(corr(v[SUMS].sum().to_numpy()))
        vlo,vhi,vblocks=bootstrap(v)
        holdout.append((*key,int(pick.end_lag_ms),float(pick['corr']),r,
                        vlo,vhi,vblocks,int(v.n.sum())))
    pd.DataFrame(peaks,columns=KEYS+['future_peak_ms','future_peak_corr','n',
        'absolute_peak_ms','absolute_peak_corr','pointwise_low','pointwise_high','blocks']).to_csv(OUT/'response_peaks.csv',index=False)
    pd.DataFrame(holdout,columns=KEYS+['selected_ms','discovery_corr','validation_corr',
        'validation_low','validation_high','validation_blocks','validation_n']).to_csv(OUT/'discovery_validation.csv',index=False)
    pd.DataFrame(lag_resampling,columns=KEYS+['bootstrap_draws','lag_q025_ms',
        'lag_q975_ms','lag_counts_json']).to_csv(OUT/'peak_lag_bootstrap.csv',index=False)
    (OUT/'summary_manifest.json').write_text(json.dumps(dict(
        day_count=len(C.DAYS),bootstrap_draws=3000,
        interval='pointwise block percentile, not simultaneous, full-sample peak selected on same data',
        validation='discovery 10/15-10/23 07:00-block; lag fixed; validation 10/24-11/01 blocks',
        bootstrapping='same 07:00 KST block groups; US session stays within block; peak stability uses 1000 resamples'),indent=2)+'\n')
    print('SUMMARY',len(daily),'daily rows',len(pooled),'curve rows',len(peaks),'peaks')


if __name__=='__main__':
    main()
