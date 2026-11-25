"""Walk-forward time-series IC of spot-micro/HL own fair-value forecasts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work import skh_spot_grid as G
from work import skh_spot_fv_forecast as F

TARGETS=('hyperliquid_SKHX','hyperliquid_SKHY')
SEGMENT='regular_0900_1520'


def rank_corr(a,b):
    return float(np.corrcoef(rankdata(a,method='average'),rankdata(b,method='average'))[0,1])


def main():
    moments={target:E.Moments.empty(8,1) for target in TARGETS}
    daily=[]; values={(target,family):[] for target in TARGETS for family in ('own','micro')}
    for di,day in enumerate(G.DAYS):
        spot=G.load(day)
        for target in TARGETS:
            perp=C.load(target,day)
            raw=perp['ofi100_norm']*(perp['bq']+perp['aq'])
            perp['_ofi_cum']=np.r_[0.,np.cumsum(np.where(np.isfinite(raw),raw,0.))]
            x,y=F.one_design(spot,perp,500,0,1000,SEGMENT)
            if di and len(y):
                for family in ('own','micro'):
                    a,b=moments[target].solve(F.FAMILIES[family]);pred=a[0]+x@b[:,0]
                    errors=y-pred
                    daily.append((day,target,family,len(y),float(np.corrcoef(pred,y)[0,1]),
                                  rank_corr(pred,y),float(np.mean(y==0)),
                                  float(np.sum(errors**2)),float(np.abs(errors).sum())))
                    values[(target,family)].append((pred,y))
            moments[target].update(x,y[:,None])
            print('SPOT_IC',day,target,flush=True)
    daily=pd.DataFrame(daily,columns=['day','target','family','n','ic','rank_ic',
                                     'actual_zero_fraction','sse','sae'])
    expected=pd.read_parquet(G.OUT/'forecast_daily.parquet')
    expected=expected[(expected.cadence_ms==500)&(expected.phase_ms==0)&
       (expected.horizon_ms==1000)&(expected.segment==SEGMENT)&
       expected.family.isin(('own','micro'))&expected.target.isin(TARGETS)]
    check=daily.merge(expected,on=['day','target','family'],validate='one_to_one')
    assert len(check)==len(daily) and (check.n_x==check.n_y).all()
    sse_error=float(np.max(np.abs(check.sse_x-check.sse_y)))
    sae_error=float(np.max(np.abs(check.sae_x-check.sae_y)))
    assert sse_error<1e-7 and sae_error<1e-7,(sse_error,sae_error)
    rows=[]
    rng=np.random.default_rng(20261108)
    weights=rng.multinomial(len(G.DAYS)-1,np.ones(len(G.DAYS)-1)/(len(G.DAYS)-1),size=5000)
    for (target,family),parts in values.items():
        pred=np.concatenate([p for p,y in parts]);actual=np.concatenate([y for p,y in parts])
        d=daily[(daily.target==target)&(daily.family==family)]
        assert len(d)==len(G.DAYS)-1
        ic_low,ic_high=np.quantile(weights@d.ic.to_numpy()/len(d),[.025,.975])
        rank_low,rank_high=np.quantile(weights@d.rank_ic.to_numpy()/len(d),[.025,.975])
        rows.append((target,family,len(actual),len(d),float(np.corrcoef(pred,actual)[0,1]),
            rank_corr(pred,actual),float(d.ic.mean()),float(d.ic.std(ddof=1)),
            int((d.ic>0).sum()),float(d.rank_ic.mean()),float(d.rank_ic.std(ddof=1)),
            int((d.rank_ic>0).sum()),float(np.mean(actual==0)),
            float(ic_low),float(ic_high),float(rank_low),float(rank_high)))
    summary=pd.DataFrame(rows,columns=['target','family','n','days','pooled_ic','pooled_rank_ic',
        'mean_daily_ic','std_daily_ic','positive_ic_days','mean_daily_rank_ic',
        'std_daily_rank_ic','positive_rank_ic_days','actual_zero_fraction',
        'mean_daily_ic_low','mean_daily_ic_high',
        'mean_daily_rank_ic_low','mean_daily_rank_ic_high'])
    daily.to_csv(G.OUT/'ic_daily.csv',index=False)
    summary.to_csv(G.OUT/'ic_summary.csv',index=False)
    (G.OUT/'ic_verification.json').write_text(json.dumps(dict(cases=len(daily),
        max_sse_abs_error=sse_error,max_sae_abs_error=sae_error,
        target='future 1s HL midpoint log return bp',
        input='500ms spot micro + HL own versus HL own only',
        correlations='Pearson and Spearman pooled OOS predictions vs realized, with daily correlations',
        rank_ties='average rank',zero_actual_retained=True,
        confidence='5000 day-block bootstrap percentile CI of unweighted daily mean; individual, not multiplicity-adjusted',
        first_day_training_only=True,validation_days=len(G.DAYS)-1),indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
