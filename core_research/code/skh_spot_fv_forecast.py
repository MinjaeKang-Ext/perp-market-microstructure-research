"""Past-day fitted spot fair value for next 1s/5s perp mid, versus own-book controls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work import skh_spot_grid as G
from work import skh_spot_fv_transfer as T

HORIZONS_MS=(1000,5000)
FAMILIES={'own':(0,1,2,3,4),'mid':(0,1,2,3,4,5),
          'micro':(0,1,2,3,4,6),'both':(0,1,2,3,4,5,7)}
SEGMENTS=('regular_0900_1520','late_1600_2000')


def one_design(spot,perp,cadence,phase,horizon,segment):
    n=len(spot['grid_us']);grid=spot['grid_us']
    k=cadence//100; h=horizon//100
    at=np.arange(k+phase,n-h,k)
    before=at-k; ahead=at+h
    labels=T.segment_labels(n)
    code=T.SEGMENTS.index(segment)
    keep=(labels[at]==code)&(labels[before]==code)&(labels[ahead]==code)
    age=perp['age_sec'][G.START:G.END]
    keep&=(C.valid_age(spot['age_sec'][at])&
           C.valid_age(spot['age_sec'][before])&
           C.valid_age(age[at])&C.valid_age(age[before])&
           C.valid_age(age[ahead])&
           (spot['bid'][at]<spot['ask'][at])&
           (spot['bid'][before]<spot['ask'][before]))
    keep&=T.funding_clear(grid[before],grid[ahead])
    p=perp['mid'][G.START:G.END]
    pb=perp['bid'][G.START:G.END]
    pa=perp['ask'][G.START:G.END]
    full_at=G.START+at;full_before=G.START+before
    qty=perp['bq'][full_at]+perp['aq'][full_at]
    with np.errstate(invalid='ignore',divide='ignore'):
        own_bid=10000*(np.log(pb[at])-np.log(pb[before]))
        own_ask=10000*(np.log(pa[at])-np.log(pa[before]))
        own_ofi=(perp['_ofi_cum'][full_at+1]-
                 perp['_ofi_cum'][full_before+1])/qty
        sm=10000*(np.log(spot['mid'][at])-np.log(spot['mid'][before]))
        su=10000*(np.log(spot['micro'][at])-np.log(spot['micro'][before]))
        y=10000*(np.log(p[ahead])-np.log(p[at]))
    x=np.column_stack((own_bid,own_ask,
        perp['spread_bp'][G.START:G.END][at],
        perp['imbalance'][G.START:G.END][at],
        own_ofi,sm,su,su-sm))
    keep&=np.isfinite(x).all(axis=1)&np.isfinite(y)
    return x[keep],y[keep]


def main():
    models={(target,cadence,phase,horizon,seg):E.Moments.empty(8,1)
        for target in T.TARGETS for cadence in C.CADENCES_MS
        for phase in (0,1) for horizon in HORIZONS_MS for seg in SEGMENTS}
    rows=[]; coverage=[]
    for di,day in enumerate(G.DAYS):
        spot=G.load(day)
        for target in T.TARGETS:
            perp=C.load(target,day)
            raw=perp['ofi100_norm']*(perp['bq']+perp['aq'])
            perp['_ofi_cum']=np.r_[0.,np.cumsum(np.where(np.isfinite(raw),raw,0.))]
            for cadence in C.CADENCES_MS:
                k=cadence//100
                for phase_index,phase_samples in enumerate((0,k//2)):
                    for horizon in HORIZONS_MS:
                        for seg in SEGMENTS:
                            key=(target,cadence,phase_index,horizon,seg)
                            x,y=one_design(spot,perp,cadence,phase_samples,horizon,seg)
                            coverage.append((day,target,cadence,phase_samples*100,
                                             horizon,seg,len(y)))
                            if di and len(y):
                                predictions={name:models[key].solve(ids)
                                             for name,ids in FAMILIES.items()}
                                if all(z is not None for z in predictions.values()):
                                    own_intercept,own_coef=predictions['own']
                                    own_pred=own_intercept[0]+x@own_coef[:,0]
                                    own_err=y-own_pred
                                    for name,(a,b) in predictions.items():
                                        pred=a[0]+x@b[:,0]
                                        err=y-pred
                                        rows.append((day,target,cadence,phase_samples*100,
                                            horizon,seg,name,len(y),
                                            float(np.sum(err**2)),float(np.sum(np.abs(err))),
                                            float(np.sum(own_err**2)),
                                            float(np.sum(np.abs(own_err))),
                                            float(np.sum(y**2)),
                                            float(np.sum(y)),
                                            float(np.sum((pred-y)**2<(own_pred-y)**2))))
                            models[key].update(x,y[:,None])
            print('SPOT_FORECAST',day,target,flush=True)
    cols=('day','target','cadence_ms','phase_ms','horizon_ms','segment','family',
          'n','sse','sae','own_sse','own_sae','nochange_sse','sum_y','better_rows')
    daily=pd.DataFrame(rows,columns=cols)
    daily.to_parquet(G.OUT/'forecast_daily.parquet',index=False)
    pd.DataFrame(coverage,columns=('day','target','cadence_ms','phase_ms',
        'horizon_ms','segment','valid')).to_csv(G.OUT/'forecast_coverage.csv',index=False)
    keys=['target','cadence_ms','phase_ms','horizon_ms','segment','family']
    summary=daily.groupby(keys,as_index=False)[['n','sse','sae','own_sse',
        'own_sae','nochange_sse','better_rows']].sum()
    summary['mse_gain_vs_own']=1-summary.sse/summary.own_sse
    summary['mae_gain_vs_own_bp']=(summary.own_sae-summary.sae)/summary.n
    summary['mse_gain_vs_nochange']=1-summary.sse/summary.nochange_sse
    summary['better_rows_fraction']=summary.better_rows/summary.n
    summary.to_csv(G.OUT/'forecast_summary.csv',index=False)
    coefficients=[]
    for (target,cadence,phase_index,horizon,seg),moment in models.items():
        for name,ids in FAMILIES.items():
            fitted=moment.solve(ids)
            if fitted is None:
                continue
            intercept,coef=fitted
            coefficients.append((target,cadence,phase_index*(cadence//100//2)*100,
                horizon,seg,name,moment.n,float(intercept[0]),
                *[float(v) for v in coef[:,0]]))
    pd.DataFrame(coefficients,columns=('target','cadence_ms','phase_ms',
        'horizon_ms','segment','family','train_n','alpha_bp',
        'own_bid_ret_beta','own_ask_ret_beta','own_spread_beta',
        'own_imbalance_beta','own_ofi_beta',
        'spot_mid_beta','spot_micro_beta','spot_micro_minus_mid_beta')).to_csv(
        G.OUT/'fv_coefficients_all_days.csv',index=False)
    (G.OUT/'forecast_manifest.json').write_text(json.dumps(dict(
        days=list(G.DAYS),first_day_training_only=True,
        horizons_ms=list(HORIZONS_MS),families={k:list(v) for k,v in FAMILIES.items()},
        baseline='past-day OLS of own perp bid/ask Δ return, spread, imbalance, Δ OFI',
        spot_features='mid, micro, both(mid and micro-minus-mid)',
        same_valid_sample_for_families=True,funding_windows='four-contract union ±30s',
        daily_rows=len(daily),summary_rows=len(summary)),indent=2)+'\n')


if __name__=='__main__':
    main()
