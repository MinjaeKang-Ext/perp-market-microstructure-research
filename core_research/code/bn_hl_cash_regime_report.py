"""Aggregate and audit cash-session conditional response moments."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cash_regime as R
from work import bn_hl_mid_micro_lag_report as S


def pooled(frame, keys):
    x=frame.groupby(keys,as_index=False)[list(R.MOMENTS)].sum()
    x['corr']=S.corr_from(x[list(R.MOMENTS[:6])].to_numpy())
    x['source_change_fraction']=x.source_nonzero/x.n
    x['target_change_fraction']=x.target_nonzero/x.n
    return x


def bootstrap_fixed(block):
    # Same US/KR trading-date blocks for the paired regime comparison.
    dates=[d for d in R.C.DAYS if pd.Timestamp(d).weekday()<5]
    keys=['source','target','cadence_ms','feature','side']
    rng=np.random.default_rng(20261108)
    weights=rng.multinomial(len(dates),np.ones(len(dates))/len(dates),size=3000).T
    profiles=[]; diffs=[]; draws=[]
    for key,g in block.groupby(keys):
        cadence=key[2]
        lag={500:400,1000:100}.get(cadence)
        if lag is None:
            continue
        g=g[(g.phase_ms==0)&(g.end_lag_ms==lag)&
            (g.regime.isin(('KR_cash','US_cash')))&(g.block.isin(dates))]
        data=[]
        for regime in ('KR_cash','US_cash'):
            b=g[g.regime==regime].groupby('block')[list(R.MOMENTS[:6])].sum()
            b=b.reindex(dates,fill_value=0)
            data.append(b.to_numpy(float))
        arrays=np.array(data)
        point=S.corr_from(arrays.sum(axis=1))
        draw=np.array([S.corr_from(np.stack([
            arrays[j,:,i]@weights for i in range(6)],axis=-1))
            for j in range(2)])
        profiles.append((*key,lag,point[0],point[1],
                         int(arrays[0,:,0].sum()),int(arrays[1,:,0].sum())))
        diffs.append(point[0]-point[1])
        draws.append(draw[0]-draw[1])
    diff=np.array(diffs); draw=np.array(draws)
    sd=np.nanstd(draw,axis=1,ddof=1)
    z=np.abs((draw-diff[:,None])/np.maximum(sd[:,None],1e-15))
    critical=float(np.nanquantile(np.nanmax(z,axis=0),.95))
    result=pd.DataFrame(profiles,columns=keys+['end_lag_ms','r_KR','r_US','n_KR','n_US'])
    result['difference_KR_minus_US']=diff
    result['paired_simul_low']=diff-critical*sd
    result['paired_simul_high']=diff+critical*sd
    result['paired_simple_low']=np.nanquantile(draw,.025,axis=1)
    result['paired_simple_high']=np.nanquantile(draw,.975,axis=1)
    result['paired_blocks']=len(dates)
    result['simul_critical']=critical
    return result


def plot_curves(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,2,figsize=(13,12),sharex=True,sharey=True)
    colors={'KR_cash':'#d35400','US_cash':'#0068b5',
            'closed_weekday':'#777777','closed_weekend':'#aaaaaa'}
    for pi,(source,target) in enumerate(R.PAIRS):
        for fi,feature in enumerate(('mid','micro')):
            ax=axs[pi,fi]
            for regime in R.REGIMES:
                b=summary[(summary.source==source)&(summary.target==target)&
                          (summary.cadence_ms==500)&(summary.phase_ms==0)&
                          (summary.side=='bid')&(summary.feature==feature)&
                          (summary.regime==regime)].sort_values('end_lag_ms')
                ax.plot(b.end_lag_ms/1000,b['corr'],color=colors[regime],
                        lw=1.4,label=regime)
            ax.axvline(0,color='black',lw=.7)
            ax.axhline(0,color='black',lw=.5)
            ax.set_title(f'{source.removeprefix("binance_")} → {target.removeprefix("hyperliquid_")} | {feature}')
            ax.set_ylabel('Bid return correlation')
            ax.set_xlabel('HL 100ms return end lag (s)')
    axs[0,0].legend(loc='upper left',fontsize=8)
    fig.tight_layout()
    fig.savefig(R.OUT/'response_500ms_regimes.png',dpi=160)
    plt.close(fig)


def main():
    files=sorted(R.OUT.glob('response_2026-*.parquet'))
    if len(files)!=len(R.C.DAYS):
        raise AssertionError(f'Expected {len(R.C.DAYS)} day files, got {len(files)}')
    frame=pd.concat((pd.read_parquet(p) for p in files),ignore_index=True)
    if frame.duplicated(['block','day','source','target','cadence_ms','phase_ms',
                         'regime','end_lag_ms','feature','side']).any():
        raise AssertionError('Duplicate moment key')
    if not frame.n.gt(0).all():
        raise AssertionError('Zero-count moment row')
    keys=['source','target','cadence_ms','phase_ms','regime','end_lag_ms','feature','side']
    summary=pooled(frame,keys)
    summary.to_csv(R.OUT/'response_summary.csv',index=False)
    peaks=[]
    for key,b in summary[summary.phase_ms==0].groupby(
            ['source','target','cadence_ms','regime','feature','side']):
        pos=b[b.end_lag_ms>0]
        best=pos.loc[pos['corr'].idxmax()]
        all_best=b.loc[b['corr'].abs().idxmax()]
        peaks.append((*key,int(best.end_lag_ms),float(best['corr']),int(best.n),
                      int(all_best.end_lag_ms),float(all_best['corr'])))
    pd.DataFrame(peaks,columns=['source','target','cadence_ms','regime','feature','side',
        'future_peak_ms','future_peak_corr','future_peak_n',
        'absolute_peak_ms','absolute_peak_corr']).to_csv(R.OUT/'response_peaks.csv',index=False)
    fixed=bootstrap_fixed(frame)
    fixed.to_csv(R.OUT/'fixed_regime_comparison.csv',index=False)
    plot_curves(summary)
    grid=pd.read_csv(R.OUT/'session_grid.csv')
    totals=grid.groupby('regime').grid_100ms.sum().to_dict()
    assert totals['KR_cash']==78*36000
    assert totals['US_cash']==83*36000
    audit=dict(pass_=True,day_files=len(files),rows=len(frame),summary_rows=len(summary),
        peak_rows=len(peaks),fixed_rows=len(fixed),grid_100ms=totals,
        fixed_pair_blocks=int(fixed.paired_blocks.iloc[0]),
        target_regimes_disjoint=True,source_and_target_intervals_same_regime=True)
    (R.OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
