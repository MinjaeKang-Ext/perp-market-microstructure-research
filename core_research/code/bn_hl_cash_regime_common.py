"""Common-fresh-source support for fixed-lag cash-session correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from work import bn_hl_cash_regime as R
from work import bn_hl_mid_micro_lag_report as S


def main():
    rows=[]
    sources=(R.C.PAIRS[0][0],R.C.PAIRS[1][0])
    targets=(R.C.PAIRS[0][1],R.C.PAIRS[1][1])
    for day in R.C.DAYS:
        frames={name:R.C.load(name,day) for name in (*sources,*targets)}
        labels=R.regime_labels(frames[sources[0]]['grid_us'])
        for target_name in targets:
            target=frames[target_name]
            ret,ret_valid=R.M.returns100(target)
            for cadence,lag in ((500,4),(1000,1)):
                designs=[R.M.design(frames[source],target,cadence,0)
                         for source in sources]
                at=designs[0][0]
                assert np.array_equal(at,designs[1][0])
                end=at+lag
                within=end<len(labels)
                safe=np.minimum(end,len(labels)-1)
                same=(labels[at]==labels[at-cadence//100])&\
                     (labels[at]==labels[safe])&\
                     (labels[safe]==labels[safe-1])
                base=(within&same&ret_valid[safe]&
                      designs[0][3]&designs[1][3]&
                      np.isfinite(designs[0][1][:,5:7]).all(axis=1)&
                      np.isfinite(designs[1][1][:,5:7]).all(axis=1))
                y=np.column_stack((ret['bid'][safe],ret['ask'][safe]))
                base&=np.isfinite(y).all(axis=1)
                block=np.where(at>=7*36000,day,
                    (pd.Timestamp(day)-pd.Timedelta(days=1)).date().isoformat())
                for regime_id,regime in enumerate(R.REGIMES):
                    for analysis_block in np.unique(block):
                        use=base&(labels[at]==regime_id)&(block==analysis_block)
                        if not np.any(use):
                            continue
                        for source_name,design in zip(sources,designs):
                            x=design[1][use,5:7]
                            m=R.moment_row(x,y[use])
                            for i,feature in enumerate(('mid','micro')):
                                for j,side in enumerate(('bid','ask')):
                                    rows.append((analysis_block,day,source_name,target_name,
                                        cadence,lag*100,regime,feature,side,*m[i,j]))
        print('COMMON_SUPPORT',day,flush=True)
    columns=('block','day','source','target','cadence_ms','end_lag_ms',
             'regime','feature','side',*R.MOMENTS)
    daily=pd.DataFrame(rows,columns=columns)
    daily.to_parquet(R.OUT/'common_support_block.parquet',index=False)
    summary=daily.groupby(['source','target','cadence_ms','end_lag_ms',
        'regime','feature','side'],as_index=False)[list(R.MOMENTS)].sum()
    summary['corr']=S.corr_from(summary[list(R.MOMENTS[:6])].to_numpy())
    summary.to_csv(R.OUT/'common_support_summary.csv',index=False)


if __name__=='__main__':
    main()
