"""Korean regular-session sensitivity for two cross-type BN→HL transfer pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_mid_micro_lag as M

OUT=C.ROOT/'out_bn_hl_cross_type_lag'
CADENCES=(500,1000)
LAGS=range(-20,21)


def one_day(day,source_name,target_name):
    source,target=C.load(source_name,day),C.load(target_name,day)
    ret,ret_valid=M.returns100(target)
    rows=[]
    for cadence in CADENCES:
        at,x,_,source_valid=M.design(source,target,cadence,0)
        seconds=at/10.
        regular=(seconds>=9*3600)&(seconds<15*3600+20*60)
        features=x[:,5:7]
        for lag in LAGS:
            end=at+lag
            within=(end>=1)&(end<len(ret_valid))
            safe=np.clip(end,1,len(ret_valid)-1)
            y=np.column_stack((ret['bid'][safe],ret['ask'][safe]))
            base=(source_valid&within&ret_valid[safe]
                  &np.isfinite(features).all(axis=1)&np.isfinite(y).all(axis=1))
            for session,condition in [('KR_09_1520',regular),('other',~regular)]:
                keep=base&condition
                a,b=features[keep],y[keep]
                sx=a.sum(axis=0);sy=b.sum(axis=0)
                sxx=np.einsum('ij,ij->j',a,a);syy=np.einsum('ij,ij->j',b,b)
                sxy=a.T@b
                for i,name in enumerate(('mid','micro')):
                    for j,side in enumerate(('bid','ask')):
                        rows.append(dict(day=day,source=source_name,target=target_name,
                            cadence_ms=cadence,session=session,end_lag_ms=lag*100,
                            feature=name,side=side,n=len(a),sx=float(sx[i]),sy=float(sy[j]),
                            sxx=float(sxx[i]),syy=float(syy[j]),sxy=float(sxy[i,j])))
    return rows


def main():
    rows=[]
    for day in C.DAYS:
        for source,target in M.CROSS_TYPE_BN_HL_PAIRS:
            rows.extend(one_day(day,source,target))
        print('SESSION',day,flush=True)
    daily=pd.DataFrame(rows)
    daily.to_parquet(OUT/'session_response_daily.parquet',index=False)
    keys=['source','target','cadence_ms','session','end_lag_ms','feature','side']
    s=daily.groupby(keys,as_index=False)[['n','sx','sy','sxx','syy','sxy']].sum()
    n=s.n.to_numpy(float)
    vx=s.sxx.to_numpy()-s.sx.to_numpy()**2/n
    vy=s.syy.to_numpy()-s.sy.to_numpy()**2/n
    xy=s.sxy.to_numpy()-s.sx.to_numpy()*s.sy.to_numpy()/n
    s['corr']=np.divide(xy,np.sqrt(np.maximum(0,vx*vy)),out=np.full(len(s),np.nan),
                        where=(vx>1e-20)&(vy>1e-20))
    s.to_csv(OUT/'session_response_summary.csv',index=False)
    best=s[s.end_lag_ms>0].loc[s[s.end_lag_ms>0].groupby(
        ['source','target','cadence_ms','session','feature','side'])['corr'].idxmax()]
    best.to_csv(OUT/'session_response_peaks.csv',index=False)
    print('SESSION_SUMMARY',len(daily),len(s),len(best),flush=True)


if __name__=='__main__':
    main()
