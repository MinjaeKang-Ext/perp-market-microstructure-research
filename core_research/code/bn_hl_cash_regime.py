"""Cash-equity session conditional BN→HL best-price response correlations."""
from __future__ import annotations

import json
from pathlib import Path

import exchange_calendars as ec
import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_mid_micro_lag as M

OUT = C.ROOT / 'out_bn_hl_cash_regime'
PAIRS = C.PAIRS + M.CROSS_TYPE_BN_HL_PAIRS
REGIMES = ('KR_cash', 'US_cash', 'closed_weekday', 'closed_weekend')
MOMENTS = ('n','sx','sy','sxx','syy','sxy','source_nonzero','target_nonzero')


def session_intervals(calendar_name):
    cal = ec.get_calendar(calendar_name)
    sessions = cal.sessions_in_range('2026-10-13','2026-11-02')
    return [(int(cal.session_open(s).value//1000),
             int(cal.session_close(s).value//1000)) for s in sessions]


KR_INTERVALS = session_intervals('XKRX')
US_INTERVALS = session_intervals('XNYS')


def regime_labels(grid_us):
    # Grid is one KST date; the US cash session may start the preceding date.
    first = pd.Timestamp(int(grid_us[0]),unit='us',tz='UTC').tz_convert('Asia/Seoul')
    default = 3 if first.weekday() >= 5 else 2
    labels = np.full(len(grid_us),default,dtype=np.uint8)
    for code,intervals in ((0,KR_INTERVALS),(1,US_INTERVALS)):
        for lo,hi in intervals:
            left=int(np.searchsorted(grid_us,lo,side='left'))
            right=int(np.searchsorted(grid_us,hi,side='left'))
            if right>left:
                if np.any(labels[left:right]<2):
                    raise AssertionError('Cash sessions overlap')
                labels[left:right]=code
    return labels


def moment_row(a,b):
    if not len(a):
        return np.zeros((2,2,8),dtype=float)
    n=float(len(a)); sx=a.sum(axis=0); sy=b.sum(axis=0)
    sxx=np.einsum('ij,ij->j',a,a); syy=np.einsum('ij,ij->j',b,b)
    sxy=a.T@b
    out=np.empty((2,2,8))
    for i in range(2):
        for j in range(2):
            out[i,j]=n,sx[i],sy[j],sxx[i],syy[j],sxy[i,j],\
                np.count_nonzero(a[:,i]),np.count_nonzero(b[:,j])
    return out


def one_pair(day,source_name,target_name,source,target,labels):
    ret,ret_valid=M.returns100(target)
    ngrid=len(labels)
    # The 07:00 KST boundary stays outside both cash sessions. It keeps an
    # overnight US session in one resampling block even across KST midnight.
    block_split=7*3600*10
    block_names=((pd.Timestamp(day)-pd.Timedelta(days=1)).date().isoformat(),day)
    rows=[]; coverage=[]
    for cadence in C.CADENCES_MS:
        k=cadence//100
        for phase_samples in (0,k//2):
            at,x,_,source_valid=M.design(source,target,cadence,phase_samples)
            before=at-k
            features=x[:,5:7]
            source_same=labels[at]==labels[before]
            phase_ms=phase_samples*100
            block_part=(at>=block_split).astype(np.uint8)
            for lag in M.LAGS:
                end=at+lag
                within=(end>=1)&(end<ngrid)
                safe=np.clip(end,1,ngrid-1)
                y=np.column_stack((ret['bid'][safe],ret['ask'][safe]))
                target_same=(labels[safe]==labels[safe-1])
                same=source_same&target_same&(labels[at]==labels[safe])
                base=(source_valid&within&ret_valid[safe]&same&
                      np.isfinite(features).all(axis=1)&np.isfinite(y).all(axis=1))
                for code,regime in enumerate(REGIMES):
                    mask=base&(labels[at]==code)
                    for bp,block in enumerate(block_names):
                        use=mask&(block_part==bp)
                        a,b=features[use],y[use]
                        if not len(a):
                            continue
                        m=moment_row(a,b)
                        for i,feature in enumerate(('mid','micro')):
                            for j,side in enumerate(('bid','ask')):
                                rows.append((block,day,source_name,target_name,cadence,
                                    phase_ms,regime,lag*100,feature,side,*m[i,j]))
                        if lag in (1,4):
                            coverage.append((block,day,source_name,target_name,cadence,
                                phase_ms,regime,lag*100,int(np.count_nonzero(use)),
                                int(np.count_nonzero((labels[at]==code)&(block_part==bp)))))
    return rows,coverage


def main():
    OUT.mkdir(exist_ok=True)
    coverage=[]; session_grid=[]; total_rows=0
    markets=sorted(set(sum((list(p) for p in PAIRS),[])))
    for day in C.DAYS:
        day_rows=[]
        frames={m:C.load(m,day) for m in markets}
        grid=next(iter(frames.values()))['grid_us']
        assert all(np.array_equal(f['grid_us'],grid) for f in frames.values())
        labels=regime_labels(grid)
        for i,name in enumerate(REGIMES):
            session_grid.append((day,name,int(np.count_nonzero(labels==i))))
        for source_name,target_name in PAIRS:
            r,c=one_pair(day,source_name,target_name,frames[source_name],
                         frames[target_name],labels)
            day_rows.extend(r); coverage.extend(c)
            print('REGIME',day,source_name,target_name,flush=True)
        columns=('block','day','source','target','cadence_ms','phase_ms','regime',
                 'end_lag_ms','feature','side',*MOMENTS)
        daily=pd.DataFrame.from_records(day_rows,columns=columns)
        daily.to_parquet(OUT/f'response_{day}.parquet',index=False)
        total_rows+=len(daily)
    pd.DataFrame(coverage,columns=('block','day','source','target','cadence_ms',
        'phase_ms','regime','end_lag_ms','valid','candidate')).to_csv(
        OUT/'coverage.csv',index=False)
    pd.DataFrame(session_grid,columns=('day','regime','grid_100ms')).to_csv(
        OUT/'session_grid.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(
        pairs=[list(p) for p in PAIRS],days=list(C.DAYS),
        cadences_ms=list(C.CADENCES_MS),lags_ms=[int(v*100) for v in M.LAGS],
        phase='zero and half-bin',cash_calendars=['XKRX','XNYS'],
        regime='actual cash session intervals, source and target return fully inside same regime',
        resampling_block='KST 07:00 to next 07:00; partial boundary blocks retained',
        clock='collector received_at',max_event_age_sec=1,
        rows=total_rows),indent=2)+'\n')


if __name__=='__main__':
    main()
