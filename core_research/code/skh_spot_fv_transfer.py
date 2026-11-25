"""Kiwoom 000660 spot shocks to four perp BBO prices, funding windows removed."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_mid_micro_lag as M
from work import skh_spot_grid as G

OUT=G.OUT
TARGETS=(C.PAIRS[0][0],C.PAIRS[0][1],C.PAIRS[1][0],C.PAIRS[1][1])
SEGMENTS=('regular_0900_1520','close_auction_1520_1530',
          'gap_1530_1600','late_1600_2000')
MOMENTS=('n','sx','sy','sxx','syy','sxy','source_nonzero','target_nonzero')


def funding_events_us():
    times=[]; counts={}
    for market in TARGETS:
        path=C.ROOT/'work/funding_momentum'/f'{market}.json'
        rows=json.loads(path.read_text())
        field='fundingTime' if market.startswith('binance') else 'time'
        t=np.array([int(r[field])*1000 for r in rows],dtype=np.int64)
        if len(t)<2 or not np.all(np.diff(t)>0):
            raise AssertionError(f'Funding history invalid: {market}')
        times.extend(t)
        counts[market]=len(t)
    return np.unique(np.array(times,dtype=np.int64)),counts


FUNDING_US,FUNDING_COUNTS=funding_events_us()


def funding_clear(lo,hi):
    # A complete signal/response envelope must not intersect any [T−30s,T+30s].
    left=np.searchsorted(FUNDING_US,lo-30_000_000,side='left')
    right=np.searchsorted(FUNDING_US,hi+30_000_000,side='right')
    return left==right


def segment_labels(n):
    # Index zero is KST 09:00 on a 100ms grid.
    at=np.arange(n)
    boundary=(15*3600+20*60-9*3600)*10
    close=(15*3600+30*60-9*3600)*10
    late=(16*3600-9*3600)*10
    return np.select((at<boundary,at<close,at<late),
                     (0,1,2),default=3).astype(np.uint8)


def target_return(frame):
    prices={k:frame[k][G.START:G.END] for k in ('mid','bid','ask')}
    age=frame['age_sec'][G.START:G.END]
    valid=C.valid_age(age)&np.r_[False,C.valid_age(age[:-1])]
    ret={}
    for name,price in prices.items():
        r=np.full(len(price),np.nan)
        with np.errstate(invalid='ignore',divide='ignore'):
            r[1:]=10000*(np.log(price[1:])-np.log(price[:-1]))
        ret[name]=r
    return ret,valid


def moments(a,b):
    n=float(len(a)); sx=a.sum(axis=0); sy=b.sum(axis=0)
    sxx=np.einsum('ij,ij->j',a,a); syy=np.einsum('ij,ij->j',b,b)
    sxy=a.T@b
    result=np.empty((2,3,8))
    for i in range(2):
        for j in range(3):
            result[i,j]=(n,sx[i],sy[j],sxx[i],syy[j],sxy[i,j],
                         np.count_nonzero(a[:,i]),np.count_nonzero(b[:,j]))
    return result


def one_day(day):
    spot=G.load(day)
    grid=spot['grid_us']
    n=len(grid)
    assert n==(20-9)*36000
    segment=segment_labels(n)
    source_ok=(C.valid_age(spot['age_sec'])&(spot['bid']<spot['ask'])&
               np.isfinite(spot['mid'])&np.isfinite(spot['micro']))
    if np.any(spot['received_us'][source_ok]>grid[source_ok]):
        raise AssertionError('Future spot quote')
    results=[]; cover=[]
    for target in TARGETS:
        perp=C.load(target,day)
        if not np.array_equal(perp['grid_us'][G.START:G.END],grid):
            raise AssertionError('Spot/perp grid mismatch')
        ret,ret_valid=target_return(perp)
        for cadence in C.CADENCES_MS:
            k=cadence//100
            for phase in (0,k//2):
                at=np.arange(k+phase,n,k)
                before=at-k
                with np.errstate(invalid='ignore',divide='ignore'):
                    x=np.column_stack([10000*(np.log(spot[f][at])-
                                             np.log(spot[f][before]))
                                       for f in ('mid','micro')])
                src=(source_ok[at]&source_ok[before]&
                     np.isfinite(x).all(axis=1))
                for lag in M.LAGS:
                    end=at+lag
                    within=(end>=1)&(end<n)
                    safe=np.clip(end,1,n-1)
                    y=np.column_stack([ret[f][safe] for f in ('mid','bid','ask')])
                    same=(segment[at]==segment[before])&\
                         (segment[at]==segment[safe])&\
                         (segment[safe]==segment[safe-1])
                    span_lo=np.minimum(grid[before],grid[safe-1])
                    span_hi=np.maximum(grid[at],grid[safe])
                    clear=funding_clear(span_lo,span_hi)
                    valid=(src&within&ret_valid[safe]&same&clear&
                           np.isfinite(y).all(axis=1))
                    for code,name in enumerate(SEGMENTS):
                        use=valid&(segment[at]==code)
                        if not np.any(use):
                            continue
                        a,b=x[use],y[use]
                        m=moments(a,b)
                        for i,feature in enumerate(('mid','micro')):
                            for j,side in enumerate(('mid','bid','ask')):
                                results.append((day,target,cadence,phase*100,name,
                                    lag*100,feature,side,*m[i,j]))
                        if lag in (1,4):
                            cover.append((day,target,cadence,phase*100,name,lag*100,
                                int(np.count_nonzero(use)),
                                int(np.count_nonzero((segment[at]==code)&src)),
                                int(np.count_nonzero((segment[at]==code)&src&clear))))
        print('SPOT_FV',day,target,flush=True)
    columns=('day','target','cadence_ms','phase_ms','segment',
             'end_lag_ms','feature','side',*MOMENTS)
    return pd.DataFrame(results,columns=columns),cover


def main():
    OUT.mkdir(exist_ok=True)
    cover=[]; total=0; daily_events=[]
    for day in G.DAYS:
        frame,c=one_day(day)
        frame.to_parquet(OUT/f'response_{day}.parquet',index=False)
        total+=len(frame);cover.extend(c)
        lo=pd.Timestamp(day+' 09:00',tz='Asia/Seoul').tz_convert('UTC').value//1000
        hi=pd.Timestamp(day+' 20:00',tz='Asia/Seoul').tz_convert('UTC').value//1000
        grid=np.arange(lo,hi,100_000,dtype=np.int64)
        grid_clear=funding_clear(grid,grid)
        daily_events.append(dict(day=day,grid_total=len(grid),
            funding_center_excluded=int(np.count_nonzero(~grid_clear))))
    pd.DataFrame(cover,columns=('day','target','cadence_ms','phase_ms','segment',
        'end_lag_ms','valid','spot_candidate','funding_clear_candidate')).to_csv(
        OUT/'coverage.csv',index=False)
    pd.DataFrame(daily_events).to_csv(OUT/'funding_mask_audit.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(
        days=list(G.DAYS),targets=list(TARGETS),cadences_ms=list(C.CADENCES_MS),
        lags_ms=[int(l*100) for l in M.LAGS],phase='zero and half-bin',
        spot='Kiwoom 000660 L2 level 1, received_at as-of, same-venue BBO backup',
        target='four perp BBO mid/bid/ask 100ms return',
        session='KST 09:00 inclusive to 20:00 exclusive',
        funding='union of four historical settlement event windows [T−30s,T+30s], whole signal/response envelope clear',
        funding_input_rows=FUNDING_COUNTS,
        max_quote_age_sec=1,rows=total),indent=2)+'\n')


if __name__=='__main__':
    main()
