"""Independent spot/funding/response/OOS reconstruction checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mm_stage1 import config,loaders
from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work import skh_spot_grid as G
from work import skh_spot_fv_transfer as T
from work import skh_spot_fv_forecast as F


def main():
    spot=G.load('2026-10-15')
    for stamp in ('2026-10-15 09:30:00','2026-10-15 16:30:00',
                  '2026-10-15 19:59:00'):
        t=pd.Timestamp(stamp,tz='Asia/Seoul')
        ix=int(np.searchsorted(spot['grid_us'],t.tz_convert('UTC').value//1000))
        raw=loaders.load_source(config.SOURCES['krx'],
            t-pd.Timedelta(minutes=1),t+pd.Timedelta(milliseconds=100))
        row=raw[raw.t<=t].iloc[-1]
        assert abs(spot['bid'][ix]-row.bid)<1e-9
        assert abs(spot['ask'][ix]-row.ask)<1e-9
        expected=(row.ask*row.bid_q+row.bid*row.ask_q)/(row.bid_q+row.ask_q)
        if np.isfinite(expected):
            assert abs(spot['micro'][ix]-expected)<1e-9
        else:
            assert np.isnan(spot['micro'][ix])
        assert spot['received_us'][ix]<=spot['grid_us'][ix]
    # Every 09:00..20:00 hourly funding center is forbidden; 20:00 also
    # removes its preceding 30 seconds within the requested window.
    day='2026-10-15'
    for hour in range(9,21):
        t=pd.Timestamp(f'{day} {hour:02d}:00',tz='Asia/Seoul').tz_convert('UTC').value//1000
        assert not T.funding_clear(np.array([t]),np.array([t]))[0]
    near20=pd.Timestamp(day+' 19:59:40',tz='Asia/Seoul').tz_convert('UTC').value//1000
    assert not T.funding_clear(np.array([near20]),np.array([near20]))[0]
    clear=pd.Timestamp(day+' 19:58:00',tz='Asia/Seoul').tz_convert('UTC').value//1000
    assert T.funding_clear(np.array([clear]),np.array([clear]))[0]
    # Direct raw Pearson value versus saved sufficient statistics.
    target_name='hyperliquid_SKHX'
    perp=C.load(target_name,day)
    ret,ret_valid=T.target_return(perp)
    segment=T.segment_labels(len(spot['grid_us']))
    at=np.arange(5,len(segment),5);before=at-5;end=at+2
    within=end<len(segment);safe=np.minimum(end,len(segment)-1)
    with np.errstate(invalid='ignore',divide='ignore'):
        x=10000*(np.log(spot['micro'][at])-np.log(spot['micro'][before]))
    valid=(within&(segment[at]==0)&(segment[before]==0)&
        (segment[safe]==0)&(segment[safe-1]==0)&
        C.valid_age(spot['age_sec'][at])&C.valid_age(spot['age_sec'][before])&
        (spot['bid'][at]<spot['ask'][at])&(spot['bid'][before]<spot['ask'][before])&
        ret_valid[safe]&np.isfinite(x)&np.isfinite(ret['mid'][safe])&
        T.funding_clear(np.minimum(spot['grid_us'][before],spot['grid_us'][safe-1]),
                        np.maximum(spot['grid_us'][at],spot['grid_us'][safe])))
    raw_r=float(np.corrcoef(x[valid],ret['mid'][safe[valid]])[0,1])
    part=pd.read_parquet(G.OUT/f'response_{day}.parquet')
    p=part[(part.target==target_name)&(part.cadence_ms==500)&
        (part.phase_ms==0)&(part.segment=='regular_0900_1520')&
        (part.end_lag_ms==200)&(part.feature=='micro')&(part.side=='mid')]
    assert len(p)==1 and int(p.iloc[0].n)==int(valid.sum())
    z=p.iloc[0]
    rebuilt=(z.sxy-z.sx*z.sy/z.n)/np.sqrt(
        (z.sxx-z.sx*z.sx/z.n)*(z.syy-z.sy*z.sy/z.n))
    assert abs(raw_r-rebuilt)<1e-10
    # Refit on first day, score second day, compare saved OOS squared loss.
    key=dict(target=target_name,cadence_ms=500,phase_ms=0,horizon_ms=1000,
             segment='regular_0900_1520',family='micro',day='2026-10-16')
    raw=perp['ofi100_norm']*(perp['bq']+perp['aq'])
    perp['_ofi_cum']=np.r_[0.,np.cumsum(np.where(np.isfinite(raw),raw,0.))]
    x0,y0=F.one_design(spot,perp,500,0,1000,'regular_0900_1520')
    moment=E.Moments.empty(8,1);moment.update(x0,y0[:,None])
    spot1=G.load('2026-10-16');perp1=C.load(target_name,'2026-10-16')
    raw=perp1['ofi100_norm']*(perp1['bq']+perp1['aq'])
    perp1['_ofi_cum']=np.r_[0.,np.cumsum(np.where(np.isfinite(raw),raw,0.))]
    x1,y1=F.one_design(spot1,perp1,500,0,1000,'regular_0900_1520')
    a,b=moment.solve(F.FAMILIES['micro'])
    independent_sse=float(np.sum((y1-a[0]-x1@b[:,0])**2))
    daily=pd.read_parquet(G.OUT/'forecast_daily.parquet')
    q=daily.copy()
    for name,value in key.items():q=q[q[name]==value]
    assert len(q)==1 and int(q.iloc[0].n)==len(y1)
    err=abs(independent_sse-q.iloc[0].sse)
    assert err<1e-7,(err,independent_sse)
    audit=json.loads((G.OUT/'verification.json').read_text())
    audit.update(raw_spot_quotes_checked=3,funding_centers_checked=12,
        twenty_hour_boundary_checked=True,independent_response_r=raw_r,
        response_moment_error=abs(raw_r-rebuilt),
        independent_oos_sse_error=err,independent_oos_n=len(y1))
    (G.OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('pass_','raw_spot_quotes_checked',
        'funding_centers_checked','independent_response_r','response_moment_error',
        'independent_oos_sse_error','independent_oos_n')},indent=2))


if __name__=='__main__':
    main()
