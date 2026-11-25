"""Received-at 100ms L1 grid for Kiwoom 000660 spot, 09:00–20:00 KST."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mm_stage1 import config, loaders
from work import bn_hl_cadence as C

OUT=C.ROOT/'out_skh_spot_fv'
INPUT=OUT/'spot_inputs'
DAYS=tuple(d for d in C.DAYS if pd.Timestamp(d).weekday()<5)
START=9*36000
END=20*36000


def build(day):
    base=C.load(C.PAIRS[0][0],day)['grid_us'][START:END]
    lo=pd.Timestamp(day+' 09:00',tz='Asia/Seoul')
    hi=pd.Timestamp(day+' 20:00',tz='Asia/Seoul')
    events=loaders.load_source(config.SOURCES['krx'],lo,hi)
    if events.empty:
        raise AssertionError(f'No spot events {day}')
    received=events.t.astype('int64').to_numpy()//1000
    ix=np.searchsorted(received,base,side='right')-1
    has=ix>=0
    safe=np.maximum(ix,0)
    def take(name):
        return np.where(has,events[name].to_numpy(float)[safe],np.nan)
    bid,ask,bq,aq=(take(c) for c in ('bid','ask','bid_q','ask_q'))
    mid=(bid+ask)/2
    qty=bq+aq
    with np.errstate(invalid='ignore',divide='ignore'):
        micro=(ask*bq+bid*aq)/qty
    micro=np.where(np.isfinite(qty)&(qty>0),micro,np.nan)
    stamp=np.where(has,received[safe],-1)
    age=np.where(has,(base-stamp)/1e6,np.inf)
    INPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(INPUT/f'{day}.npz',grid_us=base,received_us=stamp,
        bid=bid,ask=ask,bq=bq,aq=aq,mid=mid,micro=micro,age_sec=age)
    hours=events.groupby(events.t.dt.hour).size().to_dict()
    return dict(day=day,source_events=len(events),first=events.t.min().isoformat(),
        last=events.t.max().isoformat(),events_per_hour={str(k):int(v) for k,v in hours.items()},
        valid_mid_100ms=int(np.count_nonzero((age<=1)&(bid<ask)&np.isfinite(mid))),
        valid_micro_100ms=int(np.count_nonzero((age<=1)&(bid<ask)&np.isfinite(micro))),
        missing_qty_100ms=int(np.count_nonzero(~np.isfinite(micro))),
        crossed_100ms=int(np.count_nonzero(bid>=ask)))


def load(day):
    with np.load(INPUT/f'{day}.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def main():
    rows=[]
    for day in DAYS:
        row=build(day)
        rows.append(row)
        print('SPOT_GRID',day,row['source_events'],flush=True)
    (OUT/'spot_grid_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
