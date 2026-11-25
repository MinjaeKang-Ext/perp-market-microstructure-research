"""Rebuild 250 ms BBO grids from received events, with 0 and 125 ms phases."""
from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from work import fv_mm_inputs as source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out_bn_hl_250ms' / 'inputs'
BASE = ROOT / 'out_bn_hl_transfer' / 'inputs'
FIELDS = ('grid_us', 'bid', 'ask', 'bq', 'aq', 'age_sec',
          'recv_age_sec', 'recv_stamp_us', 'recv_event_us')


def extract_hour(market, hour):
    event, received, seq, bid, ask, bq, aq = source.read_bbo_hour(market, hour)
    order = source._advancing_order(received, seq, allow_equal=market.startswith('hyperliquid'))
    rr, ee = received[order], event[order]
    bb, aa, qb, qa = (v[order] for v in (bid, ask, bq, aq))
    start = int(hour.timestamp()*1_000_000)
    results = []
    for phase in (0, 125_000):
        grid = start + phase + np.arange(14_400, dtype=np.int64)*250_000
        ix,b,a,q1,q2,age = source._sample(grid,rr,ee,bb,aa,qb,qa)
        good = ix >= 0
        safe = np.maximum(ix,0)
        assert np.all(rr[safe[good]] <= grid[good])
        for j in (0, 1, 14_399):
            candidates = np.flatnonzero(rr <= grid[j])
            expected = candidates[-1] if len(candidates) else -1
            assert int(ix[j]) == int(expected)
        results.append(dict(grid_us=grid,bid=b,ask=a,bq=q1,aq=q2,age_sec=age,
            recv_age_sec=np.where(good,(grid-rr[safe])/1e6,np.nan),
            recv_stamp_us=np.where(good,rr[safe],-1),
            recv_event_us=np.where(good,ee[safe],-1)))
    return results


def build_day(day, market):
    output = OUT/market
    output.mkdir(parents=True,exist_ok=True)
    phase_pieces = [[],[]]
    for hour_index in range(24):
        pair=extract_hour(market,day+timedelta(hours=hour_index))
        for j in range(2): phase_pieces[j].append(pair[j])
    all_data=[]
    for j in range(2):
        data={key:np.concatenate([h[key] for h in phase_pieces[j]]) for key in FIELDS}
        assert len(data['grid_us'])==345_600 and np.all(np.diff(data['grid_us'])==250_000)
        path=output/(f'{day:%Y-%m-%d}'+('' if j==0 else '_phase125')+'.npz')
        np.savez_compressed(path,**data)
        all_data.append(data)
    if day==source.START:
        with np.load(BASE/market/f'{day:%Y-%m-%d}.npz') as prior:
            for key in ('bid','ask','bq','aq','age_sec','recv_stamp_us'):
                assert np.allclose(all_data[0][key][::2],prior[key][::5],
                                   equal_nan=True,rtol=1e-6,atol=1e-6),(market,key)
    return all_data


def main():
    started=time.monotonic()
    markets=list(source.MARKETS)
    for offset in range((source.END-source.START).days):
        day=source.START+timedelta(days=offset)
        for market in markets:
            build_day(day,market)
            print('GRID250',day.date(),market,'seconds',round(time.monotonic()-started,1),flush=True)
    (OUT.parent/'grid_manifest.json').write_text(json.dumps(dict(
        version=1,days=(source.END-source.START).days,markets=markets,
        grid_us=250_000,phases_us=[0,125_000],
        source='raw parquet BBO: same canonical advancing sequence policy as earlier 100ms grid',
        event_age='grid - exchange event time',receive_age='grid - received_at',
        checks='each hour phase first two and last point independently selects last received event; first day every market 500ms intersection matches prior 100ms grid',
        elapsed_seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':
    main()
