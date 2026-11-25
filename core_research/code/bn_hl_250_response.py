"""Binance→HL 250ms response moments using independently rebuilt grids.

Each phase is kept separately. All times are collector received-at as-of.
"""
from __future__ import annotations

import json
import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_cash_regime as R
from work import skh_spot_fv_transfer as T

ROOT = C.ROOT/'out_bn_hl_250ms'
INPUT = ROOT/'inputs'
PAIRS = C.PAIRS
PHASES = (0, 125)
LAGS = range(-20, 21)
MOMENTS = ('n','sx','sy','sxx','syy','sxy')


def load(market, day, phase):
    suffix = '' if phase == 0 else '_phase125'
    with np.load(INPUT/market/(day+suffix+'.npz'), allow_pickle=False) as z:
        d = {k:z[k] for k in z.files}
    for k in ('grid_us','bid','ask','bq','aq','age_sec','recv_stamp_us'):
        if k not in d:
            raise KeyError(f'{market} {day} {phase} missing {k}: {list(d)}')
    grid=d['grid_us']; assert len(grid)==345600
    assert np.all(np.diff(grid)==250_000)
    valid=np.isfinite(d['age_sec'])&(d['age_sec']>=0)&(d['age_sec']<=1)
    assert np.all(d['recv_stamp_us'][valid]<=grid[valid]), 'future quote in grid'
    return d


def results(day, source_name, target_name, source, target, labels, phase):
    grid=source['grid_us']; n=len(grid)
    assert np.array_equal(grid,target['grid_us'])
    sgood=C.valid_age(source['age_sec'])
    tgood=C.valid_age(target['age_sec'])
    with np.errstate(invalid='ignore', divide='ignore'):
        smid=(source['bid']+source['ask'])/2
        denom=source['bq']+source['aq']
        smicro=(source['ask']*source['bq']+source['bid']*source['aq'])/denom
        source_prices=np.column_stack((smid,smicro))
        lr_s=np.full((n,2),np.nan)
        lr_s[1:]=1e4*(np.log(source_prices[1:])-np.log(source_prices[:-1]))
        target_prices=np.column_stack((target['bid'],target['ask']))
        lr_t=np.full((n,2),np.nan)
        lr_t[1:]=1e4*(np.log(target_prices[1:])-np.log(target_prices[:-1]))
    sgood &= np.isfinite(source_prices).all(axis=1)&(source['bid']<source['ask'])
    tgood &= np.isfinite(target_prices).all(axis=1)&(target['bid']<target['ask'])
    sv=sgood & np.r_[False,sgood[:-1]] & np.isfinite(lr_s).all(axis=1)
    tv=tgood & np.r_[False,tgood[:-1]] & np.isfinite(lr_t).all(axis=1)
    # The two KST pieces of a US cash session get the same 07:00 block.
    block=(np.arange(n)>=7*3600*4).astype(np.int8)
    previous=str((pd.Timestamp(day)-pd.Timedelta(days=1)).date())
    block_names=(previous,day)
    rows=[]
    for lag in LAGS:
        at=np.arange(max(1,1-lag),min(n,n-lag))
        end=at+lag
        good=sv[at]&tv[end]
        same=(labels[at]==labels[at-1])&(labels[end]==labels[end-1])&\
             (labels[at]==labels[end])
        before=np.minimum(grid[at-1],grid[end-1])
        after=np.maximum(grid[at],grid[end])
        funding=T.funding_clear(before,after)
        for excl in (False,True):
            base=good&(funding if excl else True)
            for regime,code in (('all',None),('KR_cash',0),('US_cash',1)):
                b=base if code is None else base&same&(labels[at]==code)
                for block_id in ((1,) if code is None else (0,1)):
                    use=b if code is None else b&(block[at]==block_id)
                    if not np.any(use): continue
                    x=lr_s[at[use]]; y=lr_t[end[use]]
                    count=len(x); sx=x.sum(axis=0); sy=y.sum(axis=0)
                    sxx=np.einsum('ij,ij->j',x,x)
                    syy=np.einsum('ij,ij->j',y,y)
                    sxy=x.T@y
                    block_name=day if code is None else block_names[block_id]
                    for i,feature in enumerate(('mid','micro')):
                        for j,side in enumerate(('bid','ask')):
                            rows.append((day,block_name,source_name,target_name,phase,
                                regime,excl,lag*250,feature,side,count,
                                sx[i],sy[j],sxx[i],syy[j],sxy[i,j]))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true',help='process input days as generation completes')
    args=parser.parse_args()
    columns=('day','block','source','target','phase_ms','regime',
             'exclude_funding','end_lag_ms','feature','side',*MOMENTS)
    ROOT.mkdir(exist_ok=True)
    pending=list(C.DAYS)
    while pending:
        ready=[]
        for day in pending:
            paths=[INPUT/m/(day+('' if p==0 else '_phase125')+'.npz')
                   for pair in PAIRS for m in pair for p in PHASES]
            # Generator currently writes NPZ directly; require an idle interval
            # and completed ZIP footer on every file before attempting reads.
            if all(p.exists() and time.time()-p.stat().st_mtime>5 and
                   zipfile.is_zipfile(p) for p in paths):
                ready.append(day)
        if not ready:
            if not args.watch:
                raise FileNotFoundError(f'Inputs not ready for {pending}')
            time.sleep(3)
            continue
        day=ready[0]
        rows=[]
        for phase in PHASES:
            frames={m:load(m,day,phase) for pair in PAIRS for m in pair}
            labels=R.regime_labels(next(iter(frames.values()))['grid_us'])
            for source,target in PAIRS:
                rows.extend(results(day,source,target,frames[source],frames[target],labels,phase))
                print('RESPONSE250',day,phase,source,target,flush=True)
        pd.DataFrame(rows,columns=columns).to_parquet(ROOT/f'response_{day}.parquet',index=False)
        pending.remove(day)
    (ROOT/'response_manifest.json').write_text(json.dumps(dict(days=list(C.DAYS),
        phases_ms=PHASES,lags_ms=[i*250 for i in LAGS],pairs=PAIRS,
        signal='BN mid/micro log return over (t-250ms,t]',
        response='HL bid/ask log return over (t+lag-250ms,t+lag]',
        clock='received_at as-of',max_age_s=1,
        regimes=['all','KR_cash','US_cash'],all_regime_boundary_exclusion=False,
        funding='included baseline and +/- 30s four-contract union sensitivity',
        cash_block='KST 07:00 to next 07:00',all_block='KST day'),indent=2)+'\n')


if __name__=='__main__':
    main()
