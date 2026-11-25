"""Matched-instrument BN→HL transfer after actually rebuilding each time grid.

The causal primary clock is collector received_at. Each grid samples both
venues at the same endpoint; features describe the completed bin. Lag zero is
contemporaneous. Only positive lag bins and past-day fitted forecasts are
prospective. This is a forecast diagnostic, not an executable PnL simulation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from work import bn_hl_transfer_eval as E

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'out_bn_hl_transfer/inputs'
OUT = ROOT / 'out_bn_hl_cadence'
PAIRS = ((E.BN[0], E.HL[0]), (E.BN[1], E.HL[1]))
DAYS = E.DAYS
CADENCES_MS = (500, 1000, 2000, 3000, 5000, 10000, 30000, 60000)
LAGS = (-3, -2, -1, 0, 1, 2, 3, 4, 5)
FEATURES = ('mid', 'micro', 'obi_level', 'obi_change', 'ofi')
FAMILIES = ('mid', 'micro', 'obi', 'ofi', 'mid_ofi', 'joint')
OWN = 5
GROUPS = {'mid': [5], 'micro': [6], 'obi': [7, 8], 'ofi': [9],
          'mid_ofi': [5, 9], 'joint': [5, 6, 7, 8, 9]}
FIELDS = ('grid_us', 'bid', 'ask', 'bq', 'aq', 'mid', 'imbalance',
          'spread_bp', 'weighted_offset_bp', 'ofi100_norm',
          'age_sec', 'recv_stamp_us')


def load(market: str, day: str) -> dict:
    with np.load(INPUT/market/(day+'.npz'), allow_pickle=False) as f:
        return {name: f[name].astype(np.float64) if name not in ('grid_us','recv_stamp_us')
                else f[name] for name in FIELDS}


def valid_age(a):
    return np.isfinite(a) & (a >= 0) & (a <= 1)


def assemble(source: dict, target: dict, cadence_ms: int, phase_samples: int):
    """One observation at t, using only BBO received by t and flow in (t-Δ,t]."""
    k = cadence_ms // 100
    if cadence_ms % 100 or not 0 <= phase_samples < k:
        raise ValueError('Cadence/phase must lie on the validated 100ms grid')
    if not np.array_equal(source['grid_us'], target['grid_us']):
        raise AssertionError('Source and target grid times differ')
    n = len(source['grid_us'])
    at = np.arange(k+phase_samples, n, k)
    before = at-k
    def micro(frame):
        return frame['mid']*(1+frame['weighted_offset_bp']/10000)
    def flow(frame):
        raw100 = frame['ofi100_norm']*(frame['bq']+frame['aq'])
        raw100 = np.where(np.isfinite(raw100), raw100, 0.)
        cumulative = np.r_[0., np.cumsum(raw100)]
        amount = cumulative[at+1]-cumulative[before+1]
        return amount/(frame['bq'][at]+frame['aq'][at])
    def ret(values):
        return 10000*(np.log(values[at])-np.log(values[before]))
    sx = np.column_stack((ret(source['mid']), ret(micro(source)),
                          source['imbalance'][at],
                          source['imbalance'][at]-source['imbalance'][before],
                          flow(source)))
    tx = np.column_stack((ret(target['bid']), ret(target['ask']),
                          target['spread_bp'][at], target['imbalance'][at], flow(target)))
    x = np.column_stack((tx, sx))
    source_valid = valid_age(source['age_sec'][at]) & valid_age(source['age_sec'][before])
    target_valid = valid_age(target['age_sec'][at]) & valid_age(target['age_sec'][before])
    base_valid = source_valid & target_valid & np.isfinite(x).all(axis=1)
    if np.any(source['recv_stamp_us'][at][source_valid] > source['grid_us'][at][source_valid]):
        raise AssertionError('Future source receive state')
    if np.any(target['recv_stamp_us'][at][target_valid] > target['grid_us'][at][target_valid]):
        raise AssertionError('Future target receive state')
    return at, before, x, base_valid, source_valid


def future_target(target: dict, at: np.ndarray, horizon_ms: int, base_valid: np.ndarray):
    ahead = at+horizon_ms//100
    good = ahead < len(target['mid'])
    safe = np.minimum(ahead, len(target['mid'])-1)
    y = np.column_stack([10000*(np.log(target[side][safe])-np.log(target[side][at]))
                         for side in ('bid','ask')])
    valid = base_valid & good & valid_age(target['age_sec'][safe]) & np.isfinite(y).all(axis=1)
    return y, valid


def fit(moment: E.Moments):
    controls = list(range(OWN))
    return [('own', moment.solve(controls))] + [
        (family, moment.solve(controls+GROUPS[family])) for family in FAMILIES]


def score(day, source_name, target_name, cadence_ms, phase_ms, horizon_ms,
          x, y, valid, fitted):
    ids = np.flatnonzero(valid)
    if not len(ids) or fitted[0][1] is None:
        return []
    xs, actual = x[ids], y[ids]
    own_intercept, own_coef = fitted[0][1]
    own_pred = own_intercept+xs@own_coef
    own_abs = np.abs(actual-own_pred)
    nochange_abs = np.abs(actual)
    rows = []
    for family, model in fitted[1:]:
        intercept, coef = model
        pred = intercept+xs@coef
        difference = own_abs-np.abs(actual-pred)
        for side_index, side in enumerate(('bid','ask')):
            rows.append(dict(day=day, source=source_name, target=target_name,
                cadence_ms=cadence_ms, phase_ms=phase_ms, horizon_ms=horizon_ms,
                family=family, side=side, n=len(ids),
                gain_sum=float(difference[:,side_index].sum()),
                own_abs_sum=float(own_abs[:,side_index].sum()),
                model_abs_sum=float(np.abs(actual[:,side_index]-pred[:,side_index]).sum()),
                nochange_abs_sum=float(nochange_abs[:,side_index].sum()),
                nonzero=int(np.count_nonzero(actual[:,side_index])),
                hit=int(np.count_nonzero((np.sign(pred[:,side_index])==np.sign(actual[:,side_index]))
                                            & (actual[:,side_index]!=0)))))
    return rows


def descriptive(day, source_name, target_name, cadence_ms, phase_ms,
                target, at, x, source_valid):
    k = cadence_ms//100
    feature = x[:, OWN:]
    rows = []
    for lag in LAGS:
        start = at+(lag-1)*k
        end = at+lag*k
        within = (start >= 0) & (end < len(target['mid']))
        safe_start, safe_end = np.clip(start,0,len(target['mid'])-1), np.clip(end,0,len(target['mid'])-1)
        target_valid = (within & valid_age(target['age_sec'][safe_start])
                        & valid_age(target['age_sec'][safe_end]))
        for side in ('bid','ask'):
            y = 10000*(np.log(target[side][safe_end])-np.log(target[side][safe_start]))
            for j, name in enumerate(FEATURES):
                xj = feature[:,j]
                keep = source_valid & target_valid & np.isfinite(xj) & np.isfinite(y)
                a,b = xj[keep],y[keep]
                rows.append(dict(day=day, source=source_name,target=target_name,
                    cadence_ms=cadence_ms,phase_ms=phase_ms,feature=name,side=side,
                    lag_bins=lag,end_lag_ms=lag*cadence_ms,n=len(a),
                    sx=float(a.sum()),sy=float(b.sum()),sxx=float(a@a),syy=float(b@b),
                    sxy=float(a@b),source_change=int(np.count_nonzero(a)),
                    target_change=int(np.count_nonzero(b))))
    return rows


def corr(frame):
    n=frame.n.to_numpy(float)
    xx=frame.sxx.to_numpy()-frame.sx.to_numpy()**2/n
    yy=frame.syy.to_numpy()-frame.sy.to_numpy()**2/n
    xy=frame.sxy.to_numpy()-frame.sx.to_numpy()*frame.sy.to_numpy()/n
    return np.divide(xy,np.sqrt(np.maximum(xx*yy,0)),out=np.full(len(frame),np.nan),
                     where=(xx>1e-20)&(yy>1e-20))


def process():
    OUT.mkdir(parents=True,exist_ok=True)
    forecast_rows, response_rows, coverage_rows = [],[],[]
    moments = {(pair,cadence,phase,horizon): E.Moments.empty(10,2)
        for pair in range(2) for cadence in CADENCES_MS
        for phase in (0,1) for horizon in (0,1)}
    for day_index,day in enumerate(DAYS):
        for pair_index,(source_name,target_name) in enumerate(PAIRS):
            source,target=load(source_name,day),load(target_name,day)
            for cadence_ms in CADENCES_MS:
                k=cadence_ms//100
                for phase_index,phase_samples in enumerate((0,k//2)):
                    phase_ms=phase_samples*100
                    at,before,x,base_valid,source_valid=assemble(source,target,cadence_ms,phase_samples)
                    for h_index,horizon_ms in enumerate((cadence_ms,60000)):
                        if h_index==1 and cadence_ms==60000:
                            continue
                        y,valid=future_target(target,at,horizon_ms,base_valid)
                        coverage_rows.append(dict(day=day,source=source_name,target=target_name,
                            cadence_ms=cadence_ms,phase_ms=phase_ms,horizon_ms=horizon_ms,
                            grid_rows=len(at),valid_rows=int(valid.sum()),
                            valid_fraction=float(valid.mean())))
                        key=(pair_index,cadence_ms,phase_index,h_index)
                        if day_index:
                            forecast_rows.extend(score(day,source_name,target_name,cadence_ms,
                                phase_ms,horizon_ms,x,y,valid,fit(moments[key])))
                        # Labels end inside this day and are learned only after
                        # its complete OOS scoring. No next-day label is borrowed.
                        moments[key].update(x[valid],y[valid])
                    response_rows.extend(descriptive(day,source_name,target_name,
                        cadence_ms,phase_ms,target,at,x,source_valid))
            print('PROCESSED',day,source_name,target_name,flush=True)
            del source,target
    pd.DataFrame(forecast_rows).to_parquet(OUT/'forecast_daily.parquet',index=False)
    pd.DataFrame(response_rows).to_parquet(OUT/'response_daily.parquet',index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT/'coverage_daily.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(
        date_start=DAYS[0],date_end=DAYS[-1],training_day=DAYS[0],
        pairs=[list(v) for v in PAIRS],cadences_ms=list(CADENCES_MS),
        phase_ms='0 and half-bin rounded down to 100ms',
        primary_clock='received_at',freshness_max_sec=1,
        features='BBO as-of at each cadence endpoint; bin return/OBI change and summed raw OFI numerator',
        horizons='next bin and common 60s; duplicate omitted at 60s cadence',
        notes='lag 0 contemporaneous, positive lag future; same old research dates, not untouched holdout',
        forecast_daily_rows=len(forecast_rows),response_daily_rows=len(response_rows),
        coverage_daily_rows=len(coverage_rows)),indent=2)+'\n')


class Tests(unittest.TestCase):
    def test_aggregation_and_lag(self):
        n=100
        g=np.arange(n,dtype=np.int64)*100000
        def frame(mid):
            bid=np.asarray(mid,dtype=float)-1
            ask=np.asarray(mid,dtype=float)+1
            q=np.ones(n)*10
            a=dict(grid_us=g,bid=bid,ask=ask,bq=q,aq=q,mid=np.asarray(mid,dtype=float),
                imbalance=np.zeros(n),spread_bp=10000*(ask-bid)/mid,
                weighted_offset_bp=np.zeros(n),ofi100_norm=np.zeros(n),
                age_sec=np.zeros(n),recv_stamp_us=g.copy())
            return a
        src=frame(np.ones(n)*100)
        tgt=frame(np.ones(n)*100)
        src['mid'][5]=101
        tgt['bid'][10]=100
        tgt['ask'][10]=102
        src['ofi100_norm'][1:6]=.1
        at,before,x,valid,source_valid=assemble(src,tgt,500,0)
        self.assertEqual((int(at[0]),int(before[0])),(5,0))
        self.assertAlmostEqual(x[0,9],.5)
        self.assertTrue(valid[0])
        # Source at t=500ms, target at t=1000ms: one prospective bin.
        rows=descriptive('x','s','t',500,0,tgt,at,x,source_valid)
        self.assertEqual(next(r for r in rows if r['feature']=='mid' and r['side']=='bid'
                              and r['lag_bins']==1)['end_lag_ms'],500)
        # A future receive-state mutation cannot change the original feature.
        saved=x[0].copy()
        src['mid'][6:]*=2
        tgt['mid'][6:]*=3
        _,_,altered,_,_=assemble(src,tgt,500,0)
        np.testing.assert_array_equal(saved,altered[0])


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true')
    args=p.parse_args()
    if args.selftest:
        unittest.main(argv=['ignored'],exit=False)
    else:
        process()


if __name__=='__main__':
    main()
