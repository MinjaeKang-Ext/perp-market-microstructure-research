"""Independent raw-event spot and artifact checks for cadence transfer results."""
from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import fv_mm_inputs as raw


def raw_spot():
    day=C.DAYS[0]
    market_names=[v for pair in C.PAIRS for v in pair]
    checks=0
    max_quote_error=0.
    max_flow_error=0.
    max_normalized_flow_error=0.
    for market in market_names:
        with np.load(C.INPUT/market/(day+'.npz'),allow_pickle=False) as f:
            cache={v:f[v] for v in ('grid_us','bid','ask','bq','aq','ofi100_norm')}
        for clock in ('13:00:00','13:01:00'):
            query=datetime.fromisoformat(day+'T'+clock).replace(tzinfo=raw.KST)
            hour=query.replace(minute=0,second=0,microsecond=0)
            event,received,seq,bid,ask,bq,aq=raw.read_bbo_hour(market,hour)
            order=raw._advancing_order(received,seq,allow_equal=market.startswith('hyperliquid'))
            rr=received[order]
            bb,aa,qb,qa=[v[order] for v in (bid,ask,bq,aq)]
            inc=np.zeros(len(rr))
            inc[1:]=((bb[1:]>=bb[:-1])*qb[1:]-(bb[1:]<=bb[:-1])*qb[:-1]
                     -(aa[1:]<=aa[:-1])*qa[1:]+(aa[1:]>=aa[:-1])*qa[:-1])
            cumulative=np.r_[0.,np.cumsum(inc)]
            us=int(query.timestamp()*1_000_000)
            ci=int(np.searchsorted(cache['grid_us'],us))
            if ci>=len(cache['grid_us']) or cache['grid_us'][ci]!=us:
                raise AssertionError('Raw spot query is absent from 100ms grid')
            ri=int(np.searchsorted(rr,us,side='right')-1)
            if ri<0:
                raise AssertionError('No raw state before query')
            for field,arr in [('bid',bb),('ask',aa),('bq',qb),('aq',qa)]:
                error=abs(float(cache[field][ci])-float(arr[ri]))
                max_quote_error=max(max_quote_error,error)
                if error>1e-9:
                    raise AssertionError(f'Raw BBO mismatch {market}/{clock}/{field}')
            for cadence in C.CADENCES_MS:
                k=cadence//100
                prior=ci-k
                if prior<0:
                    continue
                left=int(np.searchsorted(rr,us-cadence*1000,side='right'))
                right=int(np.searchsorted(rr,us,side='right'))
                raw_amount=cumulative[right]-cumulative[left]
                norm=cache['ofi100_norm'][prior+1:ci+1].astype(float)
                depth=(cache['bq'][prior+1:ci+1]+cache['aq'][prior+1:ci+1]).astype(float)
                cached_amount=float(np.sum(norm*depth))
                error=abs(raw_amount-cached_amount)
                max_flow_error=max(max_flow_error,error)
                max_normalized_flow_error=max(max_normalized_flow_error,
                    error/(float(qb[ri])+float(qa[ri])))
                if error>max(2e-4,2e-6*abs(raw_amount)):
                    raise AssertionError(f'Raw OFI mismatch {market}/{clock}/{cadence}: {error}')
                checks+=1
    return dict(raw_events_checked=checks,markets=len(market_names),
                clocks=['13:00:00 hourly boundary','13:01:00 non-boundary'],
                max_quote_error=max_quote_error,max_ofi_amount_error=max_flow_error,
                max_normalized_ofi_error=max_normalized_flow_error)


def artifacts():
    fd=pd.read_parquet(C.OUT/'forecast_daily.parquet')
    rd=pd.read_parquet(C.OUT/'response_daily.parquet')
    cd=pd.read_csv(C.OUT/'coverage_daily.csv')
    fs=pd.read_csv(C.OUT/'forecast_summary.csv')
    rs=pd.read_csv(C.OUT/'response_summary.csv')
    cs=pd.read_csv(C.OUT/'coverage_summary.csv')
    expected=(12240,51840,1080,768,2880,60)
    observed=tuple(map(len,(fd,rd,cd,fs,rs,cs)))
    if observed!=expected:
        raise AssertionError(f'Row counts {observed} vs {expected}')
    if set(zip(fd.source,fd.target))!=set(C.PAIRS):
        raise AssertionError('Non-matched instrument entered forecast study')
    if set(zip(rd.source,rd.target))!=set(C.PAIRS):
        raise AssertionError('Non-matched instrument entered response study')
    if fd.duplicated(['day','source','target','cadence_ms','phase_ms','horizon_ms','family','side']).any():
        raise AssertionError('Duplicate daily forecast key')
    if rd.duplicated(['day','source','target','cadence_ms','phase_ms','feature','side','lag_bins']).any():
        raise AssertionError('Duplicate daily response key')
    if not fd.groupby(['source','target','cadence_ms','phase_ms','horizon_ms','family','side']).day.nunique().eq(17).all():
        raise AssertionError('Forecast dates incomplete')
    if not rd.groupby(['source','target','cadence_ms','phase_ms','feature','side','lag_bins']).day.nunique().eq(18).all():
        raise AssertionError('Response dates incomplete')
    for side,block in fd.groupby(['source','target','cadence_ms','phase_ms','horizon_ms','side']):
        if block.groupby('day').n.nunique().max()!=1:
            raise AssertionError('Family rows use different samples')
    check=(fd.own_abs_sum-fd.model_abs_sum-fd.gain_sum).abs().max()
    if check>1e-6:
        raise AssertionError(f'Forecast MAE identity error {check}')
    source_counts=rs[rs.lag_bins==0].groupby(['source','target','cadence_ms','phase_ms','side']).n.nunique()
    if not source_counts.eq(1).all():
        raise AssertionError('Descriptive feature cohorts differ')
    if not ((cd.valid_rows>=0)&(cd.valid_rows<=cd.grid_rows)).all():
        raise AssertionError('Invalid coverage')
    return dict(row_counts=dict(zip(('forecast_daily','response_daily','coverage_daily',
                                     'forecast_summary','response_summary','coverage_summary'),observed)),
                matched_pairs_only=True,all_forecast_profiles_17_dates=True,
                all_response_profiles_18_dates=True,
                family_matched_n=True,forecast_mae_identity_max_error=float(check),
                response_common_cohort=True)


def main():
    C.Tests('test_aggregation_and_lag').test_aggregation_and_lag()
    result=dict(status='pass',raw_spot=raw_spot(),artifacts=artifacts())
    (C.OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
