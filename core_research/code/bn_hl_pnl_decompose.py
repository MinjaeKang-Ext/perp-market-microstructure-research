"""Fill-time spread, 60s markout and exact account-equity decomposition.

The historical 60s markout is a hypothetical fill-level diagnostic; positions
can close before 60 seconds. The reconciliation residual is not literal
post-60-second inventory PnL. All valuation prices are on the native clock.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from work import fv_mm_inputs as source

ROOT=Path(__file__).resolve().parents[1]
# BN_HL_PNL_ROOT redirects every artifact of one variant; default keeps the
# original mixed-family run in out_bn_hl_pnl intact.
BASE=Path(os.environ.get('BN_HL_PNL_ROOT',ROOT/'out_bn_hl_pnl'))
LEDGERS=BASE/'ledgers'
OUT=BASE/'analysis'
KST=ZoneInfo('Asia/Seoul')


def asof_book(event_us, mid, query_us, *, strict=False, max_age_us=1_000_000):
    """Return last valid native mid and its age, never an event after query."""
    event_us=np.asarray(event_us,dtype=np.int64)
    mid=np.asarray(mid,dtype=float)
    query_us=np.asarray(query_us,dtype=np.int64)
    assert len(event_us)==len(mid) and np.all(np.diff(event_us)>=0)
    if not len(event_us):
        return np.full(len(query_us),np.nan),np.full(len(query_us),np.inf)
    idx=np.searchsorted(event_us,query_us,side='left' if strict else 'right')-1
    safe=np.maximum(idx,0)
    age=(query_us-event_us[safe]).astype(float)
    valid=(idx>=0)&np.isfinite(mid[safe])&(mid[safe]>0)&(age>=0)&(age<=max_age_us)
    return np.where(valid,mid[safe],np.nan),np.where(idx>=0,age,np.inf)


def decompose_day(fills, event_us, event_mid, *, start_us, end_us,
                  start_inventory, end_inventory, start_mid, end_mid,
                  start_equity, end_equity, funding_usd,
                  max_age_us=1_000_000, horizon_us=60_000_000,
                  atol=1e-6):
    """Decompose one ledger day; funding is a separate realized cashflow.

    Required fills columns: time_us, side (+1 buy / -1 sell), qty, price, fee.
    For immediate mid use strict-before native BBO to avoid unknown order of
    equal-time quotes and trades. Targets use at-or-before as-of quotes.
    Matured requires +60s <= reporting end and two fresh observed mids. Other
    fills are retained in the immature or unpriced reconciliation buckets.
    """
    f={k:np.asarray(fills[k]) for k in ('time_us','side','qty','price','fee')}
    n=len(f['time_us'])
    assert all(len(v)==n for v in f.values())
    stamp=f['time_us'].astype(np.int64)
    side=f['side'].astype(float);qty=f['qty'].astype(float)
    price=f['price'].astype(float);fee=f['fee'].astype(float)
    if not np.all((stamp>=start_us)&(stamp<end_us)):
        raise AssertionError('Fill outside ledger day')
    if not np.all(np.isin(side,[-1,1])&(qty>0)&(price>0)&np.isfinite(fee)):
        raise AssertionError('Bad fill side/quantity/price/fee')
    if not (np.isfinite(start_mid) and start_mid>0 and np.isfinite(end_mid) and end_mid>0):
        raise AssertionError('Missing reporting boundary valuation mid')
    signed=side*qty
    if abs(float(start_inventory)+signed.sum()-float(end_inventory))>1e-7:
        raise AssertionError('Recorded end inventory differs from filled quantities')
    immediate,age0=asof_book(event_us,event_mid,stamp,strict=True,max_age_us=max_age_us)
    target,age60=asof_book(event_us,event_mid,stamp+horizon_us,max_age_us=max_age_us)
    m0=np.isfinite(immediate); mh=np.isfinite(target)
    time_mature=(stamp+horizon_us<=end_us)
    mature=m0&mh&time_mature
    immature=m0&~mature
    unpriced=~m0
    spread=np.sum(signed[m0]*(immediate[m0]-price[m0]))
    markout=np.sum(signed[mature]*(target[mature]-immediate[mature]))
    mature_residual=np.sum(signed[mature]*(end_mid-target[mature]))
    immature_residual=np.sum(signed[immature]*(end_mid-immediate[immature]))
    unpriced_contribution=np.sum(signed[unpriced]*(end_mid-price[unpriced]))
    opening=float(start_inventory)*(end_mid-start_mid)
    fees=-float(np.sum(fee))
    turnover=qty*price
    components=dict(spread_capture_usd=float(spread),markout_60_usd=float(markout),
        fee_pnl_usd=fees,mature_reconciliation_usd=float(mature_residual),
        immature_reconciliation_usd=float(immature_residual),
        no_fill_mid_contribution_usd=float(unpriced_contribution),
        opening_inventory_mtm_usd=float(opening),funding_usd=float(funding_usd))
    decomp=sum(components.values());ledger=float(end_equity-start_equity)
    error=decomp-ledger
    if abs(error)>atol:
        raise AssertionError(f'PnL decomposition error {error}: components={components}; ledger={ledger}')
    diagnostic=m0&mh
    mature_spread=float(np.sum(signed[mature]*(immediate[mature]-price[mature])))
    mature_fee=-float(np.sum(fee[mature]))
    result=dict(**components,partial_60_fill_edge_usd=float(spread+markout+fees),
        mature_spread_capture_usd=mature_spread,mature_fee_pnl_usd=mature_fee,
        mature_fill_edge_60_usd=mature_spread+float(markout)+mature_fee,
        total_pnl_usd=ledger,identity_error_usd=error,
        turnover_usd=float(turnover.sum()),fills=n,
        mature_60_fills=int(mature.sum()),mature_60_notional=float(turnover[mature].sum()),
        immature_fills=int(immature.sum()),immature_notional=float(turnover[immature].sum()),
        no_fill_mid_fills=int(unpriced.sum()),no_fill_mid_notional=float(turnover[unpriced].sum()),
        mark60_diagnostic_fills=int(diagnostic.sum()),
        mark60_diagnostic_notional=float(turnover[diagnostic].sum()),
        mark60_diagnostic_usd=float(np.sum(signed[diagnostic]*(target[diagnostic]-immediate[diagnostic]))),
        cross_boundary_60_fills=int(np.count_nonzero(diagnostic&~time_mature)),
        cross_boundary_60_notional=float(turnover[diagnostic&~time_mature].sum()))
    return result


def native_book(day,market):
    """Native BBO over day and neighboring 10m partitions, including +60s."""
    venue,symbol=source.MARKETS[market]
    if venue!='hyperliquid': raise ValueError(f'Expected HL market: {market}')
    date=datetime.fromisoformat(day).replace(tzinfo=KST)
    root=source.SOURCE/venue/symbol/'bbo'
    current=sorted((root/f'date={day}').rglob('*.parquet'))
    files=sorted(set(current+source.files_in_slot(root,date-timedelta(minutes=10))+
                     source.files_in_slot(root,date+timedelta(days=1))))
    if not current: raise FileNotFoundError(f'Missing native BBO {market} {day}')
    parts=[]
    for path in files:
        p=pq.ParquetFile(path).read(columns=['event_time','received_at',
            'best_bid_price','best_ask_price'])
        ev=source.us(p['event_time']);rv=source.us(p['received_at'])
        bid=p['best_bid_price'].to_numpy();ask=p['best_ask_price'].to_numpy()
        good=np.isfinite(bid)&np.isfinite(ask)&(bid>0)&(bid<ask)&(rv>=ev)
        parts.append((ev[good],rv[good],((bid+ask)/2)[good]))
    event,recv,mid=[np.concatenate([v[j] for v in parts]) for j in range(3)]
    order=np.lexsort((recv,event))
    return event[order],mid[order]


def date_bounds(day):
    dt=datetime.fromisoformat(day).replace(tzinfo=KST)
    start=int(dt.timestamp()*1e6)
    return start,int((dt+timedelta(days=1)).timestamp()*1e6)


def collect_ledgers():
    files=sorted(LEDGERS.glob('*/*/*.daily.json'))
    if not files: raise FileNotFoundError(f'No replay ledgers in {LEDGERS}')
    records=[]
    for path in files:
        d=json.loads(path.read_text())
        for k in ('day','market','cadence_ms','partition','start_cash',
                  'start_inventory','start_mid','end_cash','end_inventory',
                  'end_mid','funding_usd','pnl_usd'):
            if k not in d:raise KeyError(f'{path}: {k}')
        fills=path.with_name(path.name.replace('.daily.json','.fills.parquet'))
        if not fills.exists():raise FileNotFoundError(fills)
        d['_fills_path']=fills
        records.append(d)
    return records


def summarize(daily):
    groups=[]
    metrics=['spread_capture_usd','mature_spread_capture_usd','markout_60_usd',
             'fee_pnl_usd','mature_fee_pnl_usd',
             'mature_reconciliation_usd','immature_reconciliation_usd',
             'no_fill_mid_contribution_usd','opening_inventory_mtm_usd',
             'funding_usd','partial_60_fill_edge_usd','mature_fill_edge_60_usd',
             'total_pnl_usd','mark60_diagnostic_usd']
    for (market,cadence),g in daily.groupby(['market','cadence_ms'],sort=True):
        parts=[(str(partition),g[g.partition==partition])
               for partition in sorted(g.partition.unique())]
        parts.append(('ALL_INDEPENDENT_EPISODES',g))
        for partition,block in parts:
            turnover=float(block.turnover_usd.sum())
            mature_notional=float(block.mature_60_notional.sum())
            for metric in metrics:
                total=float(block[metric].sum())
                # IS and OOS reset at boundary; ALL is a sum of episodes.
                groups.append(dict(market=market,cadence_ms=int(cadence),
                    partition=partition,metric=metric,days=len(block),
                    total_usd=total,daily_mean_usd=total/len(block),
                    turnover_usd=turnover,
                    bp_total_turnover=total/turnover*1e4 if turnover>0 else np.nan,
                    bp_mature_turnover=(total/mature_notional*1e4
                        if mature_notional>0 and metric in ('mature_spread_capture_usd',
                                                           'markout_60_usd',
                                                           'mature_fee_pnl_usd',
                                                           'mature_fill_edge_60_usd')
                        else np.nan),
                    fills=int(block.fills.sum()),
                    mature_60_fills=int(block.mature_60_fills.sum()),
                    mature_60_notional=mature_notional,
                    mature_notional_fraction=mature_notional/turnover if turnover>0 else np.nan))
    return pd.DataFrame(groups)


def alternate_maker_fee(fills,mature_mask,fee_bp=2.):
    """Replace maker fees along an unchanged path; taker fees remain recorded."""
    maker=np.asarray(fills['maker'],dtype=bool)
    fee=np.asarray(fills['fee'],dtype=float)
    new=np.where(maker,np.asarray(fills['qty'])*np.asarray(fills['price'])*fee_bp/1e4,fee)
    difference=fee-new  # PnL effect: fee is a signed cost.
    return float(difference.sum()),float(difference[np.asarray(mature_mask,dtype=bool)].sum())


def run():
    records=collect_ledgers();OUT.mkdir(parents=True,exist_ok=True)
    rows=[];alternate=[]
    # Sort by day and market so both strategies share the same raw BBO read.
    records.sort(key=lambda d:(d['day'],d['market'],d['cadence_ms']))
    cached_key=None;event=mid=None
    for d in records:
        key=(d['day'],d['market'])
        if key!=cached_key:
            event,mid=native_book(*key);cached_key=key
        frame=pd.read_parquet(d['_fills_path'])
        f={k:frame[k].to_numpy() for k in ('time_us','side','qty','price','fee','maker')}
        start,end=date_bounds(d['day'])
        eq0=float(d['start_cash'])+float(d['start_inventory'])*float(d['start_mid'])
        eq1=float(d['end_cash'])+float(d['end_inventory'])*float(d['end_mid'])
        if abs((eq1-eq0)-float(d['pnl_usd']))>1e-6:
            raise AssertionError(f'Input account balance mismatch: {key}')
        result=decompose_day(f,event,mid,start_us=start,end_us=end,
            start_inventory=float(d['start_inventory']),end_inventory=float(d['end_inventory']),
            start_mid=float(d['start_mid']),end_mid=float(d['end_mid']),
            start_equity=eq0,end_equity=eq1,funding_usd=float(d['funding_usd']))
        rows.append(dict(day=d['day'],market=d['market'],cadence_ms=int(d['cadence_ms']),
            partition=d['partition'],start_inventory=float(d['start_inventory']),
            end_inventory=float(d['end_inventory']),start_mid=float(d['start_mid']),
            end_mid=float(d['end_mid']),start_cash=float(d['start_cash']),
            end_cash=float(d['end_cash']),start_equity=eq0,end_equity=eq1,**result))
        stamp=np.asarray(f['time_us'],dtype=np.int64)
        m0,_=asof_book(event,mid,stamp,strict=True)
        mh,_=asof_book(event,mid,stamp+60_000_000)
        mature=np.isfinite(m0)&np.isfinite(mh)&(stamp+60_000_000<=end)
        delta,delta_mature=alternate_maker_fee(f,mature,2.)
        alt=dict(rows[-1]);alt['fee_pnl_usd']+=delta
        alt['total_pnl_usd']+=delta;alt['partial_60_fill_edge_usd']+=delta
        alt['mature_fee_pnl_usd']+=delta_mature
        alt['mature_fill_edge_60_usd']+=delta_mature
        alt['fixed_path_fee_pnl_difference_usd']=delta
        alternate.append(alt)
        print('DECOMPOSE',d['day'],d['market'],d['cadence_ms'],flush=True)
    daily=pd.DataFrame(rows)
    daily.to_csv(OUT/'daily.csv',index=False)
    summarize(daily).to_csv(OUT/'summary.csv',index=False)
    sensitivity=pd.DataFrame(alternate).sort_values(['market','cadence_ms','partition','day'])
    groups=sensitivity.groupby(['market','cadence_ms','partition'],sort=False)
    sensitivity['end_equity']+=groups.fixed_path_fee_pnl_difference_usd.cumsum()
    sensitivity['start_equity']+=groups.fixed_path_fee_pnl_difference_usd.cumsum()-sensitivity.fixed_path_fee_pnl_difference_usd
    sensitivity['identity_error_usd']=sensitivity.total_pnl_usd-(sensitivity.end_equity-sensitivity.start_equity)
    assert sensitivity.identity_error_usd.abs().max()<1e-6
    sensitivity.to_csv(OUT/'fee_sensitivity_2bp_daily.csv',index=False)
    summarize(sensitivity).to_csv(OUT/'fee_sensitivity_2bp_summary.csv',index=False)
    # Episode endpoints expose mark-to-mid residuals; no hypothetical liquidation.
    ends=daily.sort_values('day').groupby(['market','cadence_ms','partition'],as_index=False).tail(1)
    ends=ends.copy();ends['marked_inventory_usd']=ends.end_inventory*ends.end_mid
    ends[['market','cadence_ms','partition','day','end_inventory','end_mid',
          'marked_inventory_usd','end_cash','end_equity']].to_csv(
        OUT/'episode_terminal_inventory.csv',index=False)
    manifest=dict(horizon_s=60,price_reference='native HL BBO',
        fill_mid='last event strictly before fill; event age <=1 second',
        after_mid='last event at or before fill+60 seconds; age <=1 second',
        daily_mature='fill+60 seconds <= KST day end and both book prices valid',
        cross_boundary='60-second diagnostic may use next day; daily decomposition reports it as immature',
        funding='signed daily recorded settlement cashflow',
        denominator='sum of absolute execution notional, both maker and taker, no doubling',
        residual='fill-level virtual reconciliation, not actual post-60s inventory PnL',
        fee_sensitivity='replace maker fee only by +2bp on exact same fills; taker fee unchanged; assumes fees do not affect quotes, sizes, positions or margin',
        episode='IS/OOS separately reset and ALL sums independent episodes; terminal inventories marked to mid without liquidation assumption',
        days=len(daily),max_abs_identity_error_usd=float(daily.identity_error_usd.abs().max()))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


class DecompositionTest(unittest.TestCase):
    def test_asof_equal_timestamp_selection(self):
        stamp=np.array([1_000_000,10_000_000,10_000_000])
        px=np.array([99.,100.,101.]);q=np.array([10_000_000])
        strict,_=asof_book(stamp,px,q,strict=True,max_age_us=10_000_000)
        inclusive,_=asof_book(stamp,px,q,max_age_us=10_000_000)
        self.assertAlmostEqual(strict[0],99.)
        self.assertAlmostEqual(inclusive[0],101.)

    def test_mature_and_boundary_censored(self):
        us=1_000_000
        f=dict(time_us=np.array([10,100])*us,side=np.array([1,-1]),
               qty=np.array([2.,1.]),price=np.array([99.,103.]),fee=np.array([.01,.02]))
        event=np.array([9_900_000,70*us,99_900_000,160*us])
        mid=np.array([100.,101.,101.,104.])
        result=decompose_day(f,event,mid,start_us=0,end_us=120*us,
            start_inventory=1,end_inventory=2,start_mid=100,end_mid=102,
            start_equity=100,end_equity=108.97,funding_usd=0)
        self.assertAlmostEqual(result['spread_capture_usd'],4.)
        self.assertAlmostEqual(result['markout_60_usd'],2.)
        self.assertAlmostEqual(result['mature_reconciliation_usd'],2.)
        self.assertAlmostEqual(result['immature_reconciliation_usd'],-1.)
        self.assertEqual(result['mature_60_fills'],1)
        self.assertEqual(result['cross_boundary_60_fills'],1)
        self.assertAlmostEqual(result['identity_error_usd'],0.)

    def test_missing_and_strict_before(self):
        f=dict(time_us=np.array([10_000_000]),side=np.array([1]),qty=np.array([1.]),
               price=np.array([100.]),fee=np.array([0.]))
        # An equal timestamp cannot be used to explain the fill. The only
        # earlier book is older than the stated freshness threshold.
        result=decompose_day(f,np.array([1_000_000,10_000_000,70_000_000]),
            np.array([99.,100.,101.]),start_us=0,end_us=120_000_000,
            start_inventory=0,end_inventory=1,start_mid=100,end_mid=102,
            start_equity=0,end_equity=2,funding_usd=0)
        self.assertEqual(result['no_fill_mid_fills'],1)
        self.assertAlmostEqual(result['no_fill_mid_contribution_usd'],2.)
        self.assertEqual(result['mature_60_fills'],0)

    def test_fee_rebate_and_funding(self):
        f=dict(time_us=np.array([10_000_000]),side=np.array([-1]),qty=np.array([1.]),
               price=np.array([101.]),fee=np.array([-.1]))
        result=decompose_day(f,np.array([9_900_000,70_000_000]),np.array([100.,99.]),
            start_us=0,end_us=120_000_000,start_inventory=0,end_inventory=-1,
            start_mid=100,end_mid=98,start_equity=0,end_equity=3.3,funding_usd=.2)
        self.assertAlmostEqual(result['fee_pnl_usd'],.1)
        self.assertAlmostEqual(result['funding_usd'],.2)
        self.assertAlmostEqual(result['identity_error_usd'],0.)

    def test_maker_fee_sensitivity_preserves_taker(self):
        f=dict(maker=np.array([True,False]),qty=np.array([1.,1.]),
               price=np.array([100.,200.]),fee=np.array([-.0003,.00576]))
        whole,mature=alternate_maker_fee(f,np.array([True,False]))
        self.assertAlmostEqual(whole,-.0203)
        self.assertAlmostEqual(mature,-.0203)

    def test_random_signed_fills_reconcile(self):
        rng=np.random.default_rng(89)
        time=np.sort(rng.integers(2,198,size=200))*1_000_000
        qty=rng.uniform(.01,2.,len(time));side=rng.choice([-1,1],len(time))
        price=rng.uniform(95.,105.,len(time));fee=rng.uniform(-.01,.02,len(time))
        f=dict(time_us=time,side=side,qty=qty,price=price,fee=fee)
        event=np.arange(0,262)*1_000_000
        mids=100+np.sin(np.arange(len(event))/20)
        q0=1.4;qend=q0+np.sum(side*qty);startmid=100.;endmid=float(mids[200])
        cash_delta=-np.sum(side*qty*price+fee);funding=-.031
        expected=cash_delta+qend*endmid-q0*startmid+funding
        result=decompose_day(f,event,mids,start_us=0,end_us=200_000_000,
            start_inventory=q0,end_inventory=qend,start_mid=startmid,end_mid=endmid,
            start_equity=0,end_equity=expected,funding_usd=funding)
        self.assertAlmostEqual(result['identity_error_usd'],0.,places=8)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.test:unittest.main(argv=['test'])
    elif args.run:run()
    else:parser.error('Use --test or --run')
