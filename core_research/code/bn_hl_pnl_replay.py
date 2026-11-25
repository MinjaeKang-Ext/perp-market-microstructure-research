"""Event-timed one-order-per-side HL maker replay for causal predictions.

Order/cancel effectiveness are scenario assumptions, 440ms/750ms. Public trade
and BBO event clocks need not be synchronized with received decision clocks.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numba import njit

from mm_stage1 import funding6
from work import fv_mm_inputs as I
from work.fv_mm_replay import (tick_at_price, project_prices, _queue_seed)
from work.run_fv_mm import MISSING_DAYS

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('BN_HL_PNL_ROOT',ROOT/'out_bn_hl_pnl'))
KST=ZoneInfo('Asia/Seoul')
START=datetime(2026,10,15,tzinfo=KST)
END=datetime(2026,11,2,tzinfo=KST)
SUBMIT_US=440_000
CANCEL_US=750_000
MAKER_BP=-.03


@njit(cache=True,inline='always')
def record(array,i,a,b,c,d,e,f,g):
    array[i,0]=a;array[i,1]=b;array[i,2]=c;array[i,3]=d
    array[i,4]=e;array[i,5]=f;array[i,6]=g


@njit(cache=True)
def simulate(grid,bid,ask,mid,pbid,pask,valid,ab,aa,abq,aaq,
             tt,tside,tp,tq,le,lb,lqb,la,lqa,code,lot,cash0,q0,allow):
    n=len(grid); event_end=grid[-1]+(grid[1]-grid[0])
    # event columns: native/us, kind, side, price, quantity, cash, inventory.
    events=np.empty((n*8+len(tt)*2+20,7),np.float64)
    fills=np.empty((len(tt)*2+20,7),np.float64)
    ne=0; nf=0
    # 0 idle, 1 awaiting activation, 2 live, 3 cancellation in flight.
    state=np.zeros(2,np.int64)
    price=np.zeros(2); qty=np.zeros(2); queue=np.zeros(2)
    arrival=np.full(2,np.int64(2**62)); cancel=np.full(2,np.int64(2**62))
    chosen=np.zeros(2,np.int64)
    cash=cash0; q=q0
    tptr=np.searchsorted(tt,grid[0],side='left')
    # submit, activate, reject, cancel request, cancel effective, fill.
    stats=np.zeros(17,np.float64)
    for i in range(n+1):
        now=grid[i] if i<n else event_end
        while True:
            nt=tt[tptr] if tptr<len(tt) else np.int64(2**62)
            nc0=cancel[0] if state[0]==3 else np.int64(2**62)
            nc1=cancel[1] if state[1]==3 else np.int64(2**62)
            na0=arrival[0] if state[0]==1 else np.int64(2**62)
            na1=arrival[1] if state[1]==1 else np.int64(2**62)
            nxt=min(nt,nc0,nc1,na0,na1)
            if nxt>=event_end or nxt>now: break
            # Same timestamp ordering: trades, cancel effectiveness, activation.
            if nt==nxt:
                j=0 if tside[tptr]<0 else 1
                if (state[j]==2 or state[j]==3) and qty[j]>1e-10:
                    side=1 if j==0 else -1
                    eligible=(tp[tptr]<=price[j]+1e-8 if j==0 else tp[tptr]>=price[j]-1e-8)
                    if eligible:
                        through=(tp[tptr]<price[j]-1e-8 if j==0 else tp[tptr]>price[j]+1e-8)
                        if through: queue[j]=0.
                        available=tq[tptr]
                        ahead=min(queue[j],available)
                        available-=ahead;queue[j]-=ahead
                        take=math_floor_lot(min(qty[j],max(0.,available)),lot)
                        if take>1e-10:
                            fee=take*price[j]*MAKER_BP/10000.
                            cash-=side*take*price[j]+fee
                            q+=side*take; qty[j]-=take
                            assert abs(q)<=1+1e-7
                            stats[14]=max(stats[14],abs(q))
                            record(fills,nf,nxt,side,take,price[j],fee,cash,q);nf+=1
                            record(events,ne,nxt,6,side,price[j],take,cash,q);ne+=1
                            stats[5]+=1;stats[6]+=take*price[j];stats[7]+=fee
                            if qty[j]<=1e-10 and state[j]==2:state[j]=0
                tptr+=1
            elif nc0==nxt or nc1==nxt:
                j=0 if nc0==nxt else 1;side=1 if j==0 else -1
                record(events,ne,nxt,5,side,price[j],qty[j],cash,q);ne+=1
                stats[4]+=1;state[j]=0;qty[j]=0.;cancel[j]=np.int64(2**62)
            else:
                j=0 if na0==nxt else 1; side=1 if j==0 else -1;k=chosen[j]
                good=(np.isfinite(ab[k]) and np.isfinite(aa[k]) and
                      np.isfinite(abq[k]) and np.isfinite(aaq[k]) and
                      0<ab[k]<aa[k] and abq[k]>=0 and aaq[k]>=0)
                crosses=(price[j]>=aa[k]-1e-8 if j==0 else price[j]<=ab[k]+1e-8)
                if not good or crosses:
                    record(events,ne,nxt,3,side,price[j],qty[j],cash,q);ne+=1
                    stats[2]+=1;state[j]=0;qty[j]=0.
                    if not good:stats[15]+=1
                else:
                    queue[j],l2age,unknown=_queue_seed(price[j],side,ab[k],aa[k],
                        abq[k],aaq[k],nxt,le,lb,lqb,la,lqa,1)
                    state[j]=2;stats[1]+=1;stats[8]+=unknown
                    if l2age>=0:
                        stats[9]+=1;stats[10]+=l2age;stats[11]=max(stats[11],l2age)
                    record(events,ne,nxt,2,side,price[j],qty[j],cash,q);ne+=1
                arrival[j]=np.int64(2**62)
        if i==n:break
        isvalid=(allow and now<event_end-3_000_000 and valid[i] and
                 np.isfinite(bid[i]) and np.isfinite(ask[i]) and 0<bid[i]<ask[i] and
                 np.isfinite(mid[i]) and mid[i]>0 and np.isfinite(pbid[i]) and
                 np.isfinite(pask[i]) and pbid[i]>0 and pask[i]>0)
        if valid[i] and np.isfinite(pbid[i]) and np.isfinite(pask[i]):
            stats[12]+=1
            if pbid[i]>pask[i]:stats[13]+=1
        if isvalid:
            fv=(pbid[i]+pask[i])/2.
            tick=tick_at_price(mid[i],code)
            skew=tick*q
            half=(ask[i]-bid[i])/2.
            wantb,wanta=project_prices(fv-half-skew,fv+half-skew,bid[i],ask[i],code)
            isvalid=(np.isfinite(wantb) and np.isfinite(wanta) and 0<wantb<wanta)
        for j in range(2):
            side=1 if j==0 else -1
            want=wantb if j==0 and isvalid else (wanta if isvalid else 0.)
            desired=math_floor_lot(min(1.,max(0.,1.-side*q)),lot) if isvalid else 0.
            if state[j]==2 and (not isvalid or desired<=1e-10 or
                   abs(want-price[j])>1e-8 or abs(desired-qty[j])>1e-8):
                state[j]=3;cancel[j]=now+CANCEL_US;stats[3]+=1
                record(events,ne,now,4,side,price[j],qty[j],cash,q);ne+=1
            elif state[j]==0 and isvalid and desired>1e-10:
                if want*desired < 10.-1e-8:
                    stats[16]+=1
                else:
                    state[j]=1;arrival[j]=now+SUBMIT_US;price[j]=want;qty[j]=desired
                    chosen[j]=i;stats[0]+=1
                    record(events,ne,now,1,side,want,desired,cash,q);ne+=1
        stats[14]=max(stats[14],abs(q))
    assert state[0]==0 and state[1]==0, 'live or in-flight order across midnight'
    return fills[:nf].copy(),events[:ne].copy(),stats,cash,q


@njit(cache=True)
def math_floor_lot(qty,lot):
    return np.floor(qty/lot+1e-8)*lot


def arrival_books(market,day,grid):
    """Native best book as-of each possible submit effectiveness timestamp."""
    result=np.full((len(grid),4),np.nan)
    for h in range(24):
        hour=day+timedelta(hours=h)
        event,recv,seq,bid,ask,bq,aq=I.read_bbo_hour(market,hour)
        order=np.lexsort((recv,event));event=event[order]
        start=int(hour.timestamp()*1_000_000);end=start+3_600_000_000
        ids=np.flatnonzero((grid>=start)&(grid<end))
        query=grid[ids]+SUBMIT_US
        at=np.searchsorted(event,query,side='right')-1
        good=at>=0
        safe=np.maximum(at,0)
        good &= (query-event[safe]>=0)&(query-event[safe]<=1_000_000)
        for col,v in enumerate((bid,ask,bq,aq)):
            result[ids[good],col]=v[order[at[good]]]
        assert np.all(event[at[good]]<=query[good])
    return result


def funding(day,market,fills,q0):
    rows=json.loads((ROOT/'work/funding_momentum'/f'{market}.json').read_text())
    start=int(day.timestamp()*1e6);end=int((day+timedelta(days=1)).timestamp()*1e6)
    times=fills['time_us'].to_numpy(np.int64)
    cache={}; cashflows=[]
    for event in rows:
        stamp=int(event['time'])*1000
        if not start<=stamp<end:continue
        i=np.searchsorted(times,stamp,side='left')-1 # funding before simultaneous fill
        q=float(fills['inventory'].iloc[i]) if i>=0 else q0
        rate=float(event['fundingRate']); px=None;oracle_age_s=None;oracle_source=None;oracle_time_us=None
        if abs(q)>1e-10:
            px,oracle_ns,oracle_source=funding6._oracle_before(ROOT/'perp_data/trading_reference_data/processed',
                                          'xyz_'+market.split('_')[-1],stamp*1000,cache)
            if px is None or not np.isfinite(px) or px<=0:
                raise ValueError(f'Funding oracle missing: {market} {day} {stamp}')
            oracle_age_s=(stamp*1000-oracle_ns)/1e9
            oracle_time_us=oracle_ns//1000
        amount=-q*px*rate if px is not None else 0.
        cashflows.append(dict(time_us=stamp,position=q,rate=rate,oracle_proxy=px,
                              oracle_age_s=oracle_age_s,oracle_time_us=oracle_time_us,
                              oracle_time_source=oracle_source,
                              cashflow=amount))
    if not cashflows:raise ValueError(f'Missing funding history: {market} {day}')
    return cashflows


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def run_one(market,day,cadence,pred,arrival,public,l2,state,allow):
    grid=pred['grid_us'].astype(np.int64);n=len(grid)
    assert np.all(np.diff(grid)==cadence*1000)
    code=1 if market.endswith('SKHX') else 2
    lot=.001 if code==1 else .01
    cash0=float(state.get('cash',0));q0=float(state.get('inventory',0))
    fills,events,stats,cash,q=simulate(grid,*[pred[k].astype(float) for k in ('bid','ask','mid','model_bid','model_ask')],
        pred['decision_valid'].astype(bool),*[arrival[:,i] for i in range(4)],
        public['tt_us'].astype(np.int64),public['tside'].astype(np.int8),
        public['tprice'].astype(float),public['tqty'].astype(float),
        l2['l2_event_us'].astype(np.int64),*[l2[k].astype(float) for k in
        ('l2_bid_px','l2_bid_qty','l2_ask_px','l2_ask_qty')],code,lot,cash0,q0,allow)
    f=pd.DataFrame(fills,columns=['time_us','side','qty','price','fee','cash','inventory'])
    f['time_us']=f.time_us.astype('int64');f['side']=f.side.astype('int8');f['maker']=True
    e=pd.DataFrame(events,columns=['time_us','kind','side','price','qty','cash','inventory'])
    e['time_us']=e.time_us.astype('int64');e['kind']=e.kind.astype('int8');e['side']=e.side.astype('int8')
    fl=funding(day,market,f,q0); funding_usd=float(sum(r['cashflow'] for r in fl));cash+=funding_usd
    funding_times=np.array([r['time_us'] for r in fl],dtype=np.int64)
    funding_cumulative=np.r_[0.,np.cumsum([r['cashflow'] for r in fl])]
    # Cash columns include funding that settled at or before each event;
    # at a timestamp shared with a fill, funding is booked first.
    for ledger in (f,e):
        ledger['cash']+=funding_cumulative[np.searchsorted(funding_times,
            ledger.time_us.to_numpy(np.int64),side='right')]
    validmid=np.flatnonzero(np.isfinite(pred['mid'])&(pred['mid']>0))
    if not len(validmid):raise ValueError('No daily midpoint')
    first=float(pred['mid'][validmid[0]]);last=float(pred['mid'][validmid[-1]])
    startmid=float(state.get('last_mid',first))
    end_us=int(grid[-1]+cadence*1000)
    last_i=int(validmid[-1]); last_recv=int(pred['hl_recv_stamp_us'][last_i])
    end_mark_receive_age_s=(end_us-last_recv)/1e6 if last_recv>=0 else None
    last_event_age=float(pred['hl_event_age_sec'][last_i])
    end_mark_event_age_s=(last_event_age+(end_us-int(grid[last_i]))/1e6
                          if np.isfinite(last_event_age) else None)
    pnl=cash+q*last-(cash0+q0*startmid)
    verify_cash=cash0-float(np.sum(f.side*f.qty*f.price+f.fee))+funding_usd
    verify_q=q0+float(np.sum(f.side*f.qty))
    assert abs(verify_cash-cash)<1e-6 and abs(verify_q-q)<1e-8
    keys=('submitted','activated','rejected','cancel_requested','cancel_effective',
          'fill_events','turnover_usd','fee_usd','unknown_queue','l2_seeds',
          'l2_age_sum_s','l2_age_max_s','valid_forecast_rows','crossed_forecasts','max_abs_inventory',
          'arrival_book_unavailable','submit_below_min_notional')
    metrics=dict(zip(keys,stats.tolist()))
    metrics.update(day=day.strftime('%Y-%m-%d'),market=market,cadence_ms=cadence,
        partition=str(pred['partition'].item()),start_cash=cash0,start_inventory=q0,
        start_mid=startmid,end_cash=cash,end_inventory=q,end_mid=last,
        funding_usd=funding_usd,pnl_usd=pnl,allow_quotes=allow,
        funding_events=fl,fees_bp=MAKER_BP,close_guard_s=3,
        order_submit_effect_us=SUBMIT_US,cancel_effect_us=CANCEL_US,
        order_submit_cancel_assumption='measured ACK medians as effectiveness delay proxy, not one-way latency measurements',
        max_position_contracts=1,order_max_contracts=1,inventory_skew_ticks_per_contract=1,
        min_new_order_notional_usd=10,max_arrival_book_native_age_s=1,
        queue_model='displayed arrival best or latest L2 exact price; unknown=infinite until price through',
        fill_model='public aggressive trades allocated once at our posted limit price, native timestamp',
        end_mark_receive_age_s=end_mark_receive_age_s,
        end_mark_event_age_s=end_mark_event_age_s,
        end_mark_stale=(end_mark_event_age_s is None or end_mark_event_age_s>1),
        clock_tie_order='trade before cancel effectiveness before activation; funding before simultaneous fill')
    folder=OUT/'ledgers'/market/str(cadence);folder.mkdir(parents=True,exist_ok=True)
    ds=day.strftime('%Y-%m-%d')
    f.to_parquet(folder/f'{ds}.fills.parquet',index=False)
    e.to_parquet(folder/f'{ds}.events.parquet',index=False)
    (folder/f'{ds}.daily.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    return dict(cash=cash,inventory=q,last_mid=last)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--market',choices=('hyperliquid_SKHX','hyperliquid_SKHY'),required=True)
    ap.add_argument('--start',default='2026-10-15');ap.add_argument('--end',default='2026-11-02')
    ap.add_argument('--cadence',type=int,choices=(500,1000),action='append')
    # Episode carry-over state is tiny; persisting it lets the 18 days be
    # replayed in several short processes without breaking the episode chain.
    ap.add_argument('--state')
    args=ap.parse_args();market=args.market;cadences=args.cadence or [500,1000]
    state_path=Path(args.state) if args.state else None
    states={c:{} for c in cadences}
    if state_path is not None and state_path.exists():
        states={c:v for c,v in ((int(k),v) for k,v in json.loads(state_path.read_text()).items())
                if c in cadences}
    for day in pd.date_range(args.start,args.end,inclusive='left',tz=KST).to_pydatetime():
        if day.strftime('%Y-%m-%d')=='2026-10-24':states={c:{} for c in cadences}
        preds={c:load_npz(OUT/'predictions'/market/f'{day:%Y-%m-%d}_{c}.npz') for c in cadences}
        master=min(cadences);native=arrival_books(market,day,preds[master]['grid_us'])
        public=load_npz(ROOT/'out_fv_mm/inputs'/market/f'{day:%Y-%m-%d}.npz')
        l2=load_npz(ROOT/'out_fv_mm/inputs'/market/f'{day:%Y-%m-%d}.l2.npz')
        for c in cadences:
            if c==master:arr=native
            else:
                at=np.searchsorted(preds[master]['grid_us'],preds[c]['grid_us'])
                assert np.array_equal(preds[master]['grid_us'][at],preds[c]['grid_us'])
                arr=native[at]
            allow=day.strftime('%Y-%m-%d') not in MISSING_DAYS[market]
            states[c]=run_one(market,day,c,preds[c],arr,public,l2,states[c],allow)
            print('REPLAY',market,day.date(),c,states[c],flush=True)
        if state_path is not None:
            state_path.write_text(json.dumps({str(c):states[c] for c in cadences})+'\n')


if __name__=='__main__':main()
