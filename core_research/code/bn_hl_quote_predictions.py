"""Create best-quote predictions for causal replay; never gate orders on labels.

IS predictions are retrospective fits on the complete first nine days.
OOS predictions use expanding moments from completed prior days only.
The only saved decision mask is built from current and previous observations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E

# BN_HL_PNL_ROOT redirects every artifact of one variant; BN_HL_PNL_FAMILY
# forces a single signal family on all four cases. Defaults reproduce the
# original per-case family selection in out_bn_hl_pnl.
BASE=Path(os.environ.get('BN_HL_PNL_ROOT',C.ROOT/'out_bn_hl_pnl'))
OUT=BASE/'predictions'
FAMILY=os.environ.get('BN_HL_PNL_FAMILY')
CASES=((0,500,'mid'),(0,1000,'joint'),(1,500,'mid'),(1,1000,'micro'))
if FAMILY:
    if FAMILY not in C.GROUPS:raise ValueError(f'Unknown family {FAMILY}')
    CASES=tuple((pi,cadence,FAMILY) for pi,cadence,_ in CASES)
TRAIN_DAYS=9


def load_case(case,day):
    pi,cadence,_=case
    source_name,target_name=C.PAIRS[pi]
    source,target=C.load(source_name,day),C.load(target_name,day)
    at,_,x,base_valid,_=C.assemble(source,target,cadence,0)
    y,label_valid=C.future_target(target,at,cadence,base_valid)
    return source,target,at,x,base_valid,y,label_valid


def fit(moment,family):
    forecast=moment.solve(list(range(C.OWN))+C.GROUPS[family])
    baseline=moment.solve(list(range(C.OWN)))
    assert forecast is not None and baseline is not None
    return forecast,baseline


def evaluate_and_write(case,day,partition,moment,training_end):
    pi,cadence,family=case
    source_name,target_name=C.PAIRS[pi]
    source,target,at,x,base_valid,y,label_valid=load_case(case,day)
    (a,b),(a0,b0)=fit(moment,family)
    predicted=a+x@b; own=a0+x@b0
    ok=base_valid&np.isfinite(predicted).all(axis=1)&np.isfinite(own).all(axis=1)
    with np.errstate(over='ignore',invalid='ignore'):
        bid_pred=target['bid'][at]*np.exp(predicted[:,0]/10000.)
        ask_pred=target['ask'][at]*np.exp(predicted[:,1]/10000.)
        bid_own=target['bid'][at]*np.exp(own[:,0]/10000.)
        ask_own=target['ask'][at]*np.exp(own[:,1]/10000.)
    ok &= np.isfinite(bid_pred)&np.isfinite(ask_pred)&np.isfinite(bid_own)&np.isfinite(ask_own)
    grid=target['grid_us'][at]
    assert np.all(source['recv_stamp_us'][at][ok]<=grid[ok])
    assert np.all(target['recv_stamp_us'][at][ok]<=grid[ok])
    # The observed state and forecast inputs are all known by t. Future label
    # availability is deliberately not stored in the replay decision stream.
    dest=OUT/target_name
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{day}_{cadence}.npz',grid_us=grid,
        bid=target['bid'][at],ask=target['ask'][at],bq=target['bq'][at],aq=target['aq'][at],
        mid=target['mid'][at],hl_event_age_sec=target['age_sec'][at],
        hl_recv_stamp_us=target['recv_stamp_us'][at],
        bn_recv_stamp_us=source['recv_stamp_us'][at],bn_event_age_sec=source['age_sec'][at],
        model_bid_bp=predicted[:,0],model_ask_bp=predicted[:,1],
        own_bid_bp=own[:,0],own_ask_bp=own[:,1],
        model_bid=bid_pred,model_ask=ask_pred,own_bid=bid_own,own_ask=ask_own,
        decision_valid=ok,partition=partition,training_end=training_end,
        model_family=family,cadence_ms=cadence,
        coefficient_intercept=a,coefficient_slope=b,
        own_intercept=a0,own_slope=b0)
    scoring=label_valid&ok
    rows=[]
    for j,side in enumerate(('bid','ask')):
        e=y[scoring,j]-predicted[scoring,j]
        e0=y[scoring,j]-own[scoring,j]
        rows.append((day,source_name,target_name,cadence,family,partition,side,
            int(ok.sum()),int(scoring.sum()),float(e@e),float(e0@e0),
            float(np.abs(e).sum()),float(np.abs(e0).sum())))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    moments={case:E.Moments.empty(10,2) for case in CASES}
    for day in C.DAYS[:TRAIN_DAYS]:
        for case in CASES:
            _,_,_,x,_,y,valid=load_case(case,day)
            moments[case].update(x[valid],y[valid])
        print('PRED_TRAIN',day,flush=True)
    rows=[]
    for day in C.DAYS[:TRAIN_DAYS]:
        for case in CASES:
            rows.extend(evaluate_and_write(case,day,'IS_retroactive',moments[case],C.DAYS[TRAIN_DAYS-1]))
        print('PRED_IS',day,flush=True)
    for di,day in enumerate(C.DAYS[TRAIN_DAYS:],start=TRAIN_DAYS):
        for case in CASES:
            rows.extend(evaluate_and_write(case,day,'OOS',moments[case],C.DAYS[di-1]))
            _,_,_,x,_,y,valid=load_case(case,day)
            moments[case].update(x[valid],y[valid])
        print('PRED_OOS',day,flush=True)
    cols=('day','source','target','cadence_ms','family','partition','side',
          'decision_n','evaluation_n','sse','own_sse','sae','own_sae')
    audit=pd.DataFrame(rows,columns=cols)
    audit.to_csv(OUT.parent/'prediction_audit.csv',index=False)
    old=pd.read_csv(C.OUT/'r2_daily.csv')
    q=audit[audit.partition=='OOS']
    check=q.merge(old,on=['day','source','target','cadence_ms','family','side'],
                  suffixes=('','_old'),validate='one_to_one')
    # The cadence study only scored the family it selected per pair, so a forced
    # family may overlap it only partially. Validate the overlap, require some.
    assert len(check)>0 and (check.evaluation_n==check.n).all()
    if not FAMILY:assert len(check)==len(q)
    errors={col:float(np.max(np.abs(check[col]-check[col+'_old'])))
            for col in ('sse','own_sse','sae','own_sae')}
    assert max(errors.values())<1e-5,errors
    meta={'cases':[list(case) for case in CASES],'days':list(C.DAYS),
      'is_days':list(C.DAYS[:TRAIN_DAYS]),'oos_days':list(C.DAYS[TRAIN_DAYS:]),
      'is_policy':'retrospective fit on all 9 IS dates; not executable walk-forward performance',
      'oos_policy':'past dates only, latest fit day day-1',
      'price_prediction':'current quote * exp(predicted log return in bp / 10000)',
      'decision_mask':'base_valid and finite predicted prices at current t only',
      'future_label':'used only in past-day training and offline scoring; never saved as decision input',
      'deadline':'labels crossing day end excluded from training',
      'forced_family':FAMILY,'verification':errors,
      'oos_validation_rows':len(check),'oos_rows':len(q)}
    (OUT.parent/'prediction_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    main()
