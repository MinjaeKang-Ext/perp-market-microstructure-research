"""Reconstruct genuine walk-forward R² for selected BN→HL best-quote models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E


CASES=((0,1000,'joint'),(1,1000,'micro'),(0,500,'mid'),(1,500,'mid'))


def main():
    models={key:E.Moments.empty(10,2) for key in CASES}
    rows=[]
    for day_index,day in enumerate(C.DAYS):
        for pair_index,(source_name,target_name) in enumerate(C.PAIRS):
            source,target=C.load(source_name,day),C.load(target_name,day)
            for case in CASES:
                pi,cadence,family=case
                if pi!=pair_index:continue
                at,before,x,base_valid,source_valid=C.assemble(source,target,cadence,0)
                y,valid=C.future_target(target,at,cadence,base_valid)
                xs=x[valid];ys=y[valid]
                moment=models[case]
                if day_index and len(xs):
                    a,b=moment.solve(list(range(5))+C.GROUPS[family])
                    a0,b0=moment.solve(list(range(5)))
                    err=ys-(a+xs@b);own=ys-(a0+xs@b0)
                    for j,side in enumerate(('bid','ask')):
                        q=ys[:,j];e=err[:,j];o=own[:,j]
                        rows.append((day,source_name,target_name,cadence,family,side,len(q),
                            float(q.sum()),float(q@q),float(e@e),float(o@o),
                            float(np.abs(e).sum()),float(np.abs(o).sum())))
                moment.update(xs,ys)
            print('R2',day,source_name,target_name,flush=True)
    cols=('day','source','target','cadence_ms','family','side','n','sum_y','sum_y2',
          'sse','own_sse','sae','own_sae')
    daily=pd.DataFrame(rows,columns=cols)
    # Exact paired reproduction of every daily forecast MAE from previous run.
    old=pd.read_parquet(C.OUT/'forecast_daily.parquet')
    old=old[old.phase_ms.eq(0)&old.horizon_ms.eq(old.cadence_ms)]
    check=daily.merge(old,on=['day','source','target','cadence_ms','family','side'],
                      how='left',validate='one_to_one')
    assert len(check)==len(daily) and not check.model_abs_sum.isna().any()
    assert (check.n_x==check.n_y).all()
    err=float(np.max(np.abs(check.sae-check.model_abs_sum)))
    err0=float(np.max(np.abs(check.own_sae-check.own_abs_sum)))
    assert err<1e-7 and err0<1e-7,(err,err0)
    daily.to_csv(C.OUT/'r2_daily.csv',index=False)
    keys=['source','target','cadence_ms','family','side']
    out=daily.groupby(keys,as_index=False)[['n','sum_y','sum_y2','sse','own_sse','sae','own_sae']].sum()
    out['sst_pooled_mean']=out.sum_y2-out.sum_y**2/out.n
    out['r2_pooled_mean']=1-out.sse/out.sst_pooled_mean
    out['r2_vs_zero_return']=1-out.sse/out.sum_y2
    out['own_r2_pooled_mean']=1-out.own_sse/out.sst_pooled_mean
    out['sse_improvement_vs_own']=1-out.sse/out.own_sse
    out.to_csv(C.OUT/'r2_summary.csv',index=False)
    (C.OUT/'r2_verification.json').write_text(json.dumps(dict(
        independent_reproduction='past-day refit and exact daily MAE match to original',
        cases=len(CASES),days=len(C.DAYS)-1,side_rows=len(daily),
        max_model_sae_difference=err,max_own_sae_difference=err0,
        r2='1 - OOS SSE / sum((OOS actual - pooled OOS actual mean)^2)',
        zero='1 - OOS SSE / sum(actual^2)',
        incremental='1 - OOS SSE / OOS HL-own-model SSE'),indent=2)+'\n')
    print(out.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
