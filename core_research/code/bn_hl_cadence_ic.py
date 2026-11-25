"""Walk-forward IC and rank IC for selected BN to HL quote predictors."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work.bn_hl_cadence_r2 import CASES


def moments(x,y):
    return np.array((len(x),x.sum(),y.sum(),x@x,y@y,x@y),dtype=float)


def pearson(m):
    n,sx,sy,sxx,syy,sxy=np.moveaxis(np.asarray(m),-1,0)
    with np.errstate(invalid='ignore',divide='ignore'):
        vx=sxx-sx*sx/n;vy=syy-sy*sy/n
        return np.where((n>2)&(vx>0)&(vy>0),(sxy-sx*sy/n)/np.sqrt(vx*vy),np.nan)


def rank_ic(x,y):
    return float(pearson(moments(rankdata(x,method='average'),
                                 rankdata(y,method='average'))))


def main():
    models={case:E.Moments.empty(10,2) for case in CASES}
    vectors={case:[] for case in CASES};rows=[]
    for di,day in enumerate(C.DAYS):
        for pi,(source_name,target_name) in enumerate(C.PAIRS):
            src,tgt=C.load(source_name,day),C.load(target_name,day)
            for case in CASES:
                pair,cadence,family=case
                if pair!=pi: continue
                at,before,x,valid0,_=C.assemble(src,tgt,cadence,0)
                y,valid=C.future_target(tgt,at,cadence,valid0)
                xs=x[valid];ys=y[valid];model=models[case]
                if di and len(xs):
                    a,b=model.solve(list(range(5))+C.GROUPS[family])
                    a0,b0=model.solve(list(range(5)))
                    pred=a+xs@b;own=a0+xs@b0
                    vectors[case].append((ys,pred,own))
                    for j,side in enumerate(('bid','ask')):
                        actual=ys[:,j]; model_pred=pred[:,j]; own_pred=own[:,j]
                        m=moments(model_pred,actual); m0=moments(own_pred,actual)
                        err=actual-model_pred;err0=actual-own_pred
                        rows.append((day,source_name,target_name,cadence,family,side,len(actual),
                            float(pearson(m)),float(pearson(m0)),
                            rank_ic(model_pred,actual),rank_ic(own_pred,actual),
                            float(np.mean(actual==0)),float(err@err),float(err0@err0),
                            float(np.abs(err).sum()),float(np.abs(err0).sum()),
                            *m,*m0))
                model.update(xs,ys)
            print('IC',day,source_name,target_name,flush=True)
    cols=('day','source','target','cadence_ms','family','side','n',
          'ic','own_ic','rank_ic','own_rank_ic','target_zero_fraction',
          'sse','own_sse','sae','own_sae',
          'm_n','m_sx','m_sy','m_sxx','m_syy','m_sxy',
          'o_n','o_sx','o_sy','o_sxx','o_syy','o_sxy')
    daily=pd.DataFrame(rows,columns=cols)
    old=pd.read_csv(C.OUT/'r2_daily.csv')
    check=daily.merge(old,on=['day','source','target','cadence_ms','family','side'],
                      validate='one_to_one',suffixes=('','_old'))
    assert len(check)==len(old)==len(daily) and (check.n==check.n_old).all()
    diffs={key:float(np.max(np.abs(check[key]-check[key+'_old'])))
           for key in ('sse','own_sse','sae','own_sae')}
    assert max(diffs.values())<1e-6,diffs
    daily.to_csv(C.OUT/'ic_daily.csv',index=False)
    weights=np.random.default_rng(20261108).multinomial(17,np.ones(17)/17,size=5000)
    summaries=[]
    for case in CASES:
        pi,cadence,family=case;source_name,target_name=C.PAIRS[pi]
        pieces=vectors[case]
        assert len(pieces)==17
        y=np.concatenate([z[0] for z in pieces]);pred=np.concatenate([z[1] for z in pieces]);own=np.concatenate([z[2] for z in pieces])
        for j,side in enumerate(('bid','ask')):
            sub=daily[(daily.source==source_name)&(daily.cadence_ms==cadence)&
                      (daily.family==family)&(daily.side==side)].sort_values('day')
            mm=sub[['m_n','m_sx','m_sy','m_sxx','m_syy','m_sxy']].to_numpy()
            oo=sub[['o_n','o_sx','o_sy','o_sxx','o_syy','o_sxy']].to_numpy()
            bc=pearson(weights@mm);bo=pearson(weights@oo)
            pooled=float(pearson(mm.sum(axis=0)));pooled_own=float(pearson(oo.sum(axis=0)))
            summaries.append((source_name,target_name,cadence,family,side,len(y),
                pooled,*np.quantile(bc,[.025,.975]),pooled_own,
                *np.quantile(bc-bo,[.025,.975]),
                float(sub.ic.mean()),*np.quantile(weights@sub.ic.to_numpy()/17,[.025,.975]),
                float(sub.ic.std(ddof=1)),int((sub.ic>0).sum()),
                float(sub.own_ic.mean()),rank_ic(pred[:,j],y[:,j]),
                rank_ic(own[:,j],y[:,j]),float(sub.rank_ic.mean()),
                float(sub.rank_ic.std(ddof=1)),int((sub.rank_ic>0).sum()),
                float(np.mean(y[:,j]==0))))
    cols=('source','target','cadence_ms','family','side','n','ic_pooled',
          'ic_boot_low','ic_boot_high','own_ic_pooled','diff_boot_low','diff_boot_high',
          'ic_daily_mean','ic_daily_boot_low','ic_daily_boot_high',
          'ic_daily_sd','ic_positive_days','own_ic_daily_mean',
          'rank_ic_pooled','own_rank_ic_pooled','rank_ic_daily_mean','rank_ic_daily_sd',
          'rank_ic_positive_days','target_zero_fraction')
    result=pd.DataFrame(summaries,columns=cols)
    result.to_csv(C.OUT/'ic_summary.csv',index=False)
    (C.OUT/'ic_verification.json').write_text(json.dumps(dict(cases=len(CASES),
        n_evaluation_days=17,daily_rows=len(daily),max_loss_errors=diffs,
        pearson='correlation of OOS predictions and realized log returns',
        rank='Spearman average ranks for ties, pooled and unweighted daily mean',
        intervals='5000 paired bootstrap resamples of 17 whole days, percentile, pointwise',
        zero_returns='included in both Pearson and rank'),indent=2)+'\n')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
