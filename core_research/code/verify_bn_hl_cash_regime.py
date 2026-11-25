"""Independent spot checks for the conditional correlation calculation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cash_regime as R


def main():
    day='2026-10-15'
    source_name,target_name=R.PAIRS[0]
    source,target=R.C.load(source_name,day),R.C.load(target_name,day)
    labels=R.regime_labels(source['grid_us'])
    def at(h,m,s=0): return (h*3600+m*60+s)*10
    assert labels[at(0,0)]==1 and labels[at(4,59,59)]==1
    assert labels[at(5,0)]==2 and labels[at(9,0)]==0
    assert labels[at(15,29,59)]==0 and labels[at(15,30)]==2
    assert labels[at(22,30)]==1
    saturday=R.C.load(source_name,'2026-10-17')
    satlabels=R.regime_labels(saturday['grid_us'])
    assert satlabels[at(0,0)]==1 and satlabels[at(5,0)]==3
    atidx,x,_,source_valid=R.M.design(source,target,500,0)
    end=atidx+4
    safe=np.minimum(end,len(labels)-1)
    ret,ret_valid=R.M.returns100(target)
    case=(source_valid&(end<len(labels))&ret_valid[safe]&
          (labels[atidx]==0)&(labels[atidx-5]==0)&
          (labels[safe]==0)&(labels[safe-1]==0))
    raw=np.corrcoef(x[case,5],ret['bid'][safe[case]])[0,1]
    summary=pd.read_csv(R.OUT/'response_summary.csv')
    q=summary[(summary.source==source_name)&(summary.target==target_name)&
        (summary.cadence_ms==500)&(summary.phase_ms==0)&
        (summary.regime=='KR_cash')&(summary.feature=='mid')&
        (summary.side=='bid')&(summary.end_lag_ms==400)]
    # Pooled result differs from a one-day direct calculation. Rebuild the
    # one-day value from saved moments, which must agree with the raw arrays.
    part=pd.read_parquet(R.OUT/f'response_{day}.parquet')
    p=part[(part.source==source_name)&(part.target==target_name)&
        (part.cadence_ms==500)&(part.phase_ms==0)&
        (part.regime=='KR_cash')&(part.feature=='mid')&
        (part.side=='bid')&(part.end_lag_ms==400)]
    assert len(p)==1 and len(q)==1
    z=p.iloc[0]
    vx=z.sxx-z.sx*z.sx/z.n; vy=z.syy-z.sy*z.sy/z.n
    rebuilt=(z.sxy-z.sx*z.sy/z.n)/np.sqrt(vx*vy)
    assert int(z.n)==int(case.sum())
    assert abs(raw-rebuilt)<1e-10,(raw,rebuilt)
    audit=json.loads((R.OUT/'verification.json').read_text())
    audit.update(independent_raw_check=True,raw_kr_500ms_400ms_mid_bid_r=raw,
        raw_check_absolute_error=abs(raw-rebuilt),
        saturday_us_tail_check=True,cash_boundary_check=True)
    (R.OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
