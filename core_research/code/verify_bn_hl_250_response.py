"""Reconstruct independent cash-session, funding-filtered Pearson moments."""
import json

import numpy as np
import pandas as pd

from work import bn_hl_250_response as R
from work import bn_hl_cash_regime as K
from work import skh_spot_fv_transfer as T


def main():
    day='2026-10-15'; source='binance_SKHYNIXUSDT'; target='hyperliquid_SKHX'
    s=R.load(source,day,125); z=R.load(target,day,125)
    grid=s['grid_us']; label=K.regime_labels(grid)
    a=np.arange(1,len(grid)-2); end=a+2
    def age(d,i):return np.isfinite(d['age_sec'][i])&(d['age_sec'][i]>=0)&(d['age_sec'][i]<=1)
    keep=(age(s,a)&age(s,a-1)&age(z,end)&age(z,end-1)&
          (label[a]==0)&(label[a-1]==0)&(label[end]==0)&(label[end-1]==0)&
          T.funding_clear(grid[a-1],grid[end]))
    with np.errstate(invalid='ignore',divide='ignore'):
        mid=(s['bid']+s['ask'])/2
        x=1e4*np.log(mid[a]/mid[a-1])
        y=1e4*np.log(z['ask'][end]/z['ask'][end-1])
    keep &= np.isfinite(x)&np.isfinite(y)
    direct=float(np.corrcoef(x[keep],y[keep])[0,1])
    d=pd.read_parquet(R.ROOT/f'response_{day}.parquet')
    q=d[(d.source==source)&(d.target==target)&(d.phase_ms==125)&
        (d.regime=='KR_cash')&d.exclude_funding&(d.end_lag_ms==500)&
        (d.feature=='mid')&(d.side=='ask')]
    m=q[['n','sx','sy','sxx','syy','sxy']].sum()
    rebuilt=(m.sxy-m.sx*m.sy/m.n)/np.sqrt((m.sxx-m.sx*m.sx/m.n)*(m.syy-m.sy*m.sy/m.n))
    assert int(m.n)==int(keep.sum())
    assert abs(direct-rebuilt)<1e-10
    result={'day':day,'phase_ms':125,'lag_ms':500,'regime':'KR_cash',
            'funding_excluded':True,'n':int(m.n),'independent_r':direct,
            'stored_moment_r':float(rebuilt),'abs_error':float(abs(direct-rebuilt))}
    (R.ROOT/'response_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
