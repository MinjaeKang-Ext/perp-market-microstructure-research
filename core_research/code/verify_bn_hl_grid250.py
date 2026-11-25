"""Independent raw-event selection checks for both 250 ms grid phases."""
import json
from datetime import timedelta

import numpy as np

from work import fv_mm_inputs as S
from work import bn_hl_grid250 as G


def main():
    verified=0
    hour=S.START+timedelta(hours=9)
    epoch=int(hour.timestamp()*1e6)
    for market in S.MARKETS:
        ev,rr,seq,b,a,q1,q2=S.read_bbo_hour(market,hour)
        for phase in (0,125_000):
            path=G.OUT/market/('2026-10-15'+('' if phase==0 else '_phase125')+'.npz')
            with np.load(path) as z:
                for off in (0,250_000,500_000,89_750_000):
                    t=epoch+phase+off
                    possible=np.flatnonzero(rr<=t)
                    max_seq=seq[possible].max()
                    match=possible[seq[possible]==max_seq]
                    receipt=(rr[match].max() if market.startswith('hyperliquid')
                             else rr[match].min())
                    j=match[rr[match]==receipt][-1]
                    i=9*3600*4+off//250_000
                    assert z['grid_us'][i]==t
                    assert z['recv_stamp_us'][i]==rr[j]
                    assert z['recv_event_us'][i]==ev[j]
                    for name,value in (('bid',b[j]),('ask',a[j]),('bq',q1[j]),('aq',q2[j])):
                        assert z[name][i]==value,(market,phase,off,name)
                    assert abs(z['age_sec'][i]-(t-ev[j])/1e6)<1e-12
                    verified+=1
    result={'independent_raw_lookups_passed':verified,'markets':list(S.MARKETS),
            'date':'2026-10-15','phases_us':[0,125_000]}
    (G.OUT.parent/'grid_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
