"""Verify replay prediction streams are aligned and do not gate on future labels."""
from __future__ import annotations

import json
import numpy as np

from work import bn_hl_cadence as C
from work.bn_hl_quote_predictions import CASES, OUT


def main():
    day='2026-10-24'; checks=[]
    for pi,cadence,family in CASES:
        source_name,target_name=C.PAIRS[pi]
        source,target=C.load(source_name,day),C.load(target_name,day)
        at,_,x,base,_=C.assemble(source,target,cadence,0)
        y,label=C.future_target(target,at,cadence,base)
        with np.load(OUT/target_name/f'{day}_{cadence}.npz') as p:
            pred=p['coefficient_intercept']+x@p['coefficient_slope']
            assert np.array_equal(p['grid_us'],target['grid_us'][at])
            assert str(p['partition'])=='OOS' and str(p['training_end'])<day
            assert np.allclose(pred[:,0],p['model_bid_bp'],atol=1e-10,equal_nan=True)
            assert np.allclose(pred[:,1],p['model_ask_bp'],atol=1e-10,equal_nan=True)
            expected=base&np.isfinite(pred).all(axis=1)
            expected &= np.isfinite(p['own_bid_bp'])&np.isfinite(p['own_ask_bp'])
            expected &= np.isfinite(p['model_bid'])&np.isfinite(p['model_ask'])
            assert np.array_equal(expected,p['decision_valid'])
            count=int(np.count_nonzero(expected&~label))
            assert count>0,'No label-invalid decisions retained'
            checks.append(dict(target=target_name,cadence_ms=cadence,
                               family=family,valid_decisions=int(expected.sum()),
                               decisions_with_unavailable_future_label=count,
                               prediction_matches_coefficients=True))
    result={'day':day,'cases':checks,'decision_mask_excludes_future_label':True}
    (OUT.parent/'prediction_label_guard_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
