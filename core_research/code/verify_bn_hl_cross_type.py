"""Independent completeness and past-only OLS checks for the two corrected pairs."""
from __future__ import annotations

import json

import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_mid_micro_lag as M
from work import verify_bn_hl_mid_micro_lag as V

OUT=C.ROOT/'out_bn_hl_cross_type_lag'


def main():
    prior=json.loads((C.OUT/'verification.json').read_text())
    if prior['status']!='pass' or prior['raw_spot']['raw_events_checked']!=64:
        raise AssertionError('Four market inputs not independently checked')
    complete=V.complete(output=OUT,pairs=M.CROSS_TYPE_BN_HL_PAIRS)
    ols=[dict(source=source,target=target,**V.score_reconstruction(source,target,output=OUT))
         for source,target in M.CROSS_TYPE_BN_HL_PAIRS]
    session={}
    if (OUT/'session_response_daily.parquet').exists():
        daily=pd.read_parquet(OUT/'session_response_daily.parquet')
        summary=pd.read_csv(OUT/'session_response_summary.csv')
        peaks=pd.read_csv(OUT/'session_response_peaks.csv')
        expected=(18*2*2*2*41*2*2,2*2*2*41*2*2,2*2*2*2*2)
        observed=(len(daily),len(summary),len(peaks))
        if expected!=observed:
            raise AssertionError(f'Incomplete session sensitivity {observed} vs {expected}')
        if not daily.groupby(['source','target','cadence_ms','session',
                             'end_lag_ms','feature','side']).day.nunique().eq(18).all():
            raise AssertionError('Session dates incomplete')
        session=dict(row_counts=observed,all_18_dates=True)
    result=dict(status='pass',pairs=[list(v) for v in M.CROSS_TYPE_BN_HL_PAIRS],
                completeness=complete,independent_ols=ols,
                four_market_raw_bbo_ofi_checks=prior['raw_spot']['raw_events_checked'],
                session=session)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
