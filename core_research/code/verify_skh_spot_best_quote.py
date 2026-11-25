"""Independently refit first-day best-quote model and score the next day."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work import skh_spot_grid as G
from work import skh_spot_fv_forecast as F
from work import skh_spot_best_quote as B


def load(day, target):
    spot = G.load(day)
    perp = C.load(target, day)
    raw = perp['ofi100_norm']*(perp['bq']+perp['aq'])
    perp['_ofi_cum'] = np.r_[0., np.cumsum(np.where(np.isfinite(raw), raw, 0.))]
    return B.design(spot, perp, 1000, 100, 'regular_0900_1520')


def main():
    target = 'hyperliquid_SKHX'
    x0, y0 = load('2026-10-15', target)
    x1, y1 = load('2026-10-16', target)
    m = E.Moments.empty(8, 2)
    m.update(x0, y0)
    a, b = m.solve(F.FAMILIES['micro'])
    a0, b0 = m.solve(F.FAMILIES['own'])
    pred = a + x1@b
    own = a0 + x1@b0
    daily = pd.read_parquet(G.OUT/'best_quote_forecast_daily.parquet')
    errors = {}
    for j, side in enumerate(B.SIDES):
        q = daily[(daily.day=='2026-10-16') & (daily.target==target) &
                  (daily.cadence_ms==1000) & (daily.end_lag_ms==100) &
                  (daily.segment=='regular_0900_1520') &
                  (daily.family=='micro') & (daily.side==side)]
        assert len(q)==1
        assert int(q.iloc[0].n)==len(y1)
        got = dict(sse=float(np.sum((y1[:,j]-pred[:,j])**2)),
                   sae=float(np.sum(np.abs(y1[:,j]-pred[:,j]))),
                   own_sse=float(np.sum((y1[:,j]-own[:,j])**2)),
                   own_sae=float(np.sum(np.abs(y1[:,j]-own[:,j]))))
        errors[side] = {k:abs(v-q.iloc[0][k]) for k,v in got.items()}
        assert all(e<1e-7 for e in errors[side].values()), errors
    audit = json.loads((G.OUT/'verification.json').read_text())
    audit['best_quote_independent_oos_n'] = len(y1)
    audit['best_quote_independent_errors'] = errors
    (G.OUT/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'n':len(y1),'errors':errors},indent=2))


if __name__=='__main__':
    main()
