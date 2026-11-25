"""Date-block uncertainty for spot-to-perp best quote prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from work import skh_spot_grid as G


def main():
    daily = pd.read_parquet(G.OUT / 'best_quote_forecast_daily.parquet')
    q = daily[(daily.segment == 'regular_0900_1520') &
              (daily.end_lag_ms == 100) &
              (daily.cadence_ms.isin((1000, 5000))) &
              (daily.family == 'micro')]
    days = G.DAYS[1:]
    rng = np.random.default_rng(20261108)
    weights = rng.multinomial(len(days), np.ones(len(days))/len(days), size=5000).T
    rows, points, draws = [], [], []
    for key, group in q.groupby(['target', 'cadence_ms', 'side']):
        sub = group.set_index('day').reindex(days)
        if sub[['n', 'sse', 'sae', 'own_sse', 'own_sae']].isna().any().any():
            raise AssertionError(f'Missing OOS day for {key}')
        for metric, numer, denom in (
            ('mse_gain_vs_own', (sub.own_sse-sub.sse).to_numpy(float),
             sub.own_sse.to_numpy(float)),
            ('mae_gain_vs_own_bp', (sub.own_sae-sub.sae).to_numpy(float),
             sub.n.to_numpy(float))):
            points.append(float(numer.sum()/denom.sum()))
            draws.append((numer@weights)/(denom@weights))
            rows.append((*key, metric, int(sub.n.sum()), len(days),
                         int(np.count_nonzero(numer > 0))))
    points = np.array(points)
    draws = np.array(draws)
    sd = np.std(draws, axis=1, ddof=1)
    z = (draws-points[:, None])/np.maximum(sd[:, None], 1e-15)
    critical = float(np.quantile(np.max(np.abs(z), axis=0), .95))
    result = pd.DataFrame(rows, columns=['target', 'cadence_ms', 'side',
        'metric', 'n', 'days', 'positive_days'])
    result['estimate'] = points
    result['simul_low'] = points-critical*sd
    result['simul_high'] = points+critical*sd
    result['simul_critical'] = critical
    result.to_csv(G.OUT/'best_quote_inference.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
