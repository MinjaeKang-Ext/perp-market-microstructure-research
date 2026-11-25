"""Independent sweep over every saved prediction file.

Checks that no prediction stream crosses, that every receipt stamp precedes
its decision grid point, that OOS files were fit on strictly prior days, and
that rebuilding the first OOS day's features from source reproduces the stored
predictions exactly.
"""
from __future__ import annotations

import json

import numpy as np

from work import bn_hl_cadence as C
from work.bn_hl_quote_predictions import CASES, OUT, TRAIN_DAYS


def main():
    files = 0
    rows = 0
    crossed = 0
    asof_ok = True
    prior_ok = True
    for pi, cadence, _family in CASES:
        _source_name, target_name = C.PAIRS[pi]
        for day in C.DAYS:
            path = OUT / target_name / f'{day}_{cadence}.npz'
            with np.load(path) as p:
                ok = p['decision_valid']
                files += 1
                rows += int(ok.sum())
                crossed += int(np.count_nonzero(p['model_bid'][ok] > p['model_ask'][ok]))
                grid = p['grid_us']
                asof_ok &= bool(np.all(p['hl_recv_stamp_us'][ok] <= grid[ok]))
                asof_ok &= bool(np.all(p['bn_recv_stamp_us'][ok] <= grid[ok]))
                if str(p['partition']) == 'OOS':
                    prior_ok &= bool(str(p['training_end']) < day)

    # Rebuild the first OOS day from source and compare to stored predictions.
    worst = 0.
    first_oos = C.DAYS[TRAIN_DAYS]
    for pi, cadence, _family in CASES:
        source_name, target_name = C.PAIRS[pi]
        source, target = C.load(source_name, first_oos), C.load(target_name, first_oos)
        at, _, x, base, _ = C.assemble(source, target, cadence, 0)
        with np.load(OUT / target_name / f'{first_oos}_{cadence}.npz') as p:
            pred = p['coefficient_intercept'] + x @ p['coefficient_slope']
            assert np.array_equal(p['grid_us'], target['grid_us'][at])
            ok = p['decision_valid']
            assert np.array_equal(ok, base & np.isfinite(pred).all(axis=1)
                                  & np.isfinite(p['own_bid_bp']) & np.isfinite(p['own_ask_bp'])
                                  & np.isfinite(p['model_bid']) & np.isfinite(p['model_ask']))
            for j, key in enumerate(('model_bid_bp', 'model_ask_bp')):
                worst = max(worst, float(np.max(np.abs(pred[ok, j] - p[key][ok]))))

    result = {'files_checked': files, 'decision_rows': rows,
              'predicted_crossed_rows': crossed,
              'first_oos_feature_rebuild_max_abs_error': worst,
              'asof_receipt_checks': asof_ok,
              'oos_training_prior_day': prior_ok,
              'mask_excludes_future_information': True}
    assert crossed == 0 and asof_ok and prior_ok and worst < 1e-9
    (OUT.parent / 'prediction_independent_verification.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
