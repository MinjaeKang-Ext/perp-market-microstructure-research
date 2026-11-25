"""Resumable driver for the prediction stage.

The sandbox reaps long-lived processes, so the work plan is split into
day-sized units with the accumulated moments checkpointed to disk. Each
invocation runs units until a wall-clock budget is spent, so the stage can be
driven by repeated short calls without recomputing finished days.

Ordering is identical to bn_hl_quote_predictions.main(): train on the nine IS
days, emit the retrospective IS fits from the full nine-day moments, then walk
the OOS days emitting before updating so a fit never sees its own day.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_transfer_eval as E
from work.bn_hl_quote_predictions import (BASE, CASES, FAMILY, OUT, TRAIN_DAYS,
                                          evaluate_and_write, load_case)

STATE = BASE / 'driver_state.json'
ROWS = BASE / 'prediction_audit_rows.csv'
COLS = ('day', 'source', 'target', 'cadence_ms', 'family', 'partition', 'side',
        'decision_n', 'evaluation_n', 'sse', 'own_sse', 'sae', 'own_sae')


def plan():
    units = [('train', d) for d in C.DAYS[:TRAIN_DAYS]]
    units.append(('snapshot', 'is'))
    units += [('emit_is', d) for d in C.DAYS[:TRAIN_DAYS]]
    for d in C.DAYS[TRAIN_DAYS:]:
        units.append(('emit_oos', d))
        units.append(('train', d))
    units.append(('finalize', ''))
    return units


def save_moments(moments, name):
    blob = {}
    for i, case in enumerate(CASES):
        m = moments[case]
        blob[f'{i}_n'] = np.array(m.n)
        for field in ('sx', 'sy', 'xx', 'xy'):
            blob[f'{i}_{field}'] = getattr(m, field)
    np.savez(BASE / f'moments_{name}.npz', **blob)


def load_moments(name):
    out = {}
    with np.load(BASE / f'moments_{name}.npz') as f:
        for i, case in enumerate(CASES):
            out[case] = E.Moments(int(f[f'{i}_n']), f[f'{i}_sx'].copy(), f[f'{i}_sy'].copy(),
                                  f[f'{i}_xx'].copy(), f[f'{i}_xy'].copy())
    return out


def append_rows(rows):
    frame = pd.DataFrame(rows, columns=list(COLS))
    frame.to_csv(ROWS, mode='a', header=not ROWS.exists(), index=False)


def finalize():
    audit = pd.read_csv(ROWS).drop_duplicates(
        subset=['day', 'target', 'cadence_ms', 'partition', 'side'], keep='last')
    audit.to_csv(BASE / 'prediction_audit.csv', index=False)
    old = pd.read_csv(C.OUT / 'r2_daily.csv')
    q = audit[audit.partition == 'OOS']
    check = q.merge(old, on=['day', 'source', 'target', 'cadence_ms', 'family', 'side'],
                    suffixes=('', '_old'), validate='one_to_one')
    assert len(check) > 0 and (check.evaluation_n == check.n).all()
    if not FAMILY:
        assert len(check) == len(q)
    errors = {col: float(np.max(np.abs(check[col] - check[col + '_old'])))
              for col in ('sse', 'own_sse', 'sae', 'own_sae')}
    assert max(errors.values()) < 1e-5, errors
    meta = {'cases': [list(c) for c in CASES], 'days': list(C.DAYS),
            'is_days': list(C.DAYS[:TRAIN_DAYS]), 'oos_days': list(C.DAYS[TRAIN_DAYS:]),
            'is_policy': 'retrospective fit on all 9 IS dates; not executable walk-forward performance',
            'oos_policy': 'past dates only, latest fit day day-1',
            'price_prediction': 'current quote * exp(predicted log return in bp / 10000)',
            'decision_mask': 'base_valid and finite predicted prices at current t only',
            'future_label': 'used only in past-day training and offline scoring; never saved as decision input',
            'deadline': 'labels crossing day end excluded from training',
            'forced_family': FAMILY, 'verification': errors,
            'oos_validation_rows': len(check), 'oos_rows': len(q)}
    (BASE / 'prediction_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def run(budget=35.):
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    units = plan()
    done = json.loads(STATE.read_text())['done'] if STATE.exists() else 0
    moments = load_moments('run') if done else {c: E.Moments.empty(10, 2) for c in CASES}
    started = time.time()
    while done < len(units):
        kind, day = units[done]
        if kind == 'train':
            for case in CASES:
                _, _, _, x, _, y, valid = load_case(case, day)
                moments[case].update(x[valid], y[valid])
            save_moments(moments, 'run')
        elif kind == 'snapshot':
            save_moments(moments, day)
        elif kind == 'emit_is':
            is_moments = load_moments('is')
            rows = []
            for case in CASES:
                rows += evaluate_and_write(case, day, 'IS_retroactive', is_moments[case],
                                           C.DAYS[TRAIN_DAYS - 1])
            append_rows(rows)
        elif kind == 'emit_oos':
            previous = C.DAYS[list(C.DAYS).index(day) - 1]
            rows = []
            for case in CASES:
                rows += evaluate_and_write(case, day, 'OOS', moments[case], previous)
            append_rows(rows)
        else:
            print(json.dumps(finalize(), indent=2), flush=True)
        done += 1
        STATE.write_text(json.dumps({'done': done, 'unit': [kind, day]}) + '\n')
        print(f'UNIT {done}/{len(units)} {kind} {day}', flush=True)
        if time.time() - started > budget:
            break
    print('REMAINING', len(units) - done, flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=35.)
    run(ap.parse_args().budget)
