"""Compare Binance→HL common/ADR crossed contract directions on identical axes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C
from work import bn_hl_mid_micro_lag as M
from work import bn_hl_mid_micro_lag_report as R

OUT=C.ROOT/'out_bn_hl_cross_type_lag'
REPORT=C.ROOT/'REPORT_bn_hl_cross_type_lag.md'
PAIRS=M.CROSS_TYPE_BN_HL_PAIRS
LABEL={
    'hyperliquid_SKHY':'BN 국장형 SKHYNIXUSDT → HL ADR형 SKHY',
    'hyperliquid_SKHX':'BN ADR형 SKHYUSDT → HL 국장형 SKHX'}
PLOT_LABEL={
    'hyperliquid_SKHY':'BN common to HL ADR',
    'hyperliquid_SKHX':'BN ADR to HL common'}


def peak_cell(block,feature,side,kind='future'):
    row=block[(block.feature==feature)&(block.side==side)].iloc[0]
    if kind=='future':
        return f"{int(row.future_peak_ms)}ms, r={row.future_peak_corr:.3f}, 하한={row.future_peak_simul_low:.3f}"
    return f"{int(row.all_abs_peak_ms)}ms, r={row.all_abs_peak_corr:.3f}"


def row_for(peaks,target,cadence):
    block=peaks[(peaks.target==target)&(peaks.cadence_ms==cadence)]
    return dict(delta=f'{cadence/1000:g}s',
        mid_all=' / '.join(peak_cell(block,'mid',side,'all') for side in ('bid','ask')),
        micro_all=' / '.join(peak_cell(block,'micro',side,'all') for side in ('bid','ask')),
        mid_future=' / '.join(peak_cell(block,'mid',side) for side in ('bid','ask')),
        micro_future=' / '.join(peak_cell(block,'micro',side) for side in ('bid','ask')),
        continuous=int(block.continuous_positive_through_ms.min()))


def direction_contrasts(daily):
    rows=[]
    draws_list=[]
    rng=np.random.default_rng(20261108)
    weights=rng.multinomial(18,np.ones(18)/18,size=2000).T
    for cadence,lag in ((500,400),(1000,100)):
        for feature in ('mid','micro'):
            for side in ('bid','ask'):
                values=[]
                for source,target in PAIRS:
                    block=daily[(daily.source==source)&(daily.target==target)
                                &(daily.cadence_ms==cadence)&(daily.phase_ms==0)
                                &(daily.end_lag_ms==lag)&(daily.feature==feature)
                                &(daily.side==side)].set_index('day').reindex(C.DAYS)
                    if block[list(R.MOMENT)].isna().any().any():
                        raise AssertionError('Matched calendar dates absent from direction contrast')
                    a=block[list(R.MOMENT)].to_numpy()
                    values.append((float(R.corr_from(a.sum(axis=0))),
                                   R.corr_from(np.stack([a[:,j]@weights for j in range(6)],axis=-1))))
                point=values[0][0]-values[1][0]
                draws=values[0][1]-values[1][1]
                rows.append(dict(cadence_ms=cadence,end_lag_ms=lag,feature=feature,side=side,
                                 corr_dom_to_adr=values[0][0],corr_adr_to_dom=values[1][0],
                                 difference=point,ci_low=float(np.quantile(draws,.025)),
                                 ci_high=float(np.quantile(draws,.975))))
                draws_list.append(draws)
    frame=pd.DataFrame(rows)
    draws=np.vstack(draws_list)
    sd=draws.std(axis=1,ddof=1)
    crit=np.quantile(np.max(np.abs((draws-frame.difference.to_numpy()[:,None])/
                                    np.maximum(sd[:,None],1e-15)),axis=0),.95)
    frame['simul_low']=frame.difference-crit*sd
    frame['simul_high']=frame.difference+crit*sd
    frame.to_csv(OUT/'direction_contrasts.csv',index=False)
    return frame


def build():
    manifest=json.loads((OUT/'manifest.json').read_text())
    if not (OUT/'session_response_peaks.csv').exists():
        raise FileNotFoundError('Korean regular-session sensitivity is required')
    if tuple(map(tuple,manifest['pairs']))!=PAIRS:
        raise AssertionError('Cross-type mapping differs from the user correction')
    rd=pd.read_parquet(OUT/'response_daily.parquet')
    fd=pd.read_parquet(OUT/'forecast_daily.parquet')
    if len(rd)!=232704 or len(fd)!=45696:
        raise AssertionError('All eight intervals and 18/17 days must be complete')
    response=R.response_summary(rd,out=OUT)
    forecast=R.forecast_summary(fd,out=OUT)
    peaks=R.response_peaks(response,out=OUT)
    joint,joint_peaks=R.forecast_peaks(forecast,out=OUT)
    R.figure(response,joint,out=OUT,labels=PLOT_LABEL)
    contrasts=direction_contrasts(rd)
    overview=[]
    for cadence in C.CADENCES_MS:
        one={'delta':f'{cadence/1000:g}s'}
        for target,tag in [('hyperliquid_SKHY','dom_to_adr'),('hyperliquid_SKHX','adr_to_dom')]:
            b=peaks[(peaks.target==target)&(peaks.cadence_ms==cadence)
                    &(peaks.feature=='mid')]
            one[tag+'_peak']=' / '.join(f"{int(r.future_peak_ms)}ms @ {r.future_peak_corr:.3f}"
                                           for r in b.sort_values('side',ascending=False).itertuples())
            one[tag+'_last']=int(peaks[(peaks.target==target)&
                                          (peaks.cadence_ms==cadence)].continuous_positive_through_ms.min())
        overview.append(one)
    lines=['# Binance→Hyperliquid 국장형↔ADR형 교차 퍼프 전이 두 케이스','',
      '2026-11-08. 사용자 교정에 따라 거래소 방향은 **Binance→Hyperliquid로 고정**하고, BN 국장형→HL ADR형과 BN ADR형→HL 국장형 두 케이스만 비교한다. 2026-10-15~11-01 KST의 기존 BBO를 사용하며 처음 하루 학습 후17일 순차검증이다.','',
      '## 대상과 측정 방법','',
      '| 케이스 | Binance 출발 | Hyperliquid 도착 |',
      '| --- | --- | --- |',
      '| 국장형→ADR형 | SKHYNIXUSDT | SKHY best bid/ask |',
      '| ADR형→국장형 | SKHYUSDT | SKHX best bid/ask |','',
      '가격 단위와 ADR 교환비율·FX가 다를 수 있으므로 원가격 차이를 회귀하지 않고 각 계약의 **로그수익률**을 사용한다. Binance mid와 L1 잔량가중 microprice의 완성된 Δ구간(500ms,1,2,3,5,10,30,60초) 변화를 두 신호로 사용한다. 결합회귀는 mid와 micro−mid를 넣어 중복을 처리한다. HL의100ms 최우선 bid/ask 변화 구간이 끝나는 시각 L=−5~+5초를 Δ 끝에 대해 측정한다. L≤0은 이미 발생한 반응이다.','',
      '## 두 방향의 미래 반응 비교','',
      '각 셀은 `best bid / best ask`의 **미래 정점 시각 @ 상관계수**다. 값은 pooled 날짜의 기술통계이며 가격 전이의 단방향 통신 지연이나 수익을 직접 측정하지 않는다. 마지막 열은 +100ms부터 mid·micro·bid·ask 네 곡선에서 동시 하한이 연속 양수인 구간이다.','',
      R.table(overview,dict(delta='BN 신호 Δ',dom_to_adr_peak='국장형→ADR형 미래 피크 ms @ r',
                             dom_to_adr_last='연속 구간 ms',
                             adr_to_dom_peak='ADR형→국장형 미래 피크 ms @ r',
                             adr_to_dom_last='연속 구간 ms')),'',
      '두 방향의 피크가 비슷하므로 대표 구간(500ms 신호의 +400ms 반응,1초 신호의 +100ms 반응)에서 **같은 날짜를 함께 재표본화**해 상관 차이를 직접 계산했다. 차이는 국장형→ADR형 r − ADR형→국장형 r이며, 하한·상한은 네 신호/양쪽을 함께 보정했다.','',
      R.table([dict(delta=f'{v.cadence_ms/1000:g}s',feature=v.feature,side=v.side,
                    difference=f'{v.difference:.4f}',low=f'{v.simul_low:.4f}',
                    high=f'{v.simul_high:.4f}') for v in contrasts.itertuples()],
              dict(delta='Δ',feature='BN 신호',side='HL 측',difference='방향차 r',
                   low='동시 하한',high='동시 상한')),'',
      '**비교 해석:** 두 교차방향 모두500ms BN 신호의 HL 반응 정점이 종료 후400ms이고, mid 상관은 국장형→ADR형 약0.269~0.270, ADR형→국장형 약0.260~0.261이다. Micro의 정점 상관은 두 방향 모두 약0.269~0.271로 사실상 비슷하다. 1초 신호에서는 둘 다100ms 근처와 r≈0.25다. 날짜를 짝지은 방향차의 동시 구간은500ms mid ask 한 항목을 제외하고0을 포함하므로 어느 방향이 일반적으로 더 강하다고 순위를 확정하지 않는다.','',
      '2초 이상 정렬에서는 전체 상관 정점이 대체로 신호 구간 종료 이전으로 옮겨간다. 신호 구간을 길게 만든 탓에 그 안에 이미 발생한 HL 가격 이동이 포함된 것이다. +100ms부터 연속 양의 상관이500ms 정렬에서 국장형→ADR형1.9초, ADR형→국장형1.4초라는 값도 **상관의 지속 구간**이지 주문 후 수익 가능한 정보 수명이 아니다.','',
      '## 전체 정점과 mid/micro 각각의 강도','',
      '전체 정점은 −5~+5초 중 절대 상관 최대 위치다. 미래 정점은 +100~+5000ms 중 양의 상관 최대 위치다. `r` 옆의 하한은 날짜 단위 bootstrap에서 가족×bid/ask×미래 lag를 함께 보정한95% 하한이다. 30초·1분에서는 신호 구간 중 이미 생긴 반응이 집계되므로 작은 미래 최대값을 장수명 신호로 읽지 않는다.','']
    for target in LABEL:
        rows=[row_for(peaks,target,cadence) for cadence in C.CADENCES_MS]
        lines += [f'### {LABEL[target]}','',R.table(rows,dict(
            delta='Δ',mid_all='mid 전체 피크 bid/ask (ms,r)',
            micro_all='micro 전체 피크 bid/ask (ms,r)',
            mid_future='mid 미래 피크 bid/ask (ms,r,하한)',
            micro_future='micro 미래 피크 bid/ask (ms,r,하한)',
            continuous='네 곡선 연속 구간 ms')),'']
    session=pd.read_csv(OUT/'session_response_peaks.csv')
    srows=[]
    for target in LABEL:
        for cadence in (500,1000):
            for window in ('KR_09_1520','other'):
                block=session[(session.target==target)&(session.cadence_ms==cadence)
                              &(session.session==window)]
                row=dict(case=LABEL[target],delta=f'{cadence/1000:g}s',session=window)
                for family in ('mid','micro'):
                    sub=block[block.feature==family]
                    row[family]=' / '.join(f'{int(v.end_lag_ms)}ms @ {v["corr"]:.3f}'
                                                 for _,v in sub.sort_values('side',ascending=False).iterrows())
                srows.append(row)
    lines += ['### 국장 정규시간대 민감도','',
       '한국시간09:00~15:20 구간과 그 외 시간의500ms·1초 미래 피크를 별도 계산했다. 각 셀은 bid/ask 순이며, 시간대는 한쪽 시장만 열려 있다는 보장이 아니므로 거래시간의 인과효과로 해석하지 않는다.','',
       R.table(srows,dict(case='케이스',delta='Δ',session='KST 구간',mid='mid 미래 피크 ms @ r',
                          micro='micro 미래 피크 ms @ r')),'',
       '두 방향 모두500ms 정렬의 미래 피크400ms는 두 시간대에서 유지된다. 정규시간대의 상관은 나머지 시간보다 대체로 높지만, 시간대별 유효 표본과 공통 기초가격·뉴스 환경이 다르므로 계약 간 선도 방향이 바뀐다는 근거로 보지 않는다.','']
    phase_rows=[]
    for target in LABEL:
        for cadence in (500,1000):
            for phase in (0,(cadence//100//2)*100):
                block=response[(response.target==target)&(response.cadence_ms==cadence)
                               &(response.phase_ms==phase)&(response.end_lag_ms>0)]
                row=dict(case=LABEL[target],delta=f'{cadence/1000:g}s',phase=f'{phase}ms')
                for family in ('mid','micro'):
                    result=[]
                    for side in ('bid','ask'):
                        sub=block[(block.feature==family)&(block.side==side)]
                        best=sub.loc[sub['corr'].idxmax()]
                        result.append(f'{int(best.end_lag_ms)}ms @ {best["corr"]:.3f}')
                    row[family]=' / '.join(result)
                phase_rows.append(row)
    lines += ['### 정렬 위상 민감도','',
      '정각 위상과 약 반 간격 이동 위상의500ms·1초 미래 피크다. 500ms의 반 간격은100ms 입력 해상도로200ms 이동이다. 정점 시각이 위상에 크게 바뀌면 그 지연을 고정된 전이시간으로 사용하지 않는다.','',
      R.table(phase_rows,dict(case='케이스',delta='Δ',phase='위상',
        mid='mid bid/ask ms @ r',micro='micro bid/ask ms @ r')),'']
    lines += [f'![교차 종목 반응 지도]({OUT/"response_heatmap.png"})','',
      '## HL 자체 기준선 대비 mid+micro 결합','',
      '각 Δ와 +100~+5000ms 후의 개별100ms best bid/ask 변화를 전일 이전 자료만으로 예측했다. HL 자체정보 기준선, Binance mid 단독, micro 단독, mid+micro 결합을 **동일 표본**에서 비교했다. 양수 MAE 개선이어도 실제 주문 수익은 아니며, mid 대비 추가분의 동시 하한이 양수여야 micro의 조건부 기여를 주장할 수 있다.','']
    for target in LABEL:
        rows=[]
        for cadence in C.CADENCES_MS:
            b=joint_peaks[(joint_peaks.target==target)&(joint_peaks.cadence_ms==cadence)].iloc[0]
            rows.append(dict(delta=f'{cadence/1000:g}s',
                best_l=f'{int(b.end_lag_ms)}ms' if b.worst_low>0 else '통과 없음',
                gain=R.fmt(b.worst_gain,5),low=R.fmt(b.worst_low,5),
                increment=R.fmt(b.worst_increment,5),
                increment_low=R.fmt(b.worst_increment_low,5)))
        lines += [f'### {LABEL[target]}','',R.table(rows,dict(delta='Δ',best_l='동시 하한 양수인 최상 L',
            gain='HL 자체 대비 최고 약한 측 개선 bp',low='동시 하한',
            increment='mid 대비 추가 bp',increment_low='추가분 동시 하한')),'']
    lines += ['각 L의 모형은 그 지연을 목표로 전일까지 다시 학습했다. 한 번 보낸 주문을 기다리는 동안 같은 정보가 살아 있다는 검사는 아니다. 100ms 가격변화에는 0값이 많아 MAE가 작은 비영 예측에 불리할 수 있다. 앞선 다음 Δ구간 누적 가격 예측과 목표가 다르다.','',
      '## 동일 종목 대비 및 한계','',
      '동일종목 BN→HL의500ms 신호에서 mid/micro 수신축 정점은 두 쌍 모두 종료 후400ms, r≈0.30~0.31이었다. 위 교차종목 결과를 이 기준과 같은 기간·같은 Δ·같은 HL best 가격 단위에서 비교한다. 교차종목의 상관이 관측되어도 국내 현물·ADR 현물·환율·거래시간·공통 뉴스·계약 승수의 영향을 분리한 것은 아니다.','',
      '정각 위상과 약 반 구간 이동 위상을 모두 계산했다. 보고서 표는 정각 위상이며, 위상별 전체 곡선과 가격 최신성은 CSV 및 날짜별 결과에 있다. 이벤트시각 기준1초 최신성 필터를 적용했지만 거래소·수집기 시계 동기화, 실제 주문 활성·취소·체결과 자기 대기열은 미검증이다.','',
      '## 근거 파일','',
      '- `response_summary.csv`, `response_peaks.csv`: 네 신호곡선의 정점 시각·상관·동시 하한.',
      '- `direction_contrasts.csv`: 같은 날짜를 짝지은 두 교차방향 상관 차이와 동시 구간.',
      '- `forecast_summary.csv`, `forecast_joint_lag_curve.csv`: HL 자체 및 mid 단독 대비 OOS 예측.',
      '- `response_daily.parquet`, `forecast_daily.parquet`, `coverage_daily.csv`, `manifest.json`: 날짜별 원자료와 두 케이스 정의.',
      '- `verification.json`: 모든 날짜·표본·지연 및 독립 OLS 재구성 검사.','']
    REPORT.write_text('\n'.join(lines))
    print('WROTE',REPORT,'rows',len(response),len(forecast),flush=True)


if __name__=='__main__':
    build()
