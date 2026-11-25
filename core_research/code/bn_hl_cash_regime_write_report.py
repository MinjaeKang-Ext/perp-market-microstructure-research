"""Write the human-readable regime study from verified outputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import bn_hl_cash_regime as R


LABEL={
    R.PAIRS[0]:'동일 국장형: SKHYNIXUSDT→SKHX',
    R.PAIRS[1]:'동일 ADR형: SKHYUSDT→SKHY',
    R.PAIRS[2]:'국장형→ADR형: SKHYNIXUSDT→SKHY',
    R.PAIRS[3]:'ADR형→국장형: SKHYUSDT→SKHX',
}
REGIME_KO={'KR_cash':'국장','US_cash':'미장',
           'closed_weekday':'양시장 휴장 평일',
           'closed_weekend':'양시장 휴장 주말'}


def f(v): return f'{v:.3f}'


def peak_cell(q):
    r=q.iloc[0]
    return f'+{int(r.future_peak_ms)}ms / {f(r.future_peak_corr)}'


def main():
    path=R.OUT
    fixed=pd.read_csv(path/'fixed_regime_comparison.csv')
    summary=pd.read_csv(path/'response_summary.csv')
    peaks=pd.read_csv(path/'response_peaks.csv')
    common=pd.read_csv(path/'common_support_summary.csv')
    audit=json.loads((path/'verification.json').read_text())
    assert audit['pass_'] and audit['independent_raw_check']
    grid={k:v/36000 for k,v in audit['grid_100ms'].items()}
    lines=[
        '# Binance→Hyperliquid best price 전이: 국장·미장 현물시간 레짐',
        '',
        '## 핵심 결과',
        '',
        '- 분석 기간은 2026-10-15~11-01 KST 18일이며, 기존 동일형 2쌍과 교차형 2쌍을 모두 재분석했다. 네 쌍의 방향은 항상 Binance→Hyperliquid다.',
        '- **500ms 신호의 국장 정규장 정점은 4쌍×mid/micro×best bid/ask 전부 +400ms**다. 미장 정규장은 +400~500ms다. 정렬 위상을 반 간격 이동하면 미장 정점 일부가 +400ms로 바뀌므로 장별 100ms 지연 차이는 확정할 수 없다.',
        f'- 사전에 고른 500ms/+400ms와 1s/+100ms의 같은 거래일 12블록 비교에서 국장 상관이 미장보다 32개 지표 모두 높았다. 32개 차이를 동시 보정한 95% 구간의 하한이 0을 넘은 것은 **{int((fixed.paired_simul_low>0).sum())}/32개**다. 동일 ADR형 SKHYUSDT→SKHY의 8개 지표는 모두 하한이 0 이하라 장별 우위를 확정할 수 없다.',
        '- 정점 상관의 크기는 대체로 0.25~0.32다. 이는 관계가 있지만 개별 100ms 호가변화의 분산 대부분을 설명하지 못한다. 이전 동일형·교차형 연구에서 HL 자체정보 대비 100ms MAE 개선이 확인되지 않았다는 결론도 유지된다.',
        '',
        '## 시장시간 정의와 데이터',
        '',
        '| 레짐 | 현물 정규장 기준 | 분석 구간의 달력상 시간 |',
        '|---|---|---:|',
        f'| 국장 | KRX 개장일 09:00–15:30 KST | {grid["KR_cash"]:g}시간 |',
        f'| 미장 | NYSE 개장일 09:30–16:00 ET, 이번 기간 KST 22:30–익일 05:00 | {grid["US_cash"]:g}시간 |',
        f'| 양시장 휴장 평일 | 위 두 정규장 밖의 KST 평일 | {grid["closed_weekday"]:g}시간 |',
        f'| 양시장 휴장 주말 | 위 두 정규장 밖의 KST 토·일 | {grid["closed_weekend"]:g}시간 |',
        '',
        'KRX 정규시간은 [KRX 거래시간](https://global.krx.co.kr/contents/GLB/06/0602/0602020204/GLB0602020204T1.jsp), 미장 정규시간과 2026 휴장일은 [NYSE 공식 달력](https://www.nyse.com/trade/hours-calendars), 미국 서머타임 종료일은 [NIST](https://www.nist.gov/pml/time-and-frequency-division/popular-links/daylight-saving-time-dst)에 맞췄다. XKRX/XNYS 거래소 달력을 UTC 개장·폐장 구간으로 변환했다. 10/15~10/30에 두 시장은 각각 12개 정규 세션이며, 첫날 00:00–05:00 KST는 10/14 미국 세션의 뒷부분으로 포함했다. 10/30 미국 세션은 10/31 토요일 05:00 KST까지 포함했다. 두 정규 세션은 겹치지 않는다.',
        '',
        '신호는 Binance에서 이미 완성된 Δ 구간 `(t−Δ,t]`의 mid 또는 L1 잔량가중 microprice 로그수익률이다. 반응은 HL best bid·ask의 `(t+L−100ms,t+L]` 로그수익률이며, Δ는 0.5·1·2·3·5·10·30·60초, L은 −5~+5초를 100ms로 훑었다. 가격은 같은 종목형의 수익률로 비교하고 계약·통화 단위가 다른 절대가격을 직접 뺐지 않았다. 인과 시계는 수집기의 `received_at`이며, 양 끝 호가 이벤트 나이≤1초와 신호·반응 전체가 같은 레짐에 있을 것을 요구했다.',
        '',
        '## 사전 고정 시차: 국장 대 미장',
        '',
        '아래 r은 **같은 거래일 12개 완전 블록**(KST 07:00~익일 07:00)만 합산했다. Δ=500ms는 L=+400ms, Δ=1s는 L=+100ms다. 차이의 대괄호는 32개 지표에 대해 같은 블록을 3,000회 재표본하고 동시 보정한 95% 구간이다. 이 시차도 이전 연구의 같은 18일에서 선택했으므로 미사용 표본 검증은 아니다.',
        '',
        '| 케이스 | Δ | 입력 | 국장 bid/ask r | 미장 bid/ask r | 국장−미장 bid 95% | 국장−미장 ask 95% |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for pair in R.PAIRS:
        for cadence in (500,1000):
            for feature in ('mid','micro'):
                q=fixed[(fixed.source==pair[0])&(fixed.target==pair[1])&
                        (fixed.cadence_ms==cadence)&(fixed.feature==feature)]
                bid=q[q.side=='bid'].iloc[0];ask=q[q.side=='ask'].iloc[0]
                bidci=f'{f(bid.difference_KR_minus_US)} [{f(bid.paired_simul_low)}, {f(bid.paired_simul_high)}]'
                askci=f'{f(ask.difference_KR_minus_US)} [{f(ask.paired_simul_low)}, {f(ask.paired_simul_high)}]'
                lines.append(f'| {LABEL[pair]} | {cadence/1000:g}s | {feature} | '
                    f'{f(bid.r_KR)} / {f(ask.r_KR)} | {f(bid.r_US)} / {f(ask.r_US)} | '
                    f'{bidci} | {askci} |')
    lines += [
        '',
        '500ms/+400ms의 국장−미장 차이는 동일 국장형 mid bid **+0.0355 [0.0205, 0.0504]**, 국장형→ADR형 mid bid **+0.0202 [0.0013, 0.0391]**, ADR형→국장형 mid bid **+0.0336 [0.0186, 0.0487]**다. 동일 ADR형 mid bid는 **+0.0117 [−0.0072, 0.0306]**로 구분되지 않는다. 숫자는 r의 차이이며 가격 변화 bp나 수익이 아니다.',
        '',
        '## 정점 시차와 8개 정렬 간격',
        '',
        '표의 `+시간 / r`은 신호 종료 후 양의 시차에서 발견한 최대 상관이다. 마지막 두 열은 전체 ±5초 중 절댓값 최대의 **시차**로, 음수면 Binance 신호가 끝나기 전에 HL 반응이 일어난 것이다. 표는 best bid이며 best ask의 전 수치는 [`response_peaks.csv`](out_bn_hl_cash_regime/response_peaks.csv)에 있다.',
        '',
    ]
    for pair in R.PAIRS:
        lines += [f'### {LABEL[pair]}', '',
            '| Δ | 국장 mid bid 미래 정점 | 미장 mid bid 미래 정점 | 국장 micro bid 미래 정점 | 미장 micro bid 미래 정점 | 전체 mid 정점 시차 국장/미장 |',
            '|---:|---:|---:|---:|---:|---:|']
        for cadence in R.C.CADENCES_MS:
            def q(regime,feature):
                return peaks[(peaks.source==pair[0])&(peaks.target==pair[1])&
                    (peaks.cadence_ms==cadence)&(peaks.regime==regime)&
                    (peaks.feature==feature)&(peaks.side=='bid')]
            km=q('KR_cash','mid');um=q('US_cash','mid')
            kmi=q('KR_cash','micro');umi=q('US_cash','micro')
            lines.append(f'| {cadence/1000:g}s | {peak_cell(km)} | {peak_cell(um)} | '
                f'{peak_cell(kmi)} | {peak_cell(umi)} | '
                f'{int(km.iloc[0].absolute_peak_ms):+d} / {int(um.iloc[0].absolute_peak_ms):+d}ms |')
        lines.append('')
    lines += [
        '500ms는 국장·미장 모두 정점이 신호 종료 뒤에 있다. 1s에서는 국장의 전체 정점이 대체로 L=0이고, Δ≥2s에서는 대부분 전체 정점이 0 또는 음수다. Δ를 길게 할수록 양의 미래 정점 r이 약해진다. 예컨대 동일 국장형의 mid bid 미래 정점은 국장 기준 Δ=500ms **0.321**에서 Δ=60s **0.057**, 미장 기준 **0.288→0.067**이다. 60초 격자의 양의 정점 시간은 100ms 반응곡선의 작은 국소 최댓값이며 60초의 물리적 전이 지연을 뜻하지 않는다.',
        '',
        '## 양시장 휴장 시간과 표본 성격',
        '',
        '아래는 18일 전체의 mid→best bid 고정 시차 r이다. 미장 열은 첫날 10/14 세션의 마지막 5시간도 포함하므로 앞의 12블록 paired 표와 약간 다르다.',
        '',
        '| 케이스 | Δ/L | 국장 r | 미장 r | 휴장 평일 r | 휴장 주말 r |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for pair in R.PAIRS:
        for cadence,lag in ((500,400),(1000,100)):
            q=summary[(summary.source==pair[0])&(summary.target==pair[1])&
                      (summary.cadence_ms==cadence)&(summary.phase_ms==0)&
                      (summary.end_lag_ms==lag)&(summary.feature=='mid')&
                      (summary.side=='bid')]
            rr={r.regime:r['corr'] for _,r in q.iterrows()}
            lines.append(f'| {LABEL[pair]} | {cadence/1000:g}s/+{lag}ms | '
                f'{f(rr["KR_cash"])} | {f(rr["US_cash"])} | '
                f'{f(rr["closed_weekday"])} | {f(rr["closed_weekend"])} |')
    q=summary[(summary.cadence_ms==500)&(summary.end_lag_ms==400)&
              (summary.phase_ms==0)&(summary.feature=='mid')&
              (summary.side=='bid')]
    lines += ['',
        '500ms/+400ms에서 목표 호가의 100ms 실제 가격변경 비율은 국장 구간 **'
        f'{q[q.regime=="KR_cash"].target_change_fraction.min()*100:.1f}~'
        f'{q[q.regime=="KR_cash"].target_change_fraction.max()*100:.1f}%**, 미장 '
        f'**{q[q.regime=="US_cash"].target_change_fraction.min()*100:.1f}~'
        f'{q[q.regime=="US_cash"].target_change_fraction.max()*100:.1f}%**, '
        '양시장 휴장 주말 **'
        f'{q[q.regime=="closed_weekend"].target_change_fraction.min()*100:.1f}~'
        f'{q[q.regime=="closed_weekend"].target_change_fraction.max()*100:.1f}%**다. '
        '레짐별 0수익률 비율과 변동성이 달라 r의 크기만으로 기대 수익을 비교할 수 없다.',
        '',
        '## 강건성·한계',
        '',
    ]
    phase=summary[(summary.regime.isin(('KR_cash','US_cash')))&
        (((summary.cadence_ms==500)&(summary.end_lag_ms==400))|
         ((summary.cadence_ms==1000)&(summary.end_lag_ms==100)))]
    phase_deltas=[]
    for cadence in (500,1000):
        pv=phase[phase.cadence_ms==cadence].pivot_table(
            index=['source','target','regime','feature','side'],
            columns='phase_ms',values='corr').dropna()
        phase_deltas.extend(np.abs(pv.iloc[:,1]-pv.iloc[:,0]).to_numpy())
    phase_max=float(np.max(phase_deltas))
    main=summary[(summary.phase_ms==0)&
        (((summary.cadence_ms==500)&(summary.end_lag_ms==400))|
         ((summary.cadence_ms==1000)&(summary.end_lag_ms==100)))]
    merged=main.merge(common,on=['source','target','cadence_ms','end_lag_ms',
                                  'regime','feature','side'],suffixes=('_all','_common'))
    common_max=float(np.max(np.abs(merged.corr_common-merged.corr_all)))
    lines += [
        f'- 500ms/1s 고정 시차에서 격자 위상을 반 간격 이동했을 때 국장·미장 r의 최대 절대 변화는 **{phase_max:.4f}**다. 정점 시각은 위상에 따라 100ms가량 이동하므로 400ms와 500ms를 별도 확정 지연으로 보지 않는다.',
        f'- 동일 HL 목표에 대해 두 Binance 소스가 **동시에 최신**인 관측만 남긴 공통표본 민감도에서 고정 시차 r의 최대 절대 변화는 **{common_max:.4f}**다. 원천별 유효표본 차이가 주된 장별 패턴을 만들었다는 증거는 약하다.',
        '- 12개 거래일의 블록 재표본은 날짜 간 변동을 반영하지만, 같은 18일에 발견한 시차·파라미터를 다시 검사한 탐색적 비교다. 현물 거래의 실제 주문흐름, 선후 인과, 거래 가능한 가격·대기열·체결 확률은 이 상관 분석에 포함되지 않는다.',
        '- 이전 [동일형 분석](REPORT_bn_hl_mid_micro_lag.md)과 [교차형 분석](REPORT_bn_hl_cross_type_lag.md)의 100ms 순차 예측은 HL 자체정보에 mid+micro를 더해도 MAE 개선을 확인하지 못했다. 현물 정규시간 중 더 높은 r만으로 fair value 신호나 maker 주문을 운용 단계로 승격하지 않는다.',
        '',
        '## 재현과 검산',
        '',
        '- 원천 100ms cache: `out_bn_hl_transfer/inputs/`; 분석: `work/bn_hl_cash_regime.py`; 요약: `work/bn_hl_cash_regime_report.py`; 공통표본: `work/bn_hl_cash_regime_common.py`; 독립 검산: `work/verify_bn_hl_cash_regime.py`.',
        f'- 18개 날짜별 파일 **{audit["rows"]:,}개** 비영(非零) 표본 모멘트 행, 상관곡선 **{audit["summary_rows"]:,}개**, 정점 **{audit["peak_rows"]}개**, 고정 시차 비교 **{audit["fixed_rows"]}개**. 달력상 국장 78h·미장 83h(첫날 전일 미장 tail 포함), 양시장 휴장 평일 142h·주말 129h를 확인했다.',
        f'- 원본 배열에서 직접 계산한 10/15 국장·500ms/+400ms·동일 국장형 mid→HL best bid r={audit["raw_kr_500ms_400ms_mid_bid_r"]:.6f}; 저장 모멘트와 오차 {audit["raw_check_absolute_error"]:.2g}. 국장 개폐·미장 새벽·토요일 새벽 경계 점검 통과.',
        '- 상세 산출물: [`response_summary.csv`](out_bn_hl_cash_regime/response_summary.csv), [`response_peaks.csv`](out_bn_hl_cash_regime/response_peaks.csv), [`fixed_regime_comparison.csv`](out_bn_hl_cash_regime/fixed_regime_comparison.csv), [`common_support_summary.csv`](out_bn_hl_cash_regime/common_support_summary.csv), [`verification.json`](out_bn_hl_cash_regime/verification.json), [`500ms 곡선 그림`](out_bn_hl_cash_regime/response_500ms_regimes.png).',
        '',
    ]
    (R.C.ROOT/'REPORT_bn_hl_cash_regime.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
