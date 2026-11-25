"""Readable research report and figures for the funded-window spot fair value study."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from work import skh_spot_grid as G
from work import skh_spot_fv_transfer as T

LABEL={
    'binance_SKHYNIXUSDT':'Binance 국장형 SKHYNIXUSDT',
    'hyperliquid_SKHX':'HL 국장형 SKHX',
    'binance_SKHYUSDT':'Binance ADR형 SKHYUSDT',
    'hyperliquid_SKHY':'HL ADR형 SKHY',
}
PLOT_LABEL={
    'binance_SKHYNIXUSDT':'Binance common SKHYNIXUSDT',
    'hyperliquid_SKHX':'HL common SKHX',
    'binance_SKHYUSDT':'Binance ADR SKHYUSDT',
    'hyperliquid_SKHY':'HL ADR SKHY',
}


def fmt(x,d=3):return f'{x:.{d}f}'


def figure(summary,inference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    colors={'binance_SKHYNIXUSDT':'#e28e20','hyperliquid_SKHX':'#0064a7',
            'binance_SKHYUSDT':'#cc552a','hyperliquid_SKHY':'#43a5c6'}
    for ri,segment in enumerate(('regular_0900_1520','late_1600_2000')):
        for ci,feature in enumerate(('mid','micro')):
            ax=axes[ri,ci]
            for target in T.TARGETS:
                b=summary[(summary.target==target)&(summary.segment==segment)&
                    (summary.cadence_ms==500)&(summary.phase_ms==0)&
                    (summary.feature==feature)&(summary.side=='mid')].sort_values('end_lag_ms')
                ax.plot(b.end_lag_ms/1000,b['corr'],lw=1.3,
                        color=colors[target],label=PLOT_LABEL[target])
            ax.axhline(0,color='black',lw=.5)
            ax.axvline(0,color='black',lw=.7)
            ax.set_title(('09:00–15:20' if ri==0 else '16:00–20:00')+f' | spot {feature}')
            ax.set_xlabel('Perp 100ms mid return end lag after spot bin (s)')
            ax.set_ylabel('Pearson r')
    axes[0,0].legend(loc='upper left',fontsize=7)
    fig.tight_layout()
    fig.savefig(G.OUT/'response_500ms_spot_perp.png',dpi=160)
    plt.close(fig)
    q=inference[(inference.segment=='regular_0900_1520')&
                (inference.metric=='micro_vs_own_mse')].set_index('target').loc[list(T.TARGETS)]
    fig,ax=plt.subplots(figsize=(9,4))
    x=np.arange(len(q));value=q.estimate.to_numpy()*100
    low=q.simul_low.to_numpy()*100;high=q.simul_high.to_numpy()*100
    ax.bar(x,value,color=[colors[t] for t in q.index])
    ax.errorbar(x,value,yerr=[value-low,high-value],fmt='none',ecolor='black',capsize=4)
    ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(x,[PLOT_LABEL[t].replace(' ','\n',1) for t in q.index])
    ax.set_ylabel('OOS 1s mid MSE reduction vs own-book model (%)')
    ax.set_title('Spot micro 500ms correction | 09:00–15:20 KST | 11 prior-day tests')
    fig.tight_layout()
    fig.savefig(G.OUT/'fv_oos_micro_regular.png',dpi=160)
    plt.close(fig)


def main():
    out=G.OUT
    audit=json.loads((out/'verification.json').read_text())
    assert audit['pass_'] and audit['independent_oos_sse_error']<1e-7
    summary=pd.read_csv(out/'response_summary.csv')
    overall=pd.read_csv(out/'response_overall_0900_2000.csv')
    peaks=pd.read_csv(out/'response_peaks.csv')
    selected=pd.read_csv(out/'selected_response_validation.csv')
    forecast=pd.read_csv(out/'forecast_summary.csv')
    inference=pd.read_csv(out/'forecast_inference.csv')
    coef=pd.read_csv(out/'fv_coefficients_all_days.csv')
    funding=pd.read_csv(out/'funding_mask_audit.csv')
    spot_audit=json.loads((out/'spot_grid_audit.json').read_text())
    figure(summary,inference)
    lines=[
        '# 하이닉스 현물 000660→Binance·Hyperliquid 퍼프 공정가치 통계',
        '',
        '## 핵심 결론',
        '',
        '- 분석 범위는 2026-10-15~10-30의 **현물 자료가 있는 12거래일**, 매일 한국시간 **09:00≤t<20:00**다. 현물 000660의 mid·L1 잔량가중 microprice를 Binance/HL의 국장형·ADR형 네 퍼프에 연결했다.',
        '- 실제 Binance·HL 펀딩 정산 기록 **4개 계약 모두의 ±30초 창을 합집합으로 제외**했다. 20:00 정산 직전 30초까지 포함해 기본 100ms 시각은 일별 6,600~6,601칸(약 11분) 제외됐고, 신호·반응 구간이 그 창을 가로지르는 관측도 추가 제외했다.',
        '- **09:00–15:20 연속 호가 구간**에서는 현물 500ms micro 변화와 HL 100ms mid 반응의 정점이 SKHX/SKHY 모두 **+200ms**, 전체 12일 r **0.267/0.242**다. 첫 6일에서 고른 정점을 마지막 6일에 그대로 적용한 r은 **0.260/0.243**이다.',
        '- Binance 국장형/ADR형은 같은 구간 micro의 양의 미래 정점이 **+100ms, r 0.056/0.064**로 작고, 전체 ±5초 정점은 **−400/−300ms**다. 수신시각 기준으로 Binance 가격이 현물 신호 완료보다 먼저 반응한 형태다.',
        '- 과거 날짜만으로 학습한 1초 공정가치 보정은 정규 호가 구간에서 현물 micro를 더했을 때 HL 자체 호가·OFI 기준선 대비 SKHX **13.92%**, SKHY **10.69%**의 MSE 감소를 보였다. 11/11 평가일 모두 양수이며 40개 지표 동시 95% 하한은 **11.59%/8.93%**다. Binance 두 계약의 개선은 **0.06~0.08%**로 동시 구간에 0이 포함된다.',
        '- HL에서 관측한 +200ms 수신축 정점은 사용자가 실측한 HL 시세 호출 지연 중앙값 240ms보다 짧다. 따라서 위 개선을 거래소의 실제 미반영 가격이나 maker 주문의 실행 가능 수익으로 해석하지 않는다.',
        '',
        '## 입력과 제외 기준',
        '',
        '현물은 키움 `000660`의 수신시각 `received_at`으로 정렬했다. 현물 BBO의 수량 필드는 결측이므로 **L2 1호가**를 주 입력으로 하고 그 사이의 BBO를 보조로 썼다. Mid=(bid+ask)/2, micro=(ask×bid잔량+bid×ask잔량)/(양쪽 잔량 합)이다. 현물의 독립 거래소 발생시각은 저장본에 없어 현물 수신시각을 경제적 발생시각으로 바꾸지 않았다. 퍼프는 기존 검증된 Binance·HL 100ms 수신시각 BBO 캐시를 사용했다. 현물과 퍼프 양 끝의 호가 나이는 ≤1초로 제한했다.',
        '',
        '| 구간 (KST) | 관측 형태 | 공정가치 평가에서의 처리 |',
        '|---|---|---|',
        '| 09:00–15:20 | 연속 호가 | 주 분석 |',
        '| 15:20–15:30 | 종가 단일가 호가 | 상관만 별도 진단, 공정가치 후보 순위에서 제외 |',
        '| 15:30–16:00 | 지속 호가 거의 없음 | 신선도 통과분 수백 관측만 별도 표시, 공정가치 후보 제외 |',
        '| 16:00–20:00 | 장후 키움 호가 스트림 | 별도 분석. 저장 FID로 KRX/NXT 전용 호가를 식별할 수 없음 |',
        '',
        '펀딩 시간은 로컬 역사 원장의 `fundingTime`(Binance)·`time`(HL) epoch ms를 그대로 썼다. 이 기간 KST 09:00–20:00의 하루당 Binance 국장형 09·13·17시, Binance ADR형 09·17시, HL 두 계약은 09~20시 매시 정산 경계가 있다. 정산 시각은 정각에서 수 ms~139ms 벗어날 수 있으므로 정각으로 반올림하지 않았다. [Binance API의 fundingTime 필드](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data), [Hyperliquid의 매시간 정산 규칙](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding)에 맞췄다.',
        '',
        '## 500ms 현물 변화→퍼프 100ms 최우선 가격',
        '',
        'Δ=500ms 현물 구간 `(t−500ms,t]`의 로그수익률과 퍼프 `(t+L−100ms,t+L]`의 로그수익률 간 Pearson r이다. “전체 정점”이 음수이면 현물 신호 완료 전에 퍼프 가격 변화가 더 강하게 관측됐다는 뜻이다. 첫 6거래일 발견·마지막 6거래일 확인은 **mid 목표 가격의 양의 미래 정점 L만** 첫 기간에서 선택해 고정했다.',
        '',
        '| 퍼프 목표 | 시간 구간 | 현물 입력 | 전체 12일 미래 정점 L / mid r | 전체 정점 L / r | 발견 6일 L / r | 확인 6일 r |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for target in T.TARGETS:
        for seg in ('regular_0900_1520','late_1600_2000'):
            for feature in ('mid','micro'):
                p=peaks[(peaks.target==target)&(peaks.cadence_ms==500)&
                    (peaks.segment==seg)&(peaks.feature==feature)&(peaks.side=='mid')].iloc[0]
                s=selected[(selected.target==target)&(selected.cadence_ms==500)&
                    (selected.segment==seg)&(selected.feature==feature)].iloc[0]
                lines.append(f'| {LABEL[target]} | {"09–15:20" if seg.startswith("regular") else "16–20"} | '
                    f'{feature} | +{int(p.future_peak_ms)}ms / {fmt(p.future_peak_corr)} | '
                    f'{int(p.absolute_peak_ms):+d}ms / {fmt(p.absolute_peak_corr)} | '
                    f'+{int(s.discovery_peak_ms)}ms / {fmt(s.discovery_r)} | {fmt(s.validation_r)} |')
    lines += [
        '',
        'HL 정규구간의 best bid/ask도 현물 micro 기준으로 각각 SKHX **r 0.234/0.229**, SKHY **0.212/0.207**, 모두 +200ms에서 정점이다. 현물 mid 입력보다 micro 입력이 강하다. 장후 16–20시 HL micro→mid 정점은 SKHX **r 0.054**, SKHY **0.026**으로 약해진다. Binance는 정규구간에도 양의 미래 r이 0.06 안팎이며, 전체 정점은 현물 신호보다 앞선다.',
        '',
        '### 8개 정렬 간격의 확인기간 결과',
        '',
        '아래는 **현물 micro→퍼프 100ms mid**에 대해 첫 6일에서 고른 미래 정점 시차를 뒤 6일에 고정한 r이다. 장후는 약한 상관과 빈번한 긴 구간 정점 때문에 [전체 CSV](out_skh_spot_fv/selected_response_validation.csv)에 별도 기록했다.',
        '',
        '| Δ | Binance SKHYNIXUSDT | HL SKHX | Binance SKHYUSDT | HL SKHY |',
        '|---:|---:|---:|---:|---:|',
    ]
    for cadence in T.C.CADENCES_MS:
        cells=[]
        for target in T.TARGETS:
            s=selected[(selected.target==target)&(selected.cadence_ms==cadence)&
                (selected.segment=='regular_0900_1520')&(selected.feature=='micro')].iloc[0]
            cells.append(f'+{int(s.discovery_peak_ms)}ms / {fmt(s.validation_r)}')
        lines.append(f'| {cadence/1000:g}s | '+' | '.join(cells)+' |')
    lines += [
        '',
        '긴 Δ의 100ms 반응 r은 HL에서 대체로 작아지고 미래 정점 위치도 불안정하다. 정렬 간격을 30~60초로 늘려 얻은 몇몇 Binance r은 반응이 수초 뒤에 분산된 결과이므로 주문 제출 가능 시차로 읽지 않는다. 전체 시차곡선·표본 수·회귀기울기는 [`response_summary.csv`](out_skh_spot_fv/response_summary.csv)에 있다.',
        '',
        '## 공정가치 보정과 날짜순 검증',
        '',
        '하나의 구체적 공정가치 후보는 `FV_t(1s) = PerpMid_t × exp(ŷ_t/10,000)`이다. ŷ는 1초 뒤 퍼프 mid 로그수익률(bp)의 과거 날짜 선형 예측값이다. 기준선은 해당 퍼프의 직전 Δ bid·ask 변화, 현재 스프레드·잔량 불균형, Δ구간 OFI다. 현물 mid, 현물 micro, 둘의 결합을 각각 **기준선과 동일한 유효 시각**에 더했다. 매일 직전 날짜까지만 학습하고 다음 날을 평가했다(첫날은 학습만, 뒤 11일 검증). 아래는 사전 지정한 Δ=500ms·1초 목표의 phase 0이다.',
        '',
        '| 퍼프 목표 | 구간 | 현물 micro 추가 MSE 감소 | 40항목 동시 95% | 현물 mid 추가 MSE 감소 | micro의 mid 대비 추가 감소 | micro MAE 개선(bp/관측) | 유효 표본 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for target in T.TARGETS:
        for seg in ('regular_0900_1520','late_1600_2000'):
            q=inference[(inference.target==target)&(inference.segment==seg)].set_index('metric')
            micro=q.loc['micro_vs_own_mse'];mid=q.loc['mid_vs_own_mse']
            extra=q.loc['micro_vs_mid_mse'];mae=q.loc['micro_vs_own_mae_bp']
            lines.append(f'| {LABEL[target]} | {"09–15:20" if seg.startswith("regular") else "16–20"} | '
                f'{micro.estimate*100:.3f}% | [{micro.simul_low*100:.3f}, {micro.simul_high*100:.3f}]% | '
                f'{mid.estimate*100:.3f}% | {extra.estimate*100:.3f}%p | '
                f'{mae.estimate:.4f} | {int(micro.n):,} |')
    lines += [
        '',
        '위 95% 구간은 날짜 11개를 같은 블록으로 5,000회 재표본하고 표의 8경우×5척도=40항목에 동시 보정했다. 정규구간 micro의 MSE 개선은 HL 두 계약 모두 11/11 평가일 양수였고, Binance 두 계약은 동시 구간에 0이 포함된다. **1초 MAE도 HL SKHX +0.0796bp, HL SKHY +0.0552bp/관측**으로 양수다. Binance는 MAE 개선이 0을 넘는다고 할 수 없다.',
        '같은 500ms 현물 micro 입력의 **5초 목표**에서는 정규구간 OOS MSE 감소가 HL SKHX **2.69%**, HL SKHY **2.19%**로 줄고 Binance 두 계약은 **0.02~0.03%**다. 1초에서 보인 관계를 5초까지 같은 강도로 연장할 수 없다.',
        '',
        '12일 전부로 다시 적합한 다음 기간용 계수에서 Δ=500ms 현물 micro 1bp 상승이 1초 목표 ŷ에 더하는 값은 HL SKHX **+0.4595bp**, HL SKHY **+0.3322bp**, Binance 국장형 **+0.0425bp**, Binance ADR형 **+0.0347bp**다. 이 계수는 다른 퍼프 자체정보를 통제한 값이며, 전체식의 절편·자체호가 계수는 [`fv_coefficients_all_days.csv`](out_skh_spot_fv/fv_coefficients_all_days.csv)에 있다. 12일 전체 적합 계수 자체는 미사용 표본 성과가 아니다.',
        '',
        '## 시간대·수준가격 해석',
        '',
    ]
    aggregate=overall[(overall.cadence_ms==500)&(overall.phase_ms==0)&
        (overall.feature=='micro')&(overall.side=='mid')]
    lines += [
        '09:00–20:00의 모든 **유효** 구간을 단순 합산한 500ms 현물 micro→100ms 퍼프 mid r은 다음과 같다. 이 값은 연속매매와 장후를 섞으므로 FV 후보 선택 기준으로 쓰지 않았다.',
        '',
        '| 퍼프 | +100ms r | +200ms r | +300ms r |',
        '|---|---:|---:|---:|',
    ]
    for target in T.TARGETS:
        values=[]
        for lag in (100,200,300):
            r=aggregate[(aggregate.target==target)&(aggregate.end_lag_ms==lag)].iloc[0]
            values.append(fmt(r['corr']))
        lines.append(f'| {LABEL[target]} | '+' | '.join(values)+' |')
    lines += [
        '',
        '현물은 KRW 보통주이고 퍼프는 USDT 명목계약이다. 환율·계약비율·ADR 환산이 확인되지 않아 **현물과 퍼프의 절대가격 차이를 공정가치 괴리로 계산하지 않았다**. 위 FV는 퍼프 자신의 현재 mid를 앵커로 한 **수익률 보정**이다. 특히 ADR형 두 계약은 보통주 현물에서 교차형으로 연결했으므로 수준가격 FV로 해석하지 않는다.',
        '',
        'HL 시세 데이터 호출 지연 중앙값 240ms와 HL 주문 제출 확정 응답 중앙값 440ms(사용자 실측)를 고려하면, +200ms 수신축 반응이 남아 있다고 해서 주문이 거래소에 도달할 때까지 가격차가 유지된다고 볼 수 없다. 현물 발생시각도 따로 없으며 자체 체결·대기열·펀딩 현금흐름·수수료를 반영한 전략 검증이 아니다. 장후 16–20시 호가의 KRX/NXT 시장 구분 역시 저장 자료만으로 확인되지 않는다.',
        '',
        '## 검증·파일',
        '',
        f'- 12일 현물 as-of 원천 이벤트 합계 **{sum(r["source_events"] for r in spot_audit):,}개**; 100ms 현물 캐시 12개와 퍼프 4개 BBO 캐시로 정렬했다. 전체 반응 모멘트 **{audit["daily_rows"]:,}행**, 요약 **{audit["response_rows"]:,}행**이다.',
        f'- 펀딩 기본 격자 제외 건수는 매일 6,600~6,601/396,000이며, 20:00 이전 경계·매시 정산·전 구간 교차 제외를 점검했다. 독립 원천 호가 3시점과 수익률 상관 모멘트의 오차 **{audit["response_moment_error"]:.2g}**, 과거 날짜 OLS 제곱오차 재구성 차이 **{audit["independent_oos_sse_error"]:.2g}**이다.',
        '- 계산: `work/skh_spot_grid.py`, `work/skh_spot_fv_transfer.py`, `work/skh_spot_fv_summary.py`, `work/skh_spot_fv_forecast.py`, `work/skh_spot_fv_inference.py`, `work/verify_skh_spot_fv.py`.',
        '- 결과: [`response_summary.csv`](out_skh_spot_fv/response_summary.csv), [`response_peaks.csv`](out_skh_spot_fv/response_peaks.csv), [`selected_response_validation.csv`](out_skh_spot_fv/selected_response_validation.csv), [`forecast_summary.csv`](out_skh_spot_fv/forecast_summary.csv), [`forecast_inference.csv`](out_skh_spot_fv/forecast_inference.csv), [`funding_mask_audit.csv`](out_skh_spot_fv/funding_mask_audit.csv), [`verification.json`](out_skh_spot_fv/verification.json), [`반응곡선`](out_skh_spot_fv/response_500ms_spot_perp.png), [`OOS FV 비교`](out_skh_spot_fv/fv_oos_micro_regular.png).',
        '',
    ]
    (G.C.ROOT/'REPORT_skh_spot_fv_09_20.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
