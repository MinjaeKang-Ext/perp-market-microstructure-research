"""Aggregate matched-pair cadence results and write a Korean evidence report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_cadence as C

OUT=C.OUT
REPORT=C.ROOT/'REPORT_bn_hl_cadence_transfer.md'
LABEL={C.PAIRS[0][1]:'SKHYNIXUSDT → SKHX',C.PAIRS[1][1]:'SKHYUSDT → SKHY'}


def fmt(value, digits=4):
    return '—' if pd.isna(value) else f'{value:.{digits}f}'


def table(rows, columns):
    lines=['| '+' | '.join(columns.values())+' |',
           '| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        lines.append('| '+' | '.join(str(row.get(key,'—')) for key in columns)+' |')
    return '\n'.join(lines)


def forecast_summary(daily):
    records=[]
    rng=np.random.default_rng(20261108)
    keys=['source','target','phase_ms','cadence_ms','horizon_ms','family','side']
    for group_kind in ('next_bin','fixed_60s'):
        frame=daily[daily.horizon_ms.eq(daily.cadence_ms if group_kind=='next_bin' else 60000)].copy()
        frame['horizon_kind']=group_kind
        frame['phase_index']=(frame.phase_ms!=0).astype(int)
        for (source,target,phase_index),block in frame.groupby(['source','target','phase_index']):
            piv=block.set_index(['cadence_ms','family','side','day']).sort_index()
            profiles=piv.index.droplevel('day').unique()
            days=list(C.DAYS[1:])
            n=np.empty((len(profiles),len(days)))
            gain=np.empty_like(n)
            own=np.empty_like(n)
            model=np.empty_like(n)
            for i,p in enumerate(profiles):
                sub=piv.loc[p].reindex(days)
                if sub[['n','gain_sum']].isna().any().any():
                    raise AssertionError('Missing date for forecast profile')
                n[i]=sub.n.to_numpy()
                gain[i]=sub.gain_sum.to_numpy()
                own[i]=sub.own_abs_sum.to_numpy()
                model[i]=sub.model_abs_sum.to_numpy()
            if len(profiles)!=8*len(C.FAMILIES)*2:
                raise AssertionError('Incomplete cadence/family/side profile grid')
            weight=rng.multinomial(len(days),np.ones(len(days))/len(days),size=2000).T
            point=gain.sum(axis=1)/n.sum(axis=1)
            draws=(gain@weight)/(n@weight)
            sd=draws.std(axis=1,ddof=1)
            standardized=np.abs((draws-point[:,None])/np.maximum(sd[:,None],1e-15))
            critical=np.quantile(np.max(standardized,axis=0),.95)
            for i,(cadence,family,side) in enumerate(profiles):
                phase_ms=0 if phase_index==0 else (int(cadence)//100//2)*100
                records.append(dict(source=source,target=target,phase_ms=phase_ms,
                    cadence_ms=int(cadence),horizon_ms=int(cadence if group_kind=='next_bin' else 60000),
                    horizon_kind=group_kind,family=family,side=side,
                    n=int(n[i].sum()),days=len(days),gain_bp=float(point[i]),
                    own_mae_bp=float(own[i].sum()/n[i].sum()),
                    model_mae_bp=float(model[i].sum()/n[i].sum()),
                    ci_low=float(np.quantile(draws[i],.025)),
                    ci_high=float(np.quantile(draws[i],.975)),
                    simultaneous_low=float(point[i]-critical*sd[i]),
                    improved_days=int(np.sum(gain[i]>0)),
                    decision_rows_per_day=float(n[i].mean())))
    result=pd.DataFrame(records)
    if len(result)!=2*2*2*8*len(C.FAMILIES)*2:
        raise AssertionError('Forecast summary row count')
    result.to_csv(OUT/'forecast_summary.csv',index=False)
    return result


def response_summary(daily):
    keys=['source','target','cadence_ms','phase_ms','feature','side','lag_bins','end_lag_ms']
    fields=['n','sx','sy','sxx','syy','sxy','source_change','target_change']
    result=daily.groupby(keys,as_index=False)[fields].sum()
    result['correlation']=C.corr(result)
    result['source_change_fraction']=result.source_change/result.n
    result['target_change_fraction']=result.target_change/result.n
    dates=daily.groupby(keys).day.nunique().to_numpy()
    if not np.all(dates==18):
        raise AssertionError('Incomplete response date grid')
    result['days']=dates
    result.to_csv(OUT/'response_summary.csv',index=False)
    return result


def coverage_summary(daily):
    keys=['source','target','cadence_ms','phase_ms','horizon_ms']
    result=daily.groupby(keys,as_index=False)[['grid_rows','valid_rows']].sum()
    result['valid_fraction']=result.valid_rows/result.grid_rows
    result.to_csv(OUT/'coverage_summary.csv',index=False)
    return result


def compact_response(response, target, feature, cadence, lag, phase=0):
    block=response[(response.target==target)&(response.feature==feature)
                   &(response.cadence_ms==cadence)&(response.lag_bins==lag)
                   &(response.phase_ms==phase)].set_index('side')
    if set(block.index)!=set(('bid','ask')):
        raise AssertionError('Response side missing')
    return f"{fmt(block.loc['bid','correlation'],3)}/{fmt(block.loc['ask','correlation'],3)}"


def joint_forecast(forecast):
    return forecast.groupby(['target','phase_ms','cadence_ms','horizon_kind','family'],as_index=False).agg(
        worst_gain=('gain_bp','min'),worst_low=('simultaneous_low','min'),
        min_positive_days=('improved_days','min'),min_n=('n','min'))


def figure(response,joint):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex='col')
    for row,target in enumerate(LABEL):
        ax=axes[row,0]
        for lag,style,label in ((0,'o-','same bin'),(1,'s-','next bin')):
            b=response[(response.target==target)&(response.phase_ms==0)
                       &(response.feature=='mid')&(response.side=='bid')
                       &(response.lag_bins==lag)].sort_values('cadence_ms')
            ax.plot(b.cadence_ms/1000,b.correlation,style,label=label,lw=1.6)
        ax.set_title(LABEL[target]+' | BN mid → HL best bid')
        ax.set_xscale('log')
        ax.set_ylim(-.05,1.05)
        ax.set_ylabel('Correlation')
        ax.grid(alpha=.2)
        ax.legend()
        ax=axes[row,1]
        for family,color in [('mid','#2471a3'),('micro','#17a589'),
                             ('obi','#8e44ad'),('ofi','#d35400')]:
            b=joint[(joint.target==target)&(joint.phase_ms==0)
                    &(joint.horizon_kind=='next_bin')&(joint.family==family)].sort_values('cadence_ms')
            ax.plot(b.cadence_ms/1000,b.worst_gain,'o-',color=color,label=family,lw=1.5)
        ax.axhline(0,color='black',lw=.8)
        ax.set_title(LABEL[target]+' | next-bin OOS MAE gain')
        ax.set_xscale('log')
        ax.set_ylabel('Worst bid/ask gain (bp)')
        ax.grid(alpha=.2)
        ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel('Actual alignment cadence Δ (seconds)')
    fig.suptitle('Same-bin co-movement increases with aggregation; prospective next-bin signal decays')
    fig.tight_layout()
    fig.savefig(OUT/'cadence_pattern.png',dpi=160)
    plt.close(fig)


def build():
    manifest=json.loads((OUT/'manifest.json').read_text())
    forecast_daily=pd.read_parquet(OUT/'forecast_daily.parquet')
    response_daily=pd.read_parquet(OUT/'response_daily.parquet')
    coverage_daily=pd.read_csv(OUT/'coverage_daily.csv')
    if len(forecast_daily)!=12240 or len(response_daily)!=51840 or len(coverage_daily)!=1080:
        raise AssertionError(f'Incomplete grid {len(forecast_daily)}/{len(response_daily)}/{len(coverage_daily)}')
    forecast=forecast_summary(forecast_daily)
    response=response_summary(response_daily)
    coverage=coverage_summary(coverage_daily)
    joint=joint_forecast(forecast)
    joint.to_csv(OUT/'forecast_worst_side.csv',index=False)
    figure(response,joint)
    lines=['# 같은 종목 Binance→Hyperliquid 정렬 간격별 최우선 호가 전이','',
       '2026-11-08. 2026-10-15~11-01 KST의 두 동일 종목 쌍만 사용했다. 첫날을 학습하고 이후17일은 전일 이전 자료만으로 순차 평가했다. 이미 사용된 연구 기간이므로 미사용 holdout은 아니다.','',
       '## 이 분석에서 바꾼 축','',
       '정렬 간격 Δ를 500ms,1,2,3,5,10,30,60초로 **실제로 바꿨다**. 각 시각의 양 거래소 호가는 수신시각까지 도착한 마지막 BBO이며, mid·잔량가중 micro의 Δ 구간 수익률과 OBI 수준·Δ변화, 해당 Δ 동안의 OFI를 다시 계산했다. OFI는 100ms 이벤트 순유입의 원래 수량을 합산한 뒤 Δ 끝의 표시잔량으로 정규화했다. 기존100ms 상태를 단순히 간격별로 건너뛰며 100ms OFI를 그대로 쓰지 않았다.','',
       '같은 구간(lag0)의 상관은 동행이다. Binance에서 이미 끝난 Δ구간을 보고 그다음 HL Δ구간(lag1)을 예측하는 것부터 선행 전이다. 30초·60초 격자에서는 수백 ms의 전이가 같은 구간에 합쳐져 세부 지연을 관측할 수 없다.','',
       '## 정렬 간격별 bid/ask 반응 패턴','',
       '각 셀은 `bid/ask` 상관계수다. 같은 구간과 다음 구간을 나란히 보인다. OFI는 Δ 구간의 순유입, OBI는 Δ 끝의 정적 불균형이다.','']
    for target in LABEL:
        rows=[]
        for cadence in C.CADENCES_MS:
            row={'cadence':f'{cadence/1000:g}s'}
            for feature,label in [('mid','mid'),('micro','micro'),('obi_level','OBI'),('ofi','OFI')]:
                row[label+'0']=compact_response(response,target,feature,cadence,0)
                row[label+'1']=compact_response(response,target,feature,cadence,1)
            rows.append(row)
        lines += [f'### {LABEL[target]}','',table(rows,{
            'cadence':'Δ','mid0':'mid 동행','mid1':'mid 다음','micro0':'micro 동행',
            'micro1':'micro 다음','OBI0':'OBI 동행','OBI1':'OBI 다음',
            'OFI0':'OFI 동행','OFI1':'OFI 다음'}),'']
    lines += ['**관측 패턴:** Mid의 같은 구간 상관은 두 쌍 모두500ms에서 약0.13이고60초에서는 약0.99까지 커진다. 이는 긴 구간에 이미 일어난 공통 가격이동이 합쳐진 결과다. 반대로 mid의 다음 구간 상관은500ms에서 약0.57,1초에서 약0.54,2초에서 약0.29,10초에서 약0.06,60초에서 약0.02~0.03으로 감소한다. 잔량가중 micro도 거의 같은 모양이다.','',
       'OBI 수준의 선행 상관은 mid보다 약하다. OFI의 같은 구간 상관은 거친 격자에서 오르지만 다음 구간 상관은 대체로 작아지고60초에서는 거의0이다. 따라서30초·60초 같은 구간 상관을 전이 지연이30초·60초 지속된다고 해석하지 않는다.','',
       f'![정렬 간격별 동행과 선행 전이]({OUT/"cadence_pattern.png"})','']
    lines += ['양의 다음 구간 상관이 있더라도 수수료·실제 주문 도달 후 가격·체결확률을 반영한 거래 수익은 아니다. 전체 −3~+5구간 곡선은 `response_summary.csv`에 있다.','',
       '## 과거 날짜 학습 예측: 다음 정렬 구간','',
       'HL 자체의 현재 bid·ask 변화, 스프레드, OBI, 구간 OFI를 기준선으로 둔다. 각 Binance 신호군을 추가해 **다음 Δ 구간**의 HL best bid/ask 예측 MAE가 얼마나 변하는지 같은 행에서 비교했다. 양수는 기준선보다 오차가 작다. 다음 구간의 길이도 Δ와 함께 달라지므로 이 표에서 절대오차 크기를 간격 간 직접 비교하지 않는다. 표의 값은 bid/ask 중 약한 측의 bp다.','']
    for target in LABEL:
        rows=[]
        for cadence in C.CADENCES_MS:
            b=joint[(joint.target==target)&(joint.phase_ms==0)
                    &(joint.cadence_ms==cadence)&(joint.horizon_kind=='next_bin')].set_index('family')
            cov=coverage[(coverage.target==target)&(coverage.phase_ms==0)
                         &(coverage.cadence_ms==cadence)&(coverage.horizon_ms==cadence)].iloc[0]
            row={'cadence':f'{cadence/1000:g}s','coverage':f'{cov.valid_fraction:.1%}',
                 'mid':fmt(b.loc['mid','worst_gain'],4),
                 'micro':fmt(b.loc['micro','worst_gain'],4),
                 'OBI':fmt(b.loc['obi','worst_gain'],4),
                 'OFI':fmt(b.loc['ofi','worst_gain'],4),
                 'joint':fmt(b.loc['joint','worst_gain'],4)}
            rows.append(row)
        lines += [f'### {LABEL[target]}','',table(rows,{
            'cadence':'Δ','coverage':'신선 표본','mid':'mid','micro':'micro',
            'OBI':'OBI','OFI':'OFI','joint':'결합'}),'']
    lines += ['다음 구간 예측 개선은 두 쌍 모두1초 정렬에서 가장 크고, 그 뒤 간격이 길어질수록 감소한다. 500ms와1초의 개선 차이는 예측 구간의 길이도 달라진 결과다. 60초 정렬의 micro 점추정은 소폭 양수지만 양쪽 호가의 동시 불확실성 하한은0보다 작다. 간격별 전일 모델·반 구간 위상 결과를 함께 보아야 한다.','']
    lines += ['각 모형의 bid·ask 별 점별95% 및 간격×신호군×양쪽 동시95% 하한, 날짜별 개선일 수는 `forecast_summary.csv`에 있다. 유효 표본은 양 거래소의 이전·현재 BBO와 미래 HL BBO의 **거래소 이벤트시각 기준 경과시간**이 모두 1초 이내일 때다. 이 기준은 HL 수신지연을 다시 더하지 않지만 거래소·수집기 시계 동기화가 검증됐다는 뜻은 아니다. 특히 저활동일에는 유효 비율이 크게 낮아 날짜별 커버리지도 함께 봐야 한다. 100ms 격자 행과 반복 상태를 독립 거래기회로 세지 않았다.','',
       '## 공통 60초 미래: 간격만 바꾼 비교','',
       '아래의 예측기간은 모든 행이 **현재부터 60초**로 같다. 60초 정렬에서는 다음 구간 평가와 동일하다. 표의 군은 약한 bid/ask 쪽 MAE 개선이 가장 큰 군이다. 동시 하한은 각 쌍·phase·예측구분 안에서 간격×군×양쪽을 함께 보정한 날짜 단위 bootstrap95% 하한이다.','']
    for target in LABEL:
        rows=[]
        for cadence in C.CADENCES_MS:
            b=joint[(joint.target==target)&(joint.phase_ms==0)
                    &(joint.cadence_ms==cadence)&(joint.horizon_kind=='fixed_60s')]
            best=b.sort_values('worst_gain',ascending=False).iloc[0]
            rows.append({'cadence':f'{cadence/1000:g}s','family':best.family,
                         'gain':fmt(best.worst_gain,5),'low':fmt(best.worst_low,5),
                         'days':int(best.min_positive_days)})
        lines += [f'### {LABEL[target]}','',table(rows,{
            'cadence':'Δ','family':'최대 점추정 군','gain':'약한 측 개선 bp',
            'low':'동시 하한','days':'최소 개선일/17'}),'']
    lines += ['군 선택과 간격 탐색을 같은18일에 반복했으므로 양수 동시 하한도 새 미사용 기간의 운용 승인은 아니다. 짧은 간격의60초 라벨은 서로 겹치므로 bootstrap은 인접 격자가 아닌 날짜 전체를 재표본화했다. 기준선 대비 MAE 개선은 체결 후 순수익이 아니다.','',
       '공통60초 평가에서는 두 쌍 모두500ms~10초 정렬의 상위군이 동시 하한 양수를 보인다. SKHX는30초부터, SKHY는60초부터 상위군 하한이0 아래로 내려간다. 이 패턴은 **현재 시점에서60초 뒤의 가격 예측**이며, 한 번 발생한 Binance 신호가10~30초 동안 주문 가능하게 남는다는 뜻이 아니다.','',
       '## 정렬 위상과 해석 범위','',
       '각 Δ에서 KST 날짜 정각 위상과 약 반 구간 이동 위상을 모두 계산했다. 500ms의 반 구간은 100ms 입력 해상도 때문에200ms 이동이다. 위 표는 정각 위상이며, 반 구간 전체 결과는 `forecast_summary.csv`, `response_summary.csv`, `coverage_summary.csv`에 함께 저장했다. 날짜별 전체 행과 정렬 규칙은 `forecast_daily.parquet`, `response_daily.parquet`, `manifest.json`에 있다.','',
       '시간 경계와 비경계에서 두 종목×양 거래소의 원본 BBO·OFI를64개 간격으로 직접 대조했고 독립 검산 결과는 `verification.json`에 남겼다.','',
       '이 결과의 Δ는 관측 정렬 간격이다. 앞선100ms 입력의 **추가 신호 나이 d** 실험에서 얻은 허용 지연과 바꿔 읽지 않는다. 1초 정렬의 다음 구간 예측이 가장 좋다는 결과만으로 HL 주문 ACK440ms 뒤의 알파를 입증할 수 없다.','',
       '시계 동기화, 실제 주문의 단방향 활성·취소 효력시각, 개인 주문 대기열·체결은 확인하지 않았다. 수신축 결과만을 주 판단으로 사용하며, 동행 상관과 실행 가능 알파를 구분한다.','']
    REPORT.write_text('\n'.join(lines))
    print('WROTE',REPORT, 'summary',len(forecast),len(response),len(coverage),flush=True)


if __name__=='__main__':
    build()
