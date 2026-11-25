"""Summarize mid+micro physical response and prior-day OOS lagged prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from work import bn_hl_mid_micro_lag as M
from work import bn_hl_cadence as C

OUT=M.OUT
REPORT=C.ROOT/'REPORT_bn_hl_mid_micro_lag.md'
LABEL={C.PAIRS[0][1]:'SKHYNIXUSDT → SKHX',C.PAIRS[1][1]:'SKHYUSDT → SKHY'}
MOMENT=('n','sx','sy','sxx','syy','sxy')


def corr_from(a):
    n,sx,sy,sxx,syy,sxy=[a[...,i] for i in range(6)]
    vx=sxx-sx*sx/n
    vy=syy-sy*sy/n
    cov=sxy-sx*sy/n
    return np.divide(cov,np.sqrt(np.maximum(vx*vy,0)),
        out=np.full_like(cov,np.nan,dtype=float),where=(vx>1e-20)&(vy>1e-20))


def response_summary(daily,out=OUT):
    keys=['source','target','cadence_ms','phase_ms','feature','side','end_lag_ms','lag_samples']
    cols=list(MOMENT)+['source_nonzero','target_nonzero']
    frame=daily.groupby(keys,as_index=False)[cols].sum()
    frame['corr']=corr_from(frame[list(MOMENT)].to_numpy())
    frame['days']=daily.groupby(keys).day.nunique().to_numpy()
    if not frame.days.eq(18).all():
        raise AssertionError('Descriptive dates incomplete')
    frame['simul_low']=np.nan
    rng=np.random.default_rng(20261108)
    for (source,target,cadence),block in daily[(daily.phase_ms==0)&(daily.end_lag_ms>0)].groupby(
            ['source','target','cadence_ms']):
        keys2=['feature','side','end_lag_ms']
        indexed=block.set_index(keys2+['day']).sort_index()
        profiles=indexed.index.droplevel('day').unique()
        if len(profiles)!=2*2*50:
            raise AssertionError('Positive lag profiles incomplete')
        data=np.empty((len(profiles),18,6))
        for i,p in enumerate(profiles):
            sub=indexed.loc[p].reindex(C.DAYS)
            if sub[list(MOMENT)].isna().any().any():
                raise AssertionError('Missing response date')
            data[i]=sub[list(MOMENT)].to_numpy()
        weights=rng.multinomial(18,np.ones(18)/18,size=1000).T
        totals=data.sum(axis=1)
        point=corr_from(totals)
        boot=np.stack([data[:,:,j]@weights for j in range(6)],axis=-1)
        draws=corr_from(boot)
        sd=np.nanstd(draws,axis=1,ddof=1)
        z=np.abs((draws-point[:,None])/np.maximum(sd[:,None],1e-15))
        crit=float(np.nanquantile(np.nanmax(z,axis=0),.95))
        low=point-crit*sd
        lookup={p:low[i] for i,p in enumerate(profiles)}
        mask=(frame.source==source)&(frame.target==target)&(frame.cadence_ms==cadence)&(frame.phase_ms==0)&(frame.end_lag_ms>0)
        frame.loc[mask,'simul_low']=[lookup[(r.feature,r.side,r.end_lag_ms)]
                                     for r in frame.loc[mask].itertuples()]
    frame.to_csv(out/'response_summary.csv',index=False)
    return frame


def forecast_summary(daily,out=OUT):
    rows=[]
    rng=np.random.default_rng(20261108)
    for (source,target,cadence,phase),block in daily.groupby(
            ['source','target','cadence_ms','phase_ms']):
        indexed=block.set_index(['end_lag_ms','family','side','day']).sort_index()
        profiles=indexed.index.droplevel('day').unique()
        if len(profiles)!=len(M.FORECAST_LAGS)*len(M.FAMILIES)*2:
            raise AssertionError('Forecast profile count')
        days=C.DAYS[1:]
        n=np.empty((len(profiles),len(days)))
        gain=np.empty_like(n); increment=np.empty_like(n)
        own=np.empty_like(n); model=np.empty_like(n)
        for i,p in enumerate(profiles):
            sub=indexed.loc[p].reindex(days)
            if sub[['n','gain_vs_own_sum','gain_vs_mid_sum']].isna().any().any():
                raise AssertionError('Missing OOS date')
            n[i]=sub.n.to_numpy()
            gain[i]=sub.gain_vs_own_sum.to_numpy()
            increment[i]=sub.gain_vs_mid_sum.to_numpy()
            own[i]=sub.own_abs_sum.to_numpy()
            model[i]=sub.model_abs_sum.to_numpy()
        weights=rng.multinomial(len(days),np.ones(len(days))/len(days),size=2000).T
        point=gain.sum(axis=1)/n.sum(axis=1)
        draws=(gain@weights)/(n@weights)
        sd=draws.std(axis=1,ddof=1)
        crit=np.quantile(np.max(np.abs((draws-point[:,None])/
                               np.maximum(sd[:,None],1e-15)),axis=0),.95)
        extra_point=increment.sum(axis=1)/n.sum(axis=1)
        extra_draws=(increment@weights)/(n@weights)
        extra_sd=extra_draws.std(axis=1,ddof=1)
        both=np.array([p[1]=='both' for p in profiles])
        extra_crit=np.quantile(np.max(np.abs((extra_draws[both]-extra_point[both,None])/
                   np.maximum(extra_sd[both,None],1e-15)),axis=0),.95)
        for i,(lag,family,side) in enumerate(profiles):
            rows.append(dict(source=source,target=target,cadence_ms=int(cadence),
                phase_ms=int(phase),end_lag_ms=int(lag),family=family,side=side,
                n=int(n[i].sum()),days=17,gain_vs_own_bp=float(point[i]),
                gain_vs_own_simul_low=float(point[i]-crit*sd[i]),
                gain_vs_mid_bp=float(extra_point[i]),
                gain_vs_mid_simul_low=float(extra_point[i]-extra_crit*extra_sd[i]) if family=='both' else np.nan,
                improved_days=int(np.sum(gain[i]>0)),
                own_mae_bp=float(own[i].sum()/n[i].sum()),
                model_mae_bp=float(model[i].sum()/n[i].sum())))
    result=pd.DataFrame(rows)
    result.to_csv(out/'forecast_summary.csv',index=False)
    return result


def response_peaks(response,out=OUT):
    rows=[]
    for key,b in response[response.phase_ms==0].groupby(['source','target','cadence_ms','feature','side']):
        future=b[b.end_lag_ms>0].sort_values('end_lag_ms')
        best=future.loc[future['corr'].idxmax()]
        absolute=b.loc[b['corr'].abs().idxmax()]
        passed=future.simul_low.to_numpy()>0
        prefix=int(future.end_lag_ms.iloc[np.flatnonzero(~passed)[0]-1]) if np.any(~passed) and np.flatnonzero(~passed)[0]>0 else (int(future.end_lag_ms.iloc[-1]) if np.all(passed) else 0)
        rows.append(dict(zip(['source','target','cadence_ms','feature','side'],key))|
            dict(all_abs_peak_ms=int(absolute.end_lag_ms),
                 all_abs_peak_corr=float(absolute['corr']),
                 future_peak_ms=int(best.end_lag_ms),future_peak_corr=float(best['corr']),
                 future_peak_simul_low=float(best.simul_low),
                 continuous_positive_through_ms=prefix,
                 future_positive_bins=int(np.count_nonzero(passed))))
    result=pd.DataFrame(rows)
    result.to_csv(out/'response_peaks.csv',index=False)
    return result


def forecast_peaks(forecast,out=OUT):
    both=forecast[(forecast.family=='both')&(forecast.phase_ms==0)]
    joint=both.groupby(['source','target','cadence_ms','end_lag_ms'],as_index=False).agg(
        worst_gain=('gain_vs_own_bp','min'),
        worst_low=('gain_vs_own_simul_low','min'),
        worst_increment=('gain_vs_mid_bp','min'),
        worst_increment_low=('gain_vs_mid_simul_low','min'),
        min_improved_days=('improved_days','min'),min_n=('n','min'))
    peaks=joint.loc[joint.groupby(['source','target','cadence_ms']).worst_gain.idxmax()].copy()
    joint.to_csv(out/'forecast_joint_lag_curve.csv',index=False)
    peaks.to_csv(out/'forecast_joint_peaks.csv',index=False)
    return joint,peaks


def figure(response,forecast,out=OUT,labels=LABEL):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(len(labels),2,figsize=(12,3.5*len(labels)),sharex='col')
    for ri,target in enumerate(labels):
        for ci,feature in enumerate(('mid','micro')):
            ax=axes[ri,ci]
            b=response[(response.target==target)&(response.phase_ms==0)
                       &(response.feature==feature)&(response.side=='bid')]
            matrix=b.pivot(index='cadence_ms',columns='end_lag_ms',values='corr').loc[list(C.CADENCES_MS)]
            image=ax.imshow(matrix.to_numpy(),aspect='auto',origin='lower',
                            extent=[-5.05,5.05,-.5,7.5],cmap='coolwarm',
                            norm=TwoSlopeNorm(vmin=-.1,vcenter=0,vmax=.32))
            ax.axvline(0,color='black',lw=.9)
            ax.set_yticks(range(8),[f'{v/1000:g}s' for v in C.CADENCES_MS])
            ax.set_title(labels[target]+' | '+feature+' → best bid')
            ax.set_xlabel('100ms best-bid return end lag after BN bin (s)')
            ax.set_ylabel('BN source interval Δ')
    fig.subplots_adjust(left=.10,right=.88,bottom=.10,top=.94,wspace=.30,hspace=.30)
    bar=fig.add_axes([.91,.20,.018,.58])
    fig.colorbar(image,cax=bar,label='Pooled correlation')
    fig.savefig(out/'response_heatmap.png',dpi=160)
    plt.close(fig)


def fmt(v,d=4):
    return '—' if pd.isna(v) else f'{v:.{d}f}'


def table(rows,columns):
    out=['| '+' | '.join(columns.values())+' |','| '+' | '.join('---' for _ in columns)+' |']
    out.extend('| '+' | '.join(str(r.get(k,'—')) for k in columns)+' |' for r in rows)
    return '\n'.join(out)


def build():
    manifest=json.loads((OUT/'manifest.json').read_text())
    rd=pd.read_parquet(OUT/'response_daily.parquet')
    fd=pd.read_parquet(OUT/'forecast_daily.parquet')
    if len(rd)!=18*2*8*2*101*2*2 or len(fd)!=17*2*8*2*len(M.FORECAST_LAGS)*3*2:
        raise AssertionError(f'Incomplete raw outputs {len(rd)}, {len(fd)}')
    response=response_summary(rd)
    forecast=forecast_summary(fd)
    peaks=response_peaks(response)
    joint,joint_peaks=forecast_peaks(forecast)
    figure(response,joint)
    lines=['# Binance mid+microprice를 함께 사용한 HL best bid/ask 전이 시각','',
      '2026-11-08. 동일 종목 두 쌍만 비교했다: SKHYNIXUSDT→SKHX, SKHYUSDT→SKHY. 2026-10-15~11-01 KST의 수신시각 BBO를 사용했으며 첫날 학습·이후17일 순차 평가다. 이미 연구에 쓴 기간이라 미사용 holdout은 아니다.','',
      '## 측정 기준','',
      'Binance 신호는 각 Δ∈{500ms,1,2,3,5,10,30,60초}의 완성된 `(t−Δ,t]` 구간 mid와 L1 잔량가중 microprice 변화다. 두 신호의 결합은 `mid 변화`와 `micro 변화−mid 변화`를 함께 회귀에 넣은 것과 같다. 추가 항은 사실상 스프레드×OBI/2의 변화이므로 독립된 두 가격원천으로 해석하지 않는다.','',
      'HL best bid·ask의100ms 수익률 구간 `(t+L−100ms,t+L]`을 L=−5~+5초,100ms 간격으로 측정했다. 표의 L은 **가격변화 구간의 끝**이다. L≤0은 신호를 완성하기 전에 이미 발생한 반응이고, L≥100ms부터 미래 구간이다. Δ(신호 생성 구간)와 L(전이 시차)은 별개의 숫자다.','',
      '## 간격별 전이 피크','',
      '전체 피크는 −5~+5초 중 절대 상관 최대 지점, 미래 피크는 +100~+5000ms 중 양의 상관 최대 지점이다. 미래 연속 구간은 +100ms부터 bid/ask·mid/micro의 날짜 bootstrap 동시95% 하한이 양수인 마지막 연속 지점이다. 이것은 관측된 연관 지속구간이며 주문 가능한 신호 수명과 같지 않다.','']
    for target in LABEL:
        rows=[]
        for cadence in C.CADENCES_MS:
            p=peaks[(peaks.target==target)&(peaks.cadence_ms==cadence)].set_index(['feature','side'])
            f=joint_peaks[(joint_peaks.target==target)&(joint_peaks.cadence_ms==cadence)].iloc[0]
            def peak_cell(feature,kind):
                time_col='all_abs_peak_ms' if kind=='all' else 'future_peak_ms'
                corr_col='all_abs_peak_corr' if kind=='all' else 'future_peak_corr'
                return ' / '.join(f"{int(p.loc[(feature,side),time_col])}ms ({p.loc[(feature,side),corr_col]:.3f})"
                                  for side in ('bid','ask'))
            row={'cadence':f'{cadence/1000:g}s',
                 'mid_all':peak_cell('mid','all'),
                 'micro_all':peak_cell('micro','all'),
                 'mid_future':peak_cell('mid','future'),
                 'micro_future':peak_cell('micro','future'),
                 'continuous':int(p.continuous_positive_through_ms.min()),
                 'model_lag':int(f.end_lag_ms) if f.worst_gain>0 and f.worst_low>0 else '통과 없음',
                 'gain':fmt(f.worst_gain,5),
                 'low':fmt(f.worst_low,5),'increment':fmt(f.worst_increment,5),
                 'increment_low':fmt(f.worst_increment_low,5)}
            rows.append(row)
        lines += [f'### {LABEL[target]}','',table(rows,{
            'cadence':'BN Δ','mid_all':'mid 전체 피크 bid/ask (시차,r)',
            'micro_all':'micro 전체 피크 bid/ask (시차,r)',
            'mid_future':'mid 미래 피크 bid/ask (시차,r)',
            'micro_future':'micro 미래 피크 bid/ask (시차,r)',
            'continuous':'네 곡선 연속 양의 구간 ms',
            'model_lag':'결합모형 최상 L ms','gain':'HL 자체 대비 개선 bp',
            'low':'동시 하한','increment':'mid 대비 추가 bp',
            'increment_low':'추가분 동시 하한'}),'']
    lines += ['**핵심 시간 패턴:** Δ=500ms일 때 mid와 micro의 미래 반응 정점은 두 계약의 bid·ask 모두 신호 종료 **400ms 뒤**다(상관 약0.30~0.31). 이는 정점의 위치가 비슷하다는 관측이며 강한 예측력이라는 뜻은 아니다. 단순 단변량 선형관계의 r²도 약0.09이고, 아래 HL 자체정보 대비100ms MAE 개선은 확인되지 않았다. Δ=1초에서는 전체 정점이 종료 시점0ms 또는 바로 뒤100ms다. Δ≥2초에서는 phase0 전체 정점이 대체로 종료 이전으로 옮겨간다. 30초·1분의 작은 상관 피크는 집계 구간 내부 사건과 시각 위상에 민감해 실제 수십 초의 전이 지연으로 읽을 수 없다.','',
      '+100ms부터 동시 하한이 연속 양수인 구간은500ms 신호에서 SKHX 최소1.4초, SKHY 최소1.8초까지다. 이는 단순 상관의 연속 구간으로, HL 자체 호가를 알 때 추가 MAE 개선이 남는 구간과 구분한다.','',
      '결합모형의 후보 L은 +100,+200,+300,+400,+500,+600,+700,+800,+900,+1000,+1500,+2000,+3000,+5000ms다. 표에서 `통과 없음`은 **어느 후보에서도 HL 자체정보 대비100ms 가격변화 MAE 개선의 동시 하한이 양수가 아니었다**는 뜻이다. 최고 점추정도 모두 음수였다. 100ms HL 가격변화에 0이 많은 만큼 이 MAE 기준은 작은 비영(非零) 예측에 불리할 수 있다. 긴 구간의 누적 가격 예측 성능은 이전 간격별 보고서와 별도로 읽어야 한다.','',
      f'![시간 간격별100ms 반응 지도]({OUT/"response_heatmap.png"})','',
      '## 결합의 추가가치와 실행 해석','',
      '`mid 대비 추가 bp`는 같은 날짜·같은100ms 미래 구간에서 mid 단독과 mid+micro 결합의 절대오차를 직접 뺀 값이다. 양수 점추정만으로 추가 정보가 강건하다고 판정하지 않는다. 동일 표본의 날짜 단위 bootstrap 동시 하한을 함께 사용한다.','',
      '고정된 Δ에서 HL이 움직인 정확한 단방향 통신 지연을 이 상관표로 식별할 수는 없다. BN 구간이 끝나기 전에 HL이 이미 반응한 사건도 신호에 섞이므로 Δ=30초나1분의 미래 피크가 늦게 보인다고 해당 계약의 물리적 전이가 그만큼 늦다고 해석하지 않는다. 기존 단일100ms BN 변화의 수신축 반응 정점 약500ms와 이 결과의 다른 조건을 구분한다.','',
      '실제 HL 주문 제출 ACK 중앙값440ms·취소 ACK750ms는 단방향 효력시각이 아니다. HL feed 호출240ms와 BN5ms도 과거 received_at 자료에 재가산하지 않았다. 시계 동기화, 개인 주문 활성·취소·체결 및 큐는 별도 측정이 필요하다. MAE 개선은 비용 후 거래 수익이 아니다.','',
      '## 결과·검증 파일','',
      '- `response_summary.csv`, `response_peaks.csv`:100ms 간격 반응과 날짜 bootstrap 하한.',
      '- `forecast_summary.csv`, `forecast_joint_lag_curve.csv`, `forecast_joint_peaks.csv`: HL 자체·mid 단독 대비 결합의 순차검증.',
      '- `response_daily.parquet`, `forecast_daily.parquet`, `coverage_daily.csv`, `manifest.json`: 날짜별 충분통계·유효 표본·정의.',
      '- `verification.json`: 원본·미래 정보·날짜·표본·계수/예측 재구성 검산.','']
    REPORT.write_text('\n'.join(lines))
    print('WROTE',REPORT,'rows',len(response),len(forecast),flush=True)


if __name__=='__main__':
    build()
