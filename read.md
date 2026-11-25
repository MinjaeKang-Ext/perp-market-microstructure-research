# SK하이닉스 현물·무기한선물 시장 미시구조 연구

## 프로젝트 개요

이 프로젝트는 SK하이닉스 현물과 이를 기초로 하는 Binance·Hyperliquid 무기한선물의 호가·체결 데이터를 사용해 **단기 공정가치를 추정하고, 그 가격 신호가 실제 거래 가능한 마켓메이킹(MM) 성과로 이어지는지** 연구한다.

핵심 질문은 다음과 같다.

1. 현재 중간호가보다 오더북 잔량, 주문흐름, 다른 시장의 가격을 활용하면 미래 가격을 더 잘 예측할 수 있는가?
2. Binance·Hyperliquid 및 현물 간 가격 변화에 선행·후행 관계가 있는가? 그 효과가 주문 지연 이후에도 남는가?
3. 공정가치, 재고, 역선택 위험을 반영한 양면 지정가 호가가 수수료와 체결 조건을 고려한 뒤에도 수익을 내는가?
4. 주문·취소 지연과 대기열, 호가 신선도, 시장별 데이터 품질이 결과를 얼마나 바꾸는가?

## 연구 대상과 데이터

| 구분 | 심볼 | 연구상 의미 |
|---|---|---|
| 키움 현물 | `000660` | SK하이닉스 보통주 체결·호가 |
| Binance 보통주형 퍼프 | `SKHYNIXUSDT` | 한국 보통주 가격을 추종하는 퍼프 |
| Hyperliquid 보통주형 퍼프 | `xyz:SKHX` | 같은 보통주형 기초자산의 퍼프 |
| Binance ADR형 퍼프 | `SKHYUSDT` | SK하이닉스 ADR 가격을 추종하는 퍼프 |
| Hyperliquid ADR형 퍼프 | `xyz:SKHY` | ADR형 기초자산의 퍼프 |

자료는 BBO(최우선 매수·매도 호가), L2 오더북, 체결과 펀딩 이력이다. 주요 퍼프 실험 기간은 **2026-10-15~2026-11-01 KST의 18일**이다. 실험별 대상·기간·훈련과 평가 구분은 각 보고서에 명시되어 있으며 모두 같은 표본을 사용한 것은 아니다. 이후 다른 실험에서도 재사용한 기간이므로 보고서의 OOS는 완전히 새로운 미사용 검증 기간을 뜻하지 않는다.

키움 체결의 원시 거래소 구분은 KRX로 확인되었다. 반면 키움 L2 기본 잔량은 시장별 전용 FID와 대조해도 KRX 단독인지 통합인지 확정할 수 없었다. **NXT 전용 BBO·L2·체결 데이터는 연구 원본에서 확인되지 않았다.** 같은 종목의 보통주형과 ADR형은 서로 다른 가격 단위·거래시간을 가지므로 단순 동일 가격으로 취급하지 않는다.

## 연구 방법

### 1. 단기 공정가치

현재 중간호가 `mid = (best bid + best ask) / 2`를 기준선으로 둔다. 호가 잔량 불균형(OBI), 주문흐름 불균형(OFI), 스프레드, 직전 가격 변화 및 다른 거래소 신호로 미래 중간호가나 최우선 호가의 변화를 추정한다. 단순 잔량가중 중간가(microprice 계열)도 비교한다.

`공정가치 = 현재 관측 mid + 예측 가격 변화`로 구성하고, 실제 미래 가격과의 오차를 현재 mid를 그대로 유지하는 기준선과 비교한다. MAE는 일반적인 절대 오차, MSE는 큰 오차에 더 민감하다. 두 지표가 서로 다르게 개선될 수 있으므로 한쪽 결과만으로 예측력을 판단하지 않는다.

### 2. 교차시장 전이와 지연

Binance의 가격·오더북 변화가 이후 Hyperliquid 호가를 설명하는지, 반대로 어떤 구간에서 현물 정보가 퍼프 가격에 영향을 주는지 살펴본다. 250ms 격자, 500ms 및 1초 단위의 신호·미래 호가를 비교하고, 거래소 이벤트시각과 수집기 수신시각, 오래된 호가, 표본 시차 및 의사결정 후 지연을 구분한다.

통계적 선행 관계가 있다고 해서 신호를 받은 직후 주문을 넣고 취소할 수 있는 것은 아니다. 서로 다른 수집 지연, 주문 도달·취소 효력, 수수료와 스프레드를 반영해야 실제 활용 가능성을 검토할 수 있다.

### 3. 양면 MM과 손익 재생

예측 공정가치를 중심으로 매수·매도 지정가를 산출하고 재고가 쌓이면 기준가를 조정한다. 체결 후 불리한 가격 이동인 **역선택(toxicity)** 위험을 고려해 측별 호가 거리를 조절한다. 평가에는 공개 호가·체결 경로를 이용한 역사적 재생을 사용하고, 주문 지연, post-only 제약, 대기열 대용치, 수수료·펀딩·재고 평가를 모형화한다.

공개 시장자료에는 우리 주문의 실제 대기열 위치나 체결 여부가 없으므로, 결과는 모형에 따른 가상 손익이다. 호가를 제출하지 않는 선택과 가능한 모든 전략을 비교한 실험도 아니다.

## 주요 결과와 해석

### 네 퍼프의 공정가치와 MM

[네 퍼프 FV·양면 MM 보고서](core_research/reports/REPORT_fv_mm_all_markets.md)에서 Hyperliquid의 동시 교차시장 피처는 미래 mid 예측 오차를 줄였지만, 신호를 1초 늦춘 비교에서는 개선이 사라졌다. 해당 연구가 탐색한 **384개 MM 설정은 모두 순손익이 음수**였다.

이는 예측 오차가 줄어도 체결되는 순간에 불리한 정보가 집중되고, 대기열·지연·재고·비용이 수익을 잠식할 수 있음을 보여준다. 결과는 해당 보고서의 호가·체결 가정에서 성립하며, 모든 시장이나 모든 MM 전략이 불가능하다는 뜻은 아니다. 당시 폭 탐색은 후보 상한에 자주 걸렸고 미제출 후보는 포함하지 않았다.

### Binance → Hyperliquid 신호

[전이·공정가치 순위 보고서](core_research/reports/REPORT_bn_hl_transfer_ranking.md)와 [250ms 비교](core_research/reports/REPORT_bn_hl_250ms_transfer.md)는 양 거래소의 최우선 호가 반응과 시차를 비교한다. 정보가 빠르게 반영되는 구간이 관측되지만, 원래 의사결정 시점의 정보를 고정하고 **500ms 이후만** 보는 강화 검사에서는 48개 family/window 조합 중 양측·두 표본 위상의 신뢰구간 하한이 동시에 양수인 조합이 없었다.

따라서 초기의 상관관계나 MAE 감소를 주문 지연 이후에도 남는 거래 수익으로 바로 환산할 수 없다. 500ms 이후에 대한 이 결과는 해당 고정 모형의 검사이며, 그 지평에 맞게 새로 만든 모든 모형이 실패한다는 증명은 아니다.

### 현물 → 퍼프 공정가치

[현물 공정가치 보고서](core_research/reports/REPORT_skh_spot_fv_09_20.md)의 정규 호가 구간에서는 현물 micro 신호를 추가한 1초 공정가치 모형이 HL 자체 호가·OFI 기준선 대비 SKHX **13.92%**, SKHY **10.69%**의 MSE 감소를 보였다. Binance 두 계약의 감소는 0.06~0.08%였고 해당 동시 신뢰구간은 0을 포함했다.

같은 보고서의 별도 best bid/ask 직접 예측에서는 HL의 MSE가 개선돼도 MAE가 악화된 사례가 있었다. 따라서 **목표 가격·표본과 오차 지표를 구분**해야 하며, 현물 신호의 통계적 기여를 그대로 거래 수익으로 해석하지 않는다.

### 실행 지연과 maker 손익

[두 전이 모형의 maker 손익](core_research/reports/REPORT_bn_hl_pnl_two_models.md)은 HL 주문 제출 440ms·취소 750ms 효력 지연 가정 아래 1초와 500ms 모형을 비교했다. 두 계약 합산 전체 기간 기준 일평균 손익은 각각 **−$413.0**, **−$328.9**였다. 60초 가격 이동으로 측정한 역선택과 불리한 체결 가격이 주요 손실 요인이었다. [Microprice 변형](core_research/reports/REPORT_bn_hl_pnl_two_models_microprice.md)도 손실이었다.

[기존 maker 판정](core_research/reports/REPORT_stage3_verdict.md)은 약 434ms 지연 근사에서 개발 구간 45개 설정이 모두 손실이라고 보고했다. [별도 taker 실험](core_research/reports/REPORT_stage7.md)은 450ms를 의사결정→노드 수신 지연으로 놓고 추가 관측 대기를 반영했을 때 일평균 **−$57.59**였다. 450ms의 실제 측정 구간 정의에 따라 입력 방식이 달라지므로, 이 수치는 실행 모형의 조건과 함께 읽어야 한다.

## 이 저장소의 파일 위치

```text
read.md                         프로젝트 개요·방법·결과 해석
core_research/
├── README.md                   연구 묶음 안내 및 읽는 순서
├── reports/                    핵심 연구 보고서 9개
│   ├── REPORT_fv_mm_all_markets.md
│   ├── REPORT_fv_quote_policy.md
│   ├── REPORT_bn_hl_transfer_ranking.md
│   ├── REPORT_bn_hl_250ms_transfer.md
│   ├── REPORT_bn_hl_pnl_two_models.md
│   ├── REPORT_bn_hl_pnl_two_models_microprice.md
│   ├── REPORT_skh_spot_fv_09_20.md
│   ├── REPORT_stage3_verdict.md
│   └── REPORT_stage7.md
└── code/                       위 연구의 주요 분석·검증 코드
```

| 코드 | 목적 |
|---|---|
| [`validate_orderbook_fair_value.py`](core_research/code/validate_orderbook_fair_value.py), [`evaluate_orderbook_fair_value.py`](core_research/code/evaluate_orderbook_fair_value.py) | 미래 mid 예측용 입력·평가 |
| [`fv_mm_inputs.py`](core_research/code/fv_mm_inputs.py), [`fv_mm_forecasts.py`](core_research/code/fv_mm_forecasts.py) | MM 분석 입력과 공정가치 예측 |
| [`fv_mm_calibrate.py`](core_research/code/fv_mm_calibrate.py), [`fv_mm_replay.py`](core_research/code/fv_mm_replay.py) | 호가 폭 보정과 주문·체결 재생 |
| [`bn_hl_transfer_eval.py`](core_research/code/bn_hl_transfer_eval.py), [`bn_hl_transfer_response.py`](core_research/code/bn_hl_transfer_response.py) | 거래소 간 전이와 예측력 평가 |
| [`bn_hl_pnl_driver.py`](core_research/code/bn_hl_pnl_driver.py), [`bn_hl_pnl_replay.py`](core_research/code/bn_hl_pnl_replay.py) | 전이 모형의 maker 손익 재생 |
| [`skh_spot_fv_forecast.py`](core_research/code/skh_spot_fv_forecast.py), [`skh_spot_fv_transfer.py`](core_research/code/skh_spot_fv_transfer.py) | 현물 신호의 퍼프 공정가치 예측 |
| [`verify_*.py`](core_research/code/) | 각 분석의 독립 대조·검증 |

## 원본 프로젝트와 재현 범위

이 공개 저장소는 원본 프로젝트의 **핵심 보고서·코드 선별본**이다. 원본의 `perp_data/` 시세 parquet, `work/funding_momentum/` 펀딩 이력, `out_*` 중간·최종 산출물, 일부 코드 의존성은 여기에 포함되지 않는다. 일부 스크립트의 import와 경로는 원래 프로젝트 구조를 가정하고, 보고서에 남은 로컬 절대경로나 이미지 링크도 이 저장소에서 동작하지 않을 수 있다. 따라서 현재 상태는 연구 내용을 읽고 코드를 검토하기 위한 자료이며 독립 실행 가능한 재현 패키지는 아니다.

모든 결과는 역사적 데이터와 명시된 수수료·지연·체결 가정에 의존한다. 개인 실주문 기록으로 검증한 실거래 수익이나 미래 성과를 의미하지 않는다. 세부 숫자, 지표 정의, 표본과 한계는 해당 보고서를 기준으로 확인한다.
