# 무기한선물 시장 미시구조 연구

SK하이닉스 현물과 Binance·Hyperliquid 무기한선물의 오더북, 공정가치, 교차시장 가격 신호와 마켓메이킹 전략을 분석한 프로젝트입니다.

## 주요 연구

- [네 퍼프의 공정가치 추정과 MM 호가](core_research/reports/REPORT_fv_mm_all_markets.md)
- [Binance–Hyperliquid 교차시장 신호 비교](core_research/reports/REPORT_bn_hl_transfer_ranking.md)
- [250ms 주기 교차시장 전이](core_research/reports/REPORT_bn_hl_250ms_transfer.md)
- [두 전이 모형의 maker 손익](core_research/reports/REPORT_bn_hl_pnl_two_models.md)
- [SK하이닉스 현물 공정가치](core_research/reports/REPORT_skh_spot_fv_09_20.md)
- [기존 maker 전략 판정](core_research/reports/REPORT_stage3_verdict.md) 및 [주문 지연 민감도](core_research/reports/REPORT_stage7.md)

## 폴더 구성

- [`core_research/reports/`](core_research/reports/) — 연구 보고서
- [`core_research/code/`](core_research/code/) — 분석·시뮬레이션·검증 코드
- [`core_research/README.md`](core_research/README.md) — 읽는 순서와 재현 시 주의사항

분석에 사용한 기간은 주로 2026-10-15~2026-11-01이며, 보고서의 성과는 역사적 시장 데이터 기반 모형 결과입니다. 실거래 성과나 미래 수익을 보장하지 않습니다. 원본 시세 데이터와 전체 코드 의존성은 이 저장소에 포함되어 있지 않으므로 그대로 실행 가능한 재현 패키지는 아닙니다.
