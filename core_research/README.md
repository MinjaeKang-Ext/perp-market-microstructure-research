# SK하이닉스 현물·퍼프 공정가치와 마켓메이킹 연구

2026년 10월 15일~11월 1일 자료를 중심으로 키움 현물, Binance 및 Hyperliquid 퍼프의 호가와 체결, 교차시장 신호, MM 재생 손익을 검토한 연구 묶음이다. 보고서는 `reports/`, 관련 분석 코드는 `code/`에 있다.

## 읽는 순서

1. `reports/REPORT_fv_mm_all_markets.md`: 네 계약의 단기 공정가치와 양면 호가 재생.
2. `reports/REPORT_bn_hl_transfer_ranking.md`, `REPORT_bn_hl_250ms_transfer.md`: Binance–Hyperliquid 선행·후행 신호와 시간 해상도 비교.
3. `reports/REPORT_bn_hl_pnl_two_models.md` 및 `_microprice.md`: 두 전이 모형의 maker 손익과 민감도.
4. `reports/REPORT_skh_spot_fv_09_20.md`: 현물 공정가치 연구.
5. `reports/REPORT_stage3_verdict.md`, `REPORT_stage7.md`: 기존 maker·taker 설계의 지연 민감도와 판정.

보고서 수치는 과거 공개 시장자료를 사용한 모형 실험이며 실계좌 체결이나 미래 수익을 보장하지 않는다. 각 보고서의 가정과 한계를 함께 읽어야 한다.

## 재현 시 주의

분석 코드는 원 프로젝트의 경로와 다른 스크립트, 시세 parquet 및 중간 산출물을 참조할 수 있다. 이 묶음에는 대용량 데이터와 전체 의존성이 포함되지 않으므로 그대로 실행 가능한 배포본이 아니다. 보고서의 로컬 절대경로·이미지 경로 또한 원 프로젝트를 가리킬 수 있다. 데이터를 포함하지 않았으므로 계정정보와 원시 시세 데이터는 이 묶음에 없다.
