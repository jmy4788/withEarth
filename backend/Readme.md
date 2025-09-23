withEarth_V0 백엔드는 Binance USDS‑M 선물 거래를 자동화하는 Flask 기반 서비스입니다. 모델 예측(Gemini/TSFM), 리스크 게이트(RR/EV/ATR/스프레드/쿨다운/MTF/쇼크), HEDGE‑safe 주문, Kill Switch(회로차단), 저널링(GCS 백업 포함)을 제공합니다.

**주요 변경점(2025-09) 요약**
- Hedge 모드 안전화: LONG/SHORT 자동 지정(BOTH 방지), 주문 멱등키(`newClientOrderId`) 자동 주입, 시그니처 정규화+소폭 재시도.
- Kill Switch: 일중 손실/연속 손실/누적 MDD 기준 자동 중지.
- 정산 Plan B: 주문 히스토리 실패 시 사용자 체결 이력(user trades)으로 폴백하여 청산 탐색 안정화.

**구성**
- 백엔드: Flask + Gunicorn (`backend/app.py`, WSGI `server`)
- 헬퍼: 거래소/시그널/예측/유틸 (`backend/helpers/*`)
- 도구: 저널 분석·교정 (`backend/tools/*`)
- 프런트엔드(선택): `frontend/`

**디렉터리**
- `backend/app.py`: 엔드포인트, 작업 크론, 로깅/메트릭, 캘리브레이션
- `backend/helpers/binance_client.py`: 거래소 연동(주문/정산/포지션/필터)
- `backend/helpers/signals.py`: 시그널 생성, 리스크 게이트, 주문/브래킷, 저널
- `backend/helpers/predictor*.py`: 예측 백엔드(Gemini/TSFM)
- `backend/helpers/utils.py`: Secret/GCS/로깅/이벤트
- `backend/tools/*`: `calibrate_from_trades.py`, 분석 스크립트 등

**빠른 실행(로컬)**
- 필수 키: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `GOOGLE_API_KEY`
- 권장: `EXECUTE_TRADES=false`로 시작해 시뮬레이션 확인
- 명령:
  - 가상환경: `python -m venv .venv` 후 활성화
  - 의존성 설치: `pip install -r backend/requirements.txt`
  - 환경변수 설정(예): `GOOGLE_API_KEY=...`, `BINANCE_API_KEY=...`, `BINANCE_API_SECRET=...`
  - 서버 실행: `python backend/app.py` → 브라우저에서 `http://localhost:8080/health`

**배포(Google App Engine 표준)**
- 작업 디렉터리: `backend/`
- 명령:
  - 배포: `gcloud app deploy app.yaml cron.yaml`
- 스케줄: `backend/cron.yaml`에서 `/tasks/trader`, `/tasks/maintain`, `/tasks/journal_sync` 등 호출 주기 설정

**주요 엔드포인트**
- 헬스체크: `/health`
- 트레이더 크론: `/tasks/trader`
- 유지/정리: `/tasks/maintain`, 저널: `/tasks/journal_sync`, `/tasks/journal_reset`
- 개요/진단: `/api/overview`, `/api/trades`, `/api/orders/history`, `/api/logs`, `/api/metrics*`

**핵심 기능**
- Hedge‑safe 주문: HEDGE 모드에서 진입은 BUY→LONG, SELL→SHORT 자동 지정, 청산(RO)은 반대 사이드 자동 지정. `_safe_new_order`가 `newClientOrderId` 주입, snake/camel 정규화, 짧은 재시도 적용.
- 리스크 Kill Switch: `MAX_DAILY_LOSS_USD`, `MAX_CONSEC_LOSSES`, `MAX_MDD_USD` 기준 초과 시 `manage_trade`/`maintain_positions`에서 즉시 중지.
- 정산 Plan B(체결 폴백): 주문 히스토리로 TP/SL Filled 탐색 실패 시, 사용자 체결 이력에서 실현손익이 0이 아닌 최근 체결을 청산으로 간주.
- EV/RR/ATR/스프레드/쿨다운/MTF/쇼크 게이트와 브래킷(TP/SL) 자동 관리.
- 저널링: `logs/trades.csv`에 거래 기록, GCS 최신/일별 스냅샷 자동 백업.

**환경변수(핵심만)**
- 실행/일반: `EXECUTE_TRADES`, `SYMBOLS`, `TZ`, `LOG_DIR`, `LOG_LEVEL`
- 거래소/주문: `POSITION_MODE`(HEDGE 권장), `TP_ORDER_TYPE`, `SL_ORDER_TYPE`, `ENTRY_MODE`, `ENTRY_POST_ONLY`, `LIMIT_TTL_SEC`, `LIMIT_MAX_REPRICES`, `LIMIT_TTL_FALLBACK_TO_MARKET`, `FEE_MAKER_BPS`, `FEE_TAKER_BPS`, `MIN_TP_BPS_NET`
- 리스크/게이트: `MIN_PROB`, `RR_MIN`, `MAX_SPREAD_BPS`, `HORIZON_MIN`, `TIME_BARRIER_ENABLED`, `MTF_ALIGN_ENABLED`, `SHOCK_BPS`, `SHOCK_ATR_MULT`, `ENTRY_COOLDOWN_MIN`, `MAX_DAILY_LOSS_USD`, `MAX_CONSEC_LOSSES`, `MAX_MDD_USD`
- 예측/캘리브레이션: `GEMINI_MODEL`, `GOOGLE_API_KEY`, `USE_CALIBRATED_PROB`, `CALIB_MIN_SAMPLES`, `CALIB_BINS`, `PROB_CALIBRATION_PATH`
- GCS 백업: `GCS_BUCKET`, `GCS_PREFIX`, `JOURNAL_SYNC_ON_START`

**Hedge 모드 동작 요약**
- Before: `POSITION_SIDE=BOTH`가 주문에 실려 HEDGE 환경에서 거래소 거절/오작동 가능.
- After: 진입/청산 맥락에 따라 `position_side`를 LONG/SHORT 자동 지정(ONEWAY는 BOTH 유지), `newClientOrderId`로 중복 방지.

**저널(trades.csv)**
- 위치: `LOG_DIR/trades.csv`
- 주요 열: `timestamp,symbol,side,qty,entry,tp,sl,exit,pnl,status,id,...`
- 백업: 최신/일별 GCS 스냅샷 자동 업로드(복구/트림 기능 포함).

**로컬 테스트 팁**
- `EXECUTE_TRADES=false`로 API/게이트/시그널 흐름 점검 후 `true` 전환.
- Kill Switch 값을 작은 숫자로 두고 동작 확인 후 운영치로 상향.
- `POSITION_MODE=HEDGE` 설정 후 유지 태스크에서 포지션 모드/레버리지/마진 타입이 베스트에포트로 맞춰지는지 로그 확인.

**주의/면책**
- 본 코드는 투자 조언이 아니며, 실거래 책임은 사용자에게 있습니다. 실서버 투입 전 테스트·리스크 한도 설정을 반드시 진행하세요.

