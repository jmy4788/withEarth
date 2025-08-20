# withEarth_V0 — Crypto Futures Auto Trader (Backend, 2025-08-19, patched)

Flask 기반 백엔드 API. Binance USDⓈ-M Futures 데이터/주문과 Google Gemini 예측을 결합해 **신호 생성 → RR/ATR/스프레드/게이트 → (옵션) 주문 집행**을 수행합니다. React SPA는 별도 호스팅(옵션)입니다.

> **이번 패치(2025‑08‑19)**  
> - **Step 1 적용:** LLM이 반환한 `support/resistance`를 TP/SL 산출에 실제 반영.  
> - **Step 3(프리게이팅) 강화:** LLM 호출 전 **쿨다운/쇼크가드**(방향 비종속)로 조기 차단, 필요 시 **MTF 정합**을 규칙 기반 힌트 방향으로 선검사.  
> - **문서 업데이트:** ENV 최적화 가이드, 튜닝 리포트 스크립트(`scripts/report.py`).

---

## 아키텍처

1. **데이터 수집** — 5m 엔트리 + 1h/4h/1d MTF, 오더북 스프레드/불균형, 지표(RSI, SMA20, 변동성), ATR(14) 수집. *부분봉(미종가) 제거.*  
2. **LLM 예측(Gemini)** — JSON 스키마 강제 `{direction, prob, support, resistance, reasoning}`. 폴백/수선/디버그 덤프.  
3. **리스크·게이트** — `MIN_PROB`·RR(수수료 반영)·스프레드·MTF·쇼크·쿨다운·호라이즌 → “enter/hold”. RULE_BACKUP(간단 규칙) 내장. BE 트레일링·타임배리어 유지보수.  
4. **집행** — LIMIT(포스트‑온리 옵션/재호가/TTL) → (옵션) MARKET 폴백. 브래킷: TP(LIMIT|MARKET) + SL(STOP_MARKET 또는 STOP_LIMIT). 틱/스텝/노셔널 정규화.  
5. **로깅** — 파일 `logs/bot.log`, 저널 `logs/trades.csv`, 페이로드 스냅샷 `logs/payloads/YYYYMMDD/*`. API로 조회 가능.

---

## 폴더 구조

- `app.py` — WSGI 엔트리(`app`/`server`), 라우트, 로깅 세팅.  
- `helpers/`
  - `binance_client.py` — Binance SDK 래퍼(주문/필터/포지션/브래킷/폴백).  
  - `data_fetch.py` — OHLCV/오더북/지표/MTF/펀딩 수집(SDK 우선→REST).  
  - `predictor.py` — Gemini 호출/스키마 강제/폴백/수선/`should_predict` 게이트.  
  - `signals.py` — 신호 생성, RR(수수료 인지)·ATR·스프레드·MTF/쇼크/쿨다운 게이팅, LIMIT→MARKET 폴백, **BE 트레일링/타임배리어**.  
  - `utils.py` — Secret Manager/GCS/파일 로깅 유틸.  
- `scripts/report.py` — 실거래 로그 기반 **체결률/TP·SL 히트/순RR 분포** 및 **확률 보정용 데이터** 요약 스크립트.

---

## 주요 ENV (권장값 포함)

### 런타임/로깅
```env
TZ=Asia/Seoul
LOG_LEVEL=INFO
EXECUTE_TRADES=true
SYMBOLS=BTCUSDT,ETHUSDT
GCS_BUCKET=tothemoon-v2-logs-vaulted-scholar-466013-r5-seoul
GCS_PREFIX=trading_bot
```
### 모델/보정
```env
GEMINI_MODEL=gemini-2.5-flash-lite
USE_CALIBRATED_PROB=true
PROB_CALIBRATION_PATH=logs/calibration.json
CALIB_MIN_SAMPLES=150
CALIB_BINS=10
```
### 리스크/게이트
```env
MIN_PROB=0.62                 # 보정 후 기준치
PROB_RELAX_THRESHOLD=0.75
RR_MIN=1.20
RR_MIN_HIGH_PROB=1.05
RR_EVAL_WITH_FEES=true        # 수수료 인지형 RR 게이트
MAX_SPREAD_BPS=2.0
HORIZON_MIN=25
TIME_BARRIER_ENABLED=true
MIN_VOL_FRAC=0.0007
# 3중 게이트
MTF_ALIGN_ENABLED=true
MTF_RSI_LONG_MIN=48
MTF_RSI_SHORT_MAX=52
SHOCK_BPS=30
SHOCK_ATR_MULT=1.5
ENTRY_COOLDOWN_MIN=10
```
### 손절/익절
```env
ATR_MULT_SL=1.2
ATR_MULT_TP=2.0
TP_ORDER_TYPE=LIMIT          # 필요 시 MARKET로 A/B
SL_ORDER_TYPE=STOP_MARKET    # STOP_LIMIT 시험 시 STOP + SL_LIMIT_SLIPPAGE_BPS 사용
SL_LIMIT_SLIPPAGE_BPS=10
```
### 집행/주문
```env
ENTRY_MODE=LIMIT
ENTRY_POST_ONLY=true
LIMIT_TTL_SEC=15
LIMIT_POLL_SEC=1.0
LIMIT_MAX_REPRICES=3
LIMIT_MAX_SLIPPAGE_BPS=2.0   # POST_ONLY 분기에서는 사용되지 않음(주의)
LIMIT_FAILOVER_TO_MARKET=true
LIMIT_TTL_FALLBACK_TO_MARKET=true
```
### BE 트레일링
```env
BE_TRAILING_ENABLED=true
BE_TRIGGER_R_MULT=0.7
BE_OFFSET_TICKS=2
```
### 포지션/사이징
```env
RISK_USDT=100
LEVERAGE=5
MARGIN_TYPE=ISOLATED
POSITION_MODE=ONEWAY
VOL_SIZE_SCALING=true
VOL_SCALAR_MIN=0.5
VOL_SCALAR_MAX=1.25
```
### 네트워킹/수수료
```env
BINANCE_HTTP_TIMEOUT_MS=10000
BINANCE_HTTP_RETRIES=3
BINANCE_HTTP_BACKOFF_MS=1000
FEE_MAKER_BPS=2.0
FEE_TAKER_BPS=4.0
```

---

## 무엇이 바뀌었나 (개발자용)

### 1) LLM `support/resistance` 반영 (Step 1)
- `helpers/signals.py/generate_signal()`에서 모델 응답의 `support/resistance`를 5m SR(`recent_high_5m`/`recent_low_5m`)과 **결합**해 TP/SL 후보에 사용.  
- 로직:  
  - Long → `tp=max(resistance, recent_high_5m, entry + k_tp*ATR)`, `sl=min(support, recent_low_5m, entry - k_sl*ATR)`  
  - Short → `tp=min(support, recent_low_5m, entry - k_tp*ATR)`, `sl=max(resistance, recent_high_5m, entry + k_sl*ATR)`

### 2) 프리게이팅 강화 (Step 3)
- **LLM 호출 이전**에 다음 사유로 즉시 `hold` 반환:  
  - **쿨다운 활성** (`ENTRY_COOLDOWN_MIN` 잔여 시간 존재)  
  - **쇼크 캔들** (최근 5m 변화량이 `SHOCK_BPS` 또는 `SHOCK_ATR_MULT` 초과) — 방향 미지정 시에도 **보수적으로 차단**  
  - **MTF 정합 실패(선택)** — 규칙 기반 힌트 방향(`RULE_BACKUP`)이 존재하는 경우에 한해 선검사

### 3) 보고서 스크립트 추가 (Step 0)
- `scripts/report.py` — `logs/trades.csv`와 `logs/payloads/*_decision.json`을 스캔하여 체결률, 승률, R 분포, 확률→성과 상관을 요약.  
- (데이터가 충분하면) `logs/calibration.json`에 보정 곡선을 기록.

---

## 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=... BINANCE_API_KEY=... BINANCE_API_SECRET=...
python app.py
# http://localhost:8080/health
```

## GAE 배포

```bash
gcloud app deploy app.yaml cron.yaml
```

- `cron.yaml` 예시: 5분 주기
```yaml
cron:
- description: run trader every 5 minutes
  url: /tasks/trader
  schedule: every 5 minutes
  timezone: Asia/Seoul
  target: default
```

---

## 튜닝 가이드 (운영 중 빠른 레시피)

- **체결률이 낮다** → `LIMIT_TTL_SEC`↑(20~25), `LIMIT_MAX_REPRICES`=2~3, 필요 시 `TP_ORDER_TYPE=MARKET` (체결가 비용↑).  
- **되돌림 손실** → `HORIZON_MIN`↓(20), `BE_TRIGGER_R_MULT`↓(0.6~0.8), `ATR_MULT_SL`↑(1.3).  
- **거래가 너무 적다** → `MIN_PROB`↓(0.58~0.60), `RR_MIN`↓(1.10), `MIN_VOL_FRAC`↓(0.0005).  
- **허위 신호 많다** → `MIN_PROB`↑(0.65), `RR_MIN`↑(1.30), `PROB_RELAX_THRESHOLD`↑(0.80), 펀딩/미시구조 특징 가중치 강화.

---

## 주의
본 코드는 레퍼런스 구현입니다. 실제 운용 시 시장·유동성·레버리지 리스크에 유의하십시오.
