# withEarth_V0 — Crypto Futures Auto Trader (Backend, 2025-08-18)

Flask 기반 백엔드 API. Binance USDⓈ-M Futures 데이터/주문과 Google Gemini 예측을 결합해 **신호 생성 → RR/ATR 게이팅 → (옵션) 주문 집행**을 수행합니다. React SPA는 별도 호스팅(옵션)입니다.

---

## 아키텍처 한눈에

1. **데이터 수집**: 5m 엔트리 + 1h/4h/1d MTF, 오더북 스프레드/불균형, 지표(RSI, SMA20, 변동성), ATR(14). *부분봉(미종가) 제거.* :contentReference[oaicite:28]{index=28}  
2. **LLM 예측**: Gemini(JSON 스키마 강제) → `{direction, prob, support, resistance, reasoning}`. 폴백·수선·디버그 덤프. :contentReference[oaicite:29]{index=29}  
3. **리스크 게이팅**: `MIN_PROB`·`RR_MIN`·스프레드·호라이즌·TP/SL 산출(ATR) → “enter/hold”. RULE_BACKUP(간단 규칙) 내장. BE 트레일링·타임배리어 유지보수. :contentReference[oaicite:30]{index=30}  
4. **집행**: LIMIT(재호가/TTL) → 실패 시 MARKET 폴백(옵션). 브래킷: TP(LIMIT|MARKET) + SL(STOP_MARKET). 틱/스텝/노셔널 정규화. :contentReference[oaicite:31]{index=31}  
5. **로깅**: 파일 `logs/bot.log`, 저널 `logs/trades.csv`, 페이로드 스냅샷 `logs/payloads/YYYYMMDD/*`. API로 조회 가능. :contentReference[oaicite:32]{index=32}

---

## 폴더 구조

- `app.py`: WSGI 엔트리(`app`/`server`), 라우트, 로깅 세팅. :contentReference[oaicite:33]{index=33}  
- `helpers/`
  - `binance_client.py`: Binance SDK 래퍼(주문/필터/포지션/브래킷/폴백). :contentReference[oaicite:34]{index=34}
  - `data_fetch.py`: OHLCV/오더북/지표/MTF/펀딩 수집(SDK 우선→REST). :contentReference[oaicite:35]{index=35}
  - `predictor.py`: Gemini 호출/스키마 강제/폴백/수선/`should_predict` 게이트. :contentReference[oaicite:36]{index=36}
  - `signals.py`: 신호 생성, RR/ATR 게이팅, LIMIT→MARKET 폴백, **BE 트레일링/타임배리어**. :contentReference[oaicite:37]{index=37}
  - `utils.py`: Secret Manager/GCS/파일 로깅 유틸. :contentReference[oaicite:38]{index=38}
  - `sentiment.py`(옵션): x.ai Grok 4 감성 점수. :contentReference[oaicite:39]{index=39}

---

## API 엔드포인트

- `GET  /health` — 헬스 체크  
- `GET|POST /tasks/trader?symbols=BTCUSDT,ETHUSDT&debug=1` — 신호 생성(+실행 여부는 `EXECUTE_TRADES`) :contentReference[oaicite:40]{index=40}  
- `GET  /api/overview` — 잔고/포지션 요약(계정/포지션 API 래핑)   
- `GET  /api/trades?limit=200` — 저널(`logs/trades.csv`) 요약 포함 :contentReference[oaicite:42]{index=42}  
- `GET  /api/logs?lines=200` — 파일 로그 tail :contentReference[oaicite:43]{index=43}  
- `GET  /api/signals` — 최근 결정(`*_decision.json`) 목록 :contentReference[oaicite:44]{index=44}  
- `GET  /api/signals/latest?symbol=BTCUSDT` — 즉시 신호(실행 없음) :contentReference[oaicite:45]{index=45}  
- `GET  /api/candles?symbol=BTCUSDT&tf=5m&limit=500` — 차트용 OHLCV :contentReference[oaicite:46]{index=46}  
- `GET  /api/orderbook?symbol=BTCUSDT&limit=10` — 오더북 스냅샷 :contentReference[oaicite:47]{index=47}  
- `GET  /api/open_orders[?symbol=BTCUSDT]` — 미체결 주문 목록 :contentReference[oaicite:48]{index=48}

---

## 환경 변수(핵심)

### 운영/로깅
- `SYMBOLS=BTCUSDT,ETHUSDT`, `EXECUTE_TRADES=true|false`, `LOG_LEVEL=INFO`, `TZ=Asia/Seoul` :contentReference[oaicite:49]{index=49}  
- `GCS_BUCKET`, `GCS_PREFIX` (스냅샷 업로드) :contentReference[oaicite:50]{index=50}

### 신호/게이팅
- `MIN_PROB=0.60`, `RR_MIN=1.20`, `PROB_RELAX_THRESHOLD=0.75`, `RR_MIN_HIGH_PROB=1.05`, `MAX_SPREAD_BPS=2.0~4.0` :contentReference[oaicite:51]{index=51}  
- `HORIZON_MIN=20~30`, `TIME_BARRIER_ENABLED=true` :contentReference[oaicite:52]{index=52}  
- **사전 호출 게이트**: `MIN_VOL_FRAC=0.0005` (변동성 미달 시 LLM 호출 건너뜀) :contentReference[oaicite:53]{index=53}

### 집행/체결
- `ENTRY_MODE=LIMIT|MARKET`, `LIMIT_TTL_SEC=15`, `LIMIT_POLL_SEC=1.0`, `LIMIT_MAX_REPRICES=3`, `LIMIT_MAX_SLIPPAGE_BPS=2~6` :contentReference[oaicite:54]{index=54}  
- `LIMIT_FAILOVER_TO_MARKET=true` (리밋 오류 시 MARKET 폴백) :contentReference[oaicite:55]{index=55}  
- `TP_ORDER_TYPE=LIMIT|MARKET` (브래킷 TP 타입) :contentReference[oaicite:56]{index=56}

### BE 트레일링/브래킷
- `BE_TRAILING_ENABLED=true`, `BE_TRIGGER_R_MULT=1.0`, `BE_OFFSET_TICKS=1` :contentReference[oaicite:57]{index=57}  
- `ATR_MULT_TP=1.8~2.0`, `ATR_MULT_SL=1.0` :contentReference[oaicite:58]{index=58}

### 사이징/계정
- `RISK_USDT=100`, `LEVERAGE=5`, `MARGIN_TYPE=ISOLATED`, `POSITION_MODE=ONEWAY` :contentReference[oaicite:59]{index=59}

### Binance/Gemini
- `BINANCE_API_KEY/SECRET`, `BINANCE_FUTURES_TESTNET`, `BINANCE_USE_TESTNET`, 타임아웃/재시도(`BINANCE_HTTP_*`) :contentReference[oaicite:60]{index=60}  
- `GOOGLE_API_KEY`, `GEMINI_MODEL=gemini-2.5-flash-lite`, `G_TEMPERATURE=0.0`, `G_MAX_TOKENS=512` :contentReference[oaicite:61]{index=61}

> **참고(펀딩 비율)**: `data_fetch.py`가 `funding_rate_pct`를 OHLCV에 부착하지만 현재 payload에는 포함하지 않습니다. `signals._build_payload()`에서 `extra.funding_rate_pct`로 전달하도록 확장하면 LLM의 편향 인식에 유리합니다. 

---

## 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GOOGLE_API_KEY=... BINANCE_API_KEY=... BINANCE_API_SECRET=...
python app.py
# http://localhost:8080/health
배포 (Google App Engine)
bash
복사
gcloud app deploy app.yaml cron.yaml
app.yaml — gunicorn 엔트리(app:server), 오토스케일링, VPC 커넥터 등 정의. 

cron.yaml — every 5 minutes로 /tasks/trader 트리거(5m 전략에 동기화).

예시 cron.yaml:

yaml
복사
cron:
- description: run trader every 5 minutes
  url: /tasks/trader
  schedule: every 5 minutes
  timezone: Asia/Seoul
  target: default
운영 런북(추천)
Dry-run 2~5일: EXECUTE_TRADES=false. 지표/스프레드/확률 분포 확인. 

테스트넷 실집행 2~3일: LIMIT→MARKET 폴백·재호가 동작 검증.

소액 실거래: RISK_USDT 낮춰 점증. HORIZON_MIN=20~30·TP_ORDER_TYPE=MARKET A/B. 

보정: 아래 절차로 확률 보정 학습/적용.

확률 보정(calibration) — 옵션 (권장)
데이터: /logs/payloads/*_decision.json + logs/trades.csv에서

레이블: “TP 먼저 도달 vs SL 먼저 도달(또는 1R 기준)”

학습: Platt/Isotonic → { "a":..., "b":... } 또는 piecewise JSON 저장(예: gs://{GCS_BUCKET}/{GCS_PREFIX}/calibration/prob_calibrator_v1.json). 

적용: signals.py에서 모델 prob 읽은 직후 보정값으로 치환 → MIN_PROB 게이트에 사용. (아래 패치 적용)

연결 패치(예시) — helpers/calibration.py가 아래 인터페이스를 제공한다고 가정:

python
복사
# helpers/calibration.py (예시)
import json, os
from typing import Optional

_CAL: Optional[dict] = None

def load_calibrator() -> Optional[dict]:
    global _CAL
    if _CAL is not None: return _CAL
    path = os.getenv("CALIBRATOR_PATH", "./calibrator.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _CAL = json.load(f)
    except Exception:
        _CAL = None
    return _CAL

def apply(p: float) -> float:
    cal = load_calibrator()
    if not cal: return float(p)
    # Platt: 1 / (1 + exp(-(a*p+b))) 형태라고 가정
    import math
    a = float(cal.get("a", 1.0)); b = float(cal.get("b", 0.0))
    return 1.0 / (1.0 + math.exp(-(a * float(p) + b)))
python
복사
# helpers/signals.py — generate_signal() 내 prob 읽은 직후
from .calibration import apply as calibrate_prob  # 상단 import
# ...
prob_raw = float(res.get("prob", 0.0))
prob = float(calibrate_prob(prob_raw))  # ← 보정 적용
보정은 월 1회 주기로 재학습 권장. Cloud Run Job 혹은 별도 엔드포인트에서 배치 실행 후 GCS에 결과 저장 → 런타임에서 로드. 

트러블슈팅
리밋 오류(Price not increased by tick size)
→ 현재 코드가 사이드별 틱 보정과 재호가를 수행하며, 오류 시 MARKET 폴백을 지원합니다. 필요 시 LIMIT_MAX_SLIPPAGE_BPS↑ 또는 TP_ORDER_TYPE=MARKET로 체결성 강화.

미체결 잔여 브래킷
→ 포지션 0인데 RO/TP/SL 남아있으면 정리 루틴이 취소 처리. 

LLM 호출 과다
→ MIN_VOL_FRAC 상향으로 저변동 구간 스킵. 

라이선스/주의
본 코드는 레퍼런스 구현입니다. 실제 운용 시 시장·유동성·레버리지 리스크에 유의하세요.

markdown
복사

---

### 보너스: “운영 중 무엇을 언제 바꿀까?” 간단 가이드
- **체결률이 낮다** → `LIMIT_MAX_SLIPPAGE_BPS`↑, `LIMIT_TTL_SEC`↑, `TP_ORDER_TYPE=MARKET` (체결가 비용↑).   
- **되돌림 손실 잦다** → `HORIZON_MIN`↓, `BE_TRAILING_ENABLED=true`, `BE_TRIGGER_R_MULT`↓. :contentReference[oaicite:73]{index=73}  
- **거래가 너무 적다** → `MIN_PROB`↓, `RR_MIN`↓, `MAX_SPREAD_BPS`↑(주의), `MIN_VOL_FRAC`↓.   
- **허위 신호 많다** → `MIN_PROB`↑, `RR_MIN`↑, `PROB_RELAX_THRESHOLD`↑, 펀딩/미시구조 특징 강화(추가 구현). 

---
