# 🚀 withEarth_V0 — Crypto Futures Auto Trader (Backend)

**Flask API × Binance USDⓈ-M × Gemini · React SPA 대시보드(옵션)**

다중 타임프레임(5m/1h/4h/1d) + 오더북/펀딩/간단 심리 → **구조화 예측(JSON)** → (옵션) **브래킷 주문**까지 자동.

---

## 📦 레포 구조 (권장)

repo-root/
├─ backend/                      # Flask API (GAE 배포 루트)
│  ├─ app.py
│  ├─ requirements.txt
│  ├─ app.yaml
│  ├─ cron.yaml                  # (선택) /tasks/trader 스케줄
│  └─ helpers/
│     ├─ __init__.py
│     ├─ binance_client.py
│     ├─ data_fetch.py
│     ├─ predictor.py
│     ├─ sentiment.py (옵션)
│     ├─ signals.py
│     └─ utils.py
└─ frontend/                     # (옵션) React SPA

---

## 🧭 백엔드 플로우

`/tasks/trader`
→ `signals.generate_signal(symbol)` (드라이런) 또는 `manage_trade(symbol)` (실행)
→ `data_fetch.fetch_data()`(OHLCV/오더북/지표)
→ **payload 생성**
→ `predictor.get_gemini_prediction()`(JSON)
→ ATR 기반 RR 체크/TP·SL 산출
→ (실행) `binance_client.*`로 주문

---

## 🔌 API

- `GET /health` — 상태
- `GET|POST /tasks/trader?symbols=BTCUSDT,ETHUSDT&debug=1` — 시그널/주문 트리거
- `GET /api/overview` — 잔고/포지션 요약
- `GET /api/trades?limit=200` — 체결 저널(CSV)
- `GET /api/logs?lines=200` — 파일 로그 tail
- `GET /api/signals` — 최근 의사결정 JSON 목록
- `GET /api/signals/latest?symbol=BTCUSDT` — 즉시 의사결정(실행 X)
- `GET /api/candles?symbol=BTCUSDT&tf=5m&limit=500` — 차트용 OHLCV
- `GET /api/orderbook?symbol=BTCUSDT&limit=10` — 오더북 Top-N

---

## ⚙️ 환경 변수

### 런타임/로깅
| 키 | 기본값 | 설명 |
|---|---|---|
| `PORT` | `8080` | 로컬 포트 |
| `TZ` | `Asia/Seoul` | 로그 타임존 |
| `LOG_DIR` | `./logs` (GAE `/tmp/trading_bot`) | 로그/페이로드/CSV |

### 시그널/리스크
| 키 | 기본값 |
|---|---|
| `SYMBOLS` | `BTCUSDT,ETHUSDT` |
| `EXECUTE_TRADES` | `false` |
| `MIN_PROB` | `0.60` |
| `RR_MIN` | `1.20` |
| `ATR_MULT_TP` | `1.8` |
| `ATR_MULT_SL` | `1.0` |
| `LEVERAGE` | `5` |
| `MARGIN_TYPE` | `ISOLATED` |
| `POSITION_MODE` | `ONEWAY` |
| `MAX_SPREAD_BPS` | `2.0` |
| `RISK_USDT` | `100` |

### Binance (USDⓈ-M Futures)
| 키 | 기본값 | 설명 |
|---|---|---|
| `BINANCE_FUTURES_TESTNET` | `false` | SDK 테스트넷 |
| `BINANCE_API_KEY/SECRET` |  | |
| `BINANCE_HTTP_TIMEOUT_MS` | `10000` | REST 타임아웃(ms) |
| `BINANCE_HTTP_RETRIES` | `3` | (옵션) 재시도 |
| `BINANCE_HTTP_BACKOFF_MS` | `1000` | (옵션) 백오프 |
| `BINANCE_USE_TESTNET` | `false` | REST 폴백 테스트넷 |
| `BINANCE_FAPI_BASE` | `https://fapi.binance.com` | REST 본번 |
| `BINANCE_FAPI_TESTNET_BASE` | `https://testnet.binancefuture.com` | REST 테스트넷 |

### Gemini (google-genai 1.28.0)
| 키 | 기본값 | 설명 |
|---|---|---|
| `GOOGLE_API_KEY` |  | 필수 |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | 모델명 |
| `G_TEMPERATURE` | `0.0` | |
| `G_MAX_TOKENS` | `512` | |

### 프론트 리다이렉트(옵션)
| 키 | 예시 |
|---|---|
| `FRONTEND_BASE_URL` | `https://spa.example.com` |

---

## 🪵 **Cloud Logging — 구조화 이벤트**

아래 이벤트가 **`INFO` 레벨 JSON**으로 출력됩니다(표준 출력으로 보내므로 GAE Cloud Logging에서 자동 수집).

- `event="gemini.request"`: `{symbol, model, payload_hint, payload_preview}`  
- `event="gemini.response"`: `{symbol, prob, direction, support?, resistance?, entry?}`  
- `event="signal.decision"`: `{symbol, direction, prob, entry, tp, sl, rr, risk_ok}`  
- `event="binance.order.request"`: `{symbol, side, type, price?, qty, reduce_only?, extras...}`  
- `event="binance.order.response"`: `{symbol, side, type, orderId?, status?, price?, qty?, raw...}`

> **대용량 payload 전문**은 파일로도 남깁니다:  
> `${LOG_DIR}/payloads/YYYYMMDD/{ts}_{symbol}_request.json` / `_decision.json`

**GAE에서 보기 예시**
```bash
# 서비스명(default) 최신 로그 tail
gcloud app logs tail -s default

# event별 필터 (예: gemini.response)
gcloud logging read 'resource.type="gae_app" jsonPayload.message:"gemini.response"' --limit=50 --freshness=1h
🧪 로컬 실행
bash
복사
편집
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# .env 준비 후
python app.py
# http://localhost:8080/health
GAE 배포

bash
복사
편집
cd backend
gcloud app deploy app.yaml cron.yaml
🧯 트러블슈팅
부분봉 혼입 방지: /fapi/v1/klines close_time 기준으로 미종가 봉 drop 처리. 

SDK/REST 폴백: SDK 실패 시 REST로 대체, 형식 통일. 

Gemini Part.from_text 오류: 본 백엔드는 dict+JSON 직렬로 contents를 구성(타입 안전). (변경 코드 반영됨) 

🧾 예시 .env
ini
복사
편집
PORT=8080
TZ=Asia/Seoul
LOG_DIR=./logs

SYMBOLS=BTCUSDT,ETHUSDT
EXECUTE_TRADES=false

# Binance
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_FUTURES_TESTNET=true
BINANCE_USE_TESTNET=true
BINANCE_FAPI_BASE=https://fapi.binance.com
BINANCE_FAPI_TESTNET_BASE=https://testnet.binancefuture.com
BINANCE_HTTP_TIMEOUT_MS=10000

# Gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite
G_TEMPERATURE=0.0
G_MAX_TOKENS=512

# Risk/Signals
MIN_PROB=0.60
RR_MIN=1.20
ATR_MULT_TP=1.8
ATR_MULT_SL=1.0
LEVERAGE=5
MARGIN_TYPE=ISOLATED
POSITION_MODE=ONEWAY
MAX_SPREAD_BPS=2.0
RISK_USDT=100

# Front redirect (옵션)
FRONTEND_BASE_URL=http://localhost:5173