backend/README.md
# 🚀 withEarth_V0 — Crypto Futures Auto Trader (Backend)

**Flask API × Binance USDⓈ-M × Gemini · React “Liquid Glass” Dashboard (Option B)**

> 다중 타임프레임(5m/1h/4h/1d), 오더북/펀딩/간단 심리를 결합해 신호 생성 → (옵션) 주문 실행.  
> 프론트는 **React SPA**(별도 호스팅), 백엔드는 **Flask API 전담**.

---

## 📦 프로젝트 구조 (권장)



repo-root/
├─ backend/ # Flask API (GAE 배포 루트)
│ ├─ app.py
│ ├─ requirements.txt
│ ├─ app.yaml
│ ├─ cron.yaml # (선택) /tasks/trader 정기 실행
│ └─ helpers/
│ ├─ init.py
│ ├─ binance_client.py
│ ├─ data_fetch.py
│ ├─ predictor.py
│ ├─ sentiment.py # (옵션) X 감성
│ ├─ signals.py
│ └─ utils.py
└─ frontend/ # React SPA (Vite + Tailwind + shadcn/ui)
└─ src/App.tsx


---

## 🧭 플로우



/tasks/trader
└─ signals.generate_signal(symbol)
├─ data_fetch.fetch_data() # SDK→REST, OHLCV + 지표 + 오더북/펀딩
├─ payload_preview 생성
├─ predictor.get_gemini_prediction() # 구조화 JSON (direction/prob/…)
├─ ATR 기반 TP/SL 제안 + RR 검증
└─ (EXECUTE_TRADES=true) manage_trade() # 포지션/마진/레버리지 + 브래킷 주문


---

## 🔌 API 엔드포인트

- `GET /health` — 상태 확인  
- `GET|POST /tasks/trader?symbols=BTCUSDT,ETHUSDT&debug=1` — 신호/주문 실행 트리거  
- `GET /api/overview` — 잔고/포지션 요약  
- `GET /api/trades?limit=200` — 체결 저널(로컬 CSV 기준)  
- `GET /api/logs?lines=200` — 최근 로그 tail  
- `GET /api/signals` — 최근 모델 의사결정 디버그 목록  
- `GET /api/signals/latest?symbol=BTCUSDT` — 해당 심볼 즉시 의사결정(실행 X)  
- `GET /api/candles?symbol=BTCUSDT&tf=5m&limit=500` — 차트용 OHLCV  
- `GET /api/orderbook?symbol=BTCUSDT&limit=10` — 오더북 Top-N

> Option B: `FRONTEND_BASE_URL` 설정 시 `/`와 `/dashboard`는 SPA로 **리다이렉트**.

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
| `BINANCE_FUTURES_TESTNET` | `false` | (SDK) 테스트넷 |
| `BINANCE_API_KEY/SECRET` |  | 키/시크릿 |
| `BINANCE_HTTP_TIMEOUT_MS` | `10000` | REST 타임아웃 |
| `BINANCE_HTTP_RETRIES` | `3` | (옵션) 재시도 |
| `BINANCE_HTTP_BACKOFF_MS` | `1000` | (옵션) 백오프 |
| `BINANCE_USE_TESTNET` | `false` | REST 폴백 테스트넷 |
| `BINANCE_FAPI_BASE` | `https://fapi.binance.com` | REST 프로덕션 |
| `BINANCE_FAPI_TESTNET_BASE` | `https://testnet.binancefuture.com` | REST 테스트넷 |

### Gemini (google-genai 1.28.0)
| 키 | 기본값 | 설명 |
|---|---|---|
| `GOOGLE_API_KEY` |  | 필수 |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | 모델명 |
| `G_TEMPERATURE` | `0.0` | 생성 온도 |
| `G_MAX_TOKENS` | `512` | 응답 토큰 상한 |

### 프론트 리다이렉트
| 키 | 예시 |
|---|---|
| `FRONTEND_BASE_URL` | `https://spa.example.com` |

---

## 🧪 로컬 실행

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# .env 생성 후
python app.py
# http://localhost:8080/health

☁️ GAE 배포
cd backend
gcloud app deploy app.yaml cron.yaml


Python 3.11(2세대) → 정적 핸들러 없음, SPA는 별도 호스팅(Cloud Storage/Cloud CDN 권장)

FRONTEND_BASE_URL 설정 시 / 및 /dashboard가 SPA로 리다이렉트

🪵 저널/로그

파일 로그: ${LOG_DIR}/bot.log

체결 저널: ${LOG_DIR}/trades.csv (체결 시 append)

페이로드/의사결정 덤프: ${LOG_DIR}/payloads/YYYYMMDD/*.json

(옵션) GCS_BUCKET 설정 시 CSV 스냅샷 업로드

🧯 트러블슈팅

DataFrame 진릿값: if df is None or df.empty: 패턴 사용

Gemini Part.from_text 에러: types.Part 제거, dict-contents 사용

/api/candles 직렬화: {t,o,h,l,c,v} 배열 출력

테스트넷/프로덕 전환: SDK/REST 각각 환경변수로 제어

🧾 예시 .env
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

# Front redirect (Option B)
FRONTEND_BASE_URL=http://localhost:5173

# GCS (optional)
GCS_BUCKET=your-bucket
GCS_PREFIX=trading_bot


---

# 빠른 실행 체크리스트 (로컬)

1. `backend/.env` 작성(Google API Key, Binance Key)  
2. `pip install -r backend/requirements.txt`  
3. `python backend/app.py` → `GET /health` 200 확인  
4. `GET /tasks/trader?symbols=BTCUSDT,ETHUSDT&debug=1`로 드라이런  
5. `GET /api/signals`에서 심볼별 의사결정 JSON 생성 여부 확인(오늘 날짜 디렉터리) :contentReference[oaicite:10]{index=10}  

필요하면 프론트 SPA 시드도 바로 만들어 드릴게요. 우선은 **백엔드 파일 완전체**를 우선순위로 정리해 드렸습니다.
