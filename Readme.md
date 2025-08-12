# 🚀 withEarth_V0 — Crypto Futures Auto Trader (Flask × Binance USDⓈ‑M × Gemini)

> 목표: 다중 타임프레임(5m/1h/4h/1d) + 오더북/펀딩/(옵션) 감성을 결합해 **신호 생성 → (옵션) 주문 실행**까지 한 번에.  
> Google App Engine(앱엔진)과 로컬 모두 지원.

---

## 📦 프로젝트 구조

```
├─ app.py                # Flask 엔드포인트(/health, /tasks/trader, /dashboard, /api/*) + Liquid Glass UI
├─ app.yaml              # App Engine 런타임/스케일링/환경변수
├─ cron.yaml             # App Engine 크론 배치(정기 실행)
├─ requirements.txt      # Python 의존성(권장 핀은 하단 참고)
└─ helpers/
   ├─ __init__.py
   ├─ binance_client.py  # USDⓈ-M 공식 '모듈형' SDK(1.0.0) 래퍼: 필터/수량/주문/계정 유틸
   ├─ data_fetch.py      # OHLCV/오더북/펀딩 수집(SDK 우선 → REST 폴백) + 지표 계산 + '미종가 봉 제거'
   ├─ predictor.py       # Google Gemini(google-genai 1.28+) 구조화 JSON 응답 강제 + 디버그 덤프
   ├─ sentiment.py       # (옵션) x.ai Grok4 기반 X 트윗 감성 점수
   ├─ signals.py         # 페이로드 생성 → 게이트(조건 부족시 hold) → Gemini 의사결정 → TP/SL 제안 → (옵션) 주문
   └─ utils.py           # 로깅/Secret Manager/GCS 유틸
```

---

## 🧭 전체 플로우

```
[Flask /tasks/trader]
  └─ for symbol in SYMBOLS
      └─ helpers/signals.generate_signal(symbol)
          ├─ helpers/data_fetch.fetch_data(...)
          │   ├─ SDK 실패 시 REST 폴백 (klines/depth/fundingRate)
          │   └─ RSI/SMA20/변동성/ATR/상대거래량/오더북 스프레드·불균형 등 계산
          ├─ (옵션) X 트윗 감성 점수 (sentiment)
          ├─ payload 요약본(payload_preview) 생성
          ├─ Google Gemini로 최종 방향/확률/근거 획득(스키마 강제)
          └─ ATR×계수 기반 TP/SL 제안 + RR 체크
      └─ (EXECUTE_TRADES=true) helpers/signals.manage_trade
          ├─ 포지션 모드/마진 타입/레버리지 설정
          ├─ 수량 최솟값·티크 정규화(거래필터)
          ├─ ENTRY_MODE=LIMIT일 때 TTL/재호가/슬리피지 예산 고려
          └─ 진입 → TP/SL(리듀스온리) 브래킷 오더
```

---

## 🔌 API 엔드포인트

### `GET /health`
- 단순 헬스체크 `{ "status": "ok" }`

### `GET|POST /tasks/trader`
- 쿼리/바디
  - `symbols`: 콤마구분 문자열 (예: `BTCUSDT,ETHUSDT`)
  - `debug=1`: 예외 발생 시 스택트레이스가 결과 JSON에 포함
- 응답 예
```json
{
  "status": "ok",
  "results": [
    {
      "symbol": "BTCUSDT",
      "result": {
        "direction": "long|short|hold",
        "entry": 120149.95,
        "tp": 121388.0,
        "sl": 119345.0,
        "prob": 0.70,
        "risk_ok": true,
        "rr": 1.53,
        "payload_preview": { "...": "요약 피처" }
      },
      "exec": { "...": "EXECUTE_TRADES=true 일 때만 포함" }
    }
  ]
}
```

### `GET /dashboard`
- Tailwind + Chart.js 기반 “Liquid Glass” 대시보드(서버 사이드 렌더링).
- 루트(`/`) 접근 시에도 자동으로 `/dashboard`로 리다이렉트.

### 대시보드용 데이터 API
- `GET /api/overview` : 계정 잔고/포지션 요약(거래소 API 기반)
- `GET /api/trades?limit=200` : `logs/trades.csv` 기반 체결 저널 + 실현/미실현/승률 집계
- `GET /api/logs?lines=200` : `logs/bot.log` 최근 n줄
- `GET /api/signals` : `logs/payloads/YYYYMMDD/*.json`에서 최신 신호 요약

> **주의**: 대시보드는 `logs/trades.csv`가 쌓인다는 전제입니다. 실제 주문이 체결되면 `helpers/signals.py::manage_trade()` 내부에서 CSV에 한 줄을 append하는 코드를 추가하세요(아래 “저널링” 참조).

---

## ⚙️ 환경 변수 (ENV)

### 서버/런타임
| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `PORT` | `8080` | 로컬 실행 포트 |
| `SYMBOLS` | `BTCUSDT,ETHUSDT` | `/tasks/trader` 기본 심볼 목록 |
| `EXECUTE_TRADES` | `false` | `true`면 주문 실행(브래킷 포함) |
| `TZ` | `UTC` | 로그 타임존 (`Asia/Seoul` 권장) |
| `LOG_DIR` | `./logs` (GAE는 `/tmp/trading_bot`) | 로그/페이로드/CSV 루트 |

### 리스크/시그널 파라미터 (`helpers/signals.py`)
| 키 | 기본값 | 의미 |
| --- | --- | --- |
| `MIN_PROB` | `0.60` | 모델 확률 임계(이상일 때만 진입 고려) |
| `RR_MIN` | `1.20` | 최소 손익비(미만이면 hold) |
| `ATR_MULT_TP` | `1.8` | ATR×계수로 TP 제안 |
| `ATR_MULT_SL` | `1.0` | ATR×계수로 SL 제안 |
| `LEVERAGE` | `5` | 레버리지 |
| `MARGIN_TYPE` | `ISOLATED` | 교차/격리 설정 |
| `POSITION_MODE` | `ONEWAY` | ONEWAY/HEDGE |
| `MAX_SPREAD_BPS` | `2.0` | 오더북 스프레드 상한(bps) |
| `ENTRY_MAX_RETRIES` | `2` | 진입 재시도 횟수 |
| `RISK_USDT` | `100` | 기준 명목(달러). 수량 산정에 사용 |
| `HORIZON_MIN` | `60` | 모델 의사결정 유효 시간(분) |
| `ENTRY_MODE` | `LIMIT` | `MARKET` 또는 `LIMIT` (LIMIT는 TTL/재호가/슬리피지 예산 사용) |
| `LIMIT_TTL_SEC` | `15` | 리밋 주문 1회 대기 시간(s) |
| `LIMIT_POLL_SEC` | `1.0` | 체결 폴링 주기(s) |
| `LIMIT_MAX_REPRICES` | `3` | 미체결 시 재호가 최대 횟수 |
| `LIMIT_MAX_SLIPPAGE_BPS` | `2.0` | 진입가 대비 허용 슬리피지(bps) |

### 바이낸스 (USDⓈ‑M Futures)
| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `BINANCE_FUTURES_TESTNET` | `false` | **모듈형 SDK** 테스트넷 스위치 |
| `BINANCE_HTTP_TIMEOUT_MS` | `10000` | REST 타임아웃(ms) |
| `BINANCE_HTTP_RETRIES` | `3` | 재시도 횟수 |
| `BINANCE_HTTP_BACKOFF_MS` | `1000` | 지수 백오프 기본(ms) |
| `BINANCE_API_KEY` | (없음) | API 키 |
| `BINANCE_API_SECRET` | (없음) | API 시크릿 |
| `BINANCE_USE_TESTNET` | `false` | **REST 폴백**에서 테스트넷 도메인 사용 여부 |
| `BINANCE_FAPI_BASE` | `https://fapi.binance.com` | REST 폴백 기본 베이스 |
| `BINANCE_FAPI_TESTNET_BASE` | `https://testnet.binancefuture.com` | REST 폴백 테스트넷 베이스 |

### Gemini (google‑genai 1.28+)
| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `GOOGLE_API_KEY` | (없음) | Gemini API 키 |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | 기본 모델명(환경값으로 교체 가능) |
| `G_TEMPERATURE` | `0.0` | 생성 온도 |
| `G_MAX_TOKENS` | `512` | 응답 토큰 상한 |

### 로깅/스토리지
| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `GCS_BUCKET` | (없음) | 설정 시, GCS에 트레이드/밸런스 CSV 스냅샷 업로드 |
| `GCS_PREFIX` | `trading_bot` | GCS 경로 프리픽스 |

---

## 🧩 데이터 수집/지표 (`helpers/data_fetch.py`)

- **SDK 우선, 실패 시 REST 폴백**: `klines`(OHLCV)·`depth`(오더북)·`fundingRate` 모두 폴백 경로 준비.  
  - REST 베이스는 `BINANCE_USE_TESTNET`/`BINANCE_FAPI_BASE`/`BINANCE_FAPI_TESTNET_BASE`로 제어.
- **미종가(진행중) 봉 제거**: 마지막 봉의 종가 시간이 미래이면 제거하여, 부분 캔들로 인한 왜곡/룩어헤드 방지.
- 컬럼 정규화: `timestamp, open, high, low, close, volume` 표준화 및 시간순 정렬.
- 파생 지표: `RSI(14), SMA_20, 변동성(std 20), ATR(14), 상대거래량` 등.
- 오더북 특징: 베스트 스프레드(bps), 상·하위 호가 누적 수량 불균형(imbalance).
- MTF 원천(1h/4h/1d)을 함께 로딩하여 5m 엔트리 문맥 강화.

> ⚠️ **테스트넷/프로덕션 전환**: SDK는 `BINANCE_FUTURES_TESTNET`, REST 폴백은 `BINANCE_USE_TESTNET` 및 베이스 URL ENV를 각각 사용합니다.

---

## 🧠 의사결정 (`helpers/predictor.py`)

- **구조화 JSON 강제**: `direction ∈ {long, short, hold}, prob ∈ [0,1], support/resistance, reasoning` 스키마를 **response_schema**로 강제.
- **프롬프트 최소화**: 제공된 **숫자 피처만** 사용하도록 설계(환각/과최적화 완화).
- **디버그 아티팩트**: 입력 페이로드/응답 후보/토큰 사용량을 `logs/payloads/YYYYMMDD/*.json`으로 덤프.
- **게이트(should_predict)**: 변동성/스프레드 등 조건이 부족하면 빠르게 `hold`로 결정(신호 과다 방지).

---

## 🎯 시그널 & 리스크 (`helpers/signals.py`)

1. **페이로드 구성**: 5m 최신 캔들 + MTF(1h/4h/1d) + 오더북/펀딩/감성 → `payload_preview` 축약본 산출
2. **게이트**: `should_predict()`에서 변동성/스프레드 등 최소 조건 미달이면 즉시 `hold`
3. **모델 호출**: Gemini로 `direction/prob/...` 결정 수신
4. **TP/SL 제안**: `entry ± ATR×{TP,SL} 계수` + 최근 고저(s/r) 고려
5. **검증**: `RR ≥ RR_MIN` 등 보수적 게이트
6. **주문 실행(옵션)**
   - 공통: 포지션 모드(ONEWAY/HEDGE), 마진 타입(ISOLATED/...) 설정
   - **MARKET**: 시장가 진입 후 즉시 **리듀스온리 TP/SL** 브래킷 오더
   - **LIMIT**(기본): 현재 호가와 `LIMIT_MAX_SLIPPAGE_BPS`를 반영해 리밋가 산출 → TTL 동안 체결 감시 → 미체결 시 재호가(최대 `LIMIT_MAX_REPRICES`) → 체결 시 브래킷 오더

> 기본값은 방어적입니다. RR·확률 임계·스프레드 상한이 보수적으로 설정되어 노이즈 구간에선 hold가 많습니다.

---

## 💱 주문/체결 (`helpers/binance_client.py`)

- **심볼 필터 로딩**: `tickSize/stepSize/minNotional` 캐시 → **가격/수량 정규화**.
- **주문**: 마켓/리밋 + 기본 브래킷(TP/SL reduce‑only) 생성 유틸 제공.
- **계정 설정**: 포지션 모드(ONEWAY/HEDGE), 마진 타입(ISOLATED), 레버리지 변경.
- **오버뷰**: 잔고/포지션 요약(대시보드 `/api/overview`) 제공.

---

## 📊 대시보드

- `/dashboard`: 실시간 로그/체결/PNL/신호 수 요약 **단일 페이지**.
- `/api/trades`: `logs/trades.csv` 읽어 테이블과 수익 곡선(Chart.js)로 시각화.
- `/api/overview`: 거래소 계정 상태(USDT 잔고 등) 표시.
- `/api/logs`: 최근 로그 tail.
- `/api/signals`: 최근 신호 스냅샷 수/목록.

### 체결 저널(필수 아답터)
- 주문을 실제로 넣는 경우, `helpers/signals.py::manage_trade()`에서 **체결 직후** 아래와 같은 형식으로 `logs/trades.csv`에 append 하세요(헤더는 자동 생성).
```csv
timestamp,symbol,side,qty,entry,tp,sl,exit,pnl,status,id
2025-08-12T00:00:00Z,BTCUSDT,long,0.01,100000,101800,99000,0,0,open,abc123
```

---

## 🧪 로컬 실행

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scriptsctivate
pip install -r requirements.txt
# .env에 키/시크릿/모델 등 설정
python app.py  # http://localhost:8080
```

## ☁️ GAE 배포(요약)

```bash
gcloud app deploy app.yaml cron.yaml
```
> 배포 전 `GOOGLE_API_KEY`, `BINANCE_API_KEY/SECRET`, `EXECUTE_TRADES` 등을 환경에 맞게 설정하세요(GCP Secret Manager 사용 권장).

---

## 🪵 로깅 & 관측성

- **파일 로그**: `logs/bot.log` (GAE는 `/tmp/trading_bot/bot.log`도 시도)
- **페이로드 덤프**: `logs/payloads/YYYYMMDD/*.json` (입력/후보/토큰 사용량)
- **GCS 업로드(옵션)**: `GCS_BUCKET` 설정 시 거래/밸런스 CSV 스냅샷을 **작은 파일**로 꾸준히 적재

---

## 🧯 트러블슈팅 체크리스트

- **DataFrame 진릿값 에러**: 조건문에서 `if df:` 대신 `if df is not None and not df.empty:` 패턴 유지
- **테스트넷 스위치**: SDK는 `BINANCE_FUTURES_TESTNET`, REST 폴백은 `BINANCE_USE_TESTNET`/베이스 URL 사용
- **과도한 HOLD**: `MAX_SPREAD_BPS`, `MIN_PROB`, `RR_MIN`, `ATR_MULT_*` 조정
- **Timeout/429**: REST 폴백의 타임아웃/재시도/백오프 파라미터 조정
- **/api/overview 직렬화**: 필요 시 DataFrame → `to_dict(orient="records")` 변환 후 JSON 응답

---

## 📚 버전/레퍼런스 (중요)

- **Binance USDⓈ‑M Futures 모듈형 SDK**: `binance-sdk-derivatives-trading-usds-futures==1.0.0`  
  ↳ 공식 저장소 패스: `binance-connector-python/clients/derivatives_trading_usds_futures`
- **Gemini**: `google-genai==1.28.0` — *반드시* [마이그레이션 가이드(ai.google.dev/gemini-api/docs/migrate?hl=ko)]를 준수해 사용

---

## 🧾 예시 requirements.txt (권장 핀)

```txt
# Core
flask>=3.0.0
gunicorn>=22.0.0
python-dotenv>=1.0.1

# Data
pandas>=2.2.2
numpy>=1.26.4
requests>=2.32.0

# Google Gemini (새 클라이언트)
google-genai==1.28.0

# Binance USDⓈ-M Futures (모듈형 SDK)
binance-sdk-derivatives-trading-usds-futures==1.0.0

# GCP (선택)
google-cloud-logging>=3.10.0
google-cloud-secret-manager>=2.20.0
google-cloud-storage>=2.16.0
```

---

## 🧾 예시 .env (샘플)

```env
# 기본
SYMBOLS=BTCUSDT,ETHUSDT
EXECUTE_TRADES=false
TZ=Asia/Seoul

# Binance
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_FUTURES_TESTNET=true
BINANCE_HTTP_TIMEOUT_MS=10000
BINANCE_HTTP_RETRIES=3
BINANCE_HTTP_BACKOFF_MS=1000

# REST 폴백
BINANCE_USE_TESTNET=true
BINANCE_FAPI_BASE=https://fapi.binance.com
BINANCE_FAPI_TESTNET_BASE=https://testnet.binancefuture.com

# Gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite
G_TEMPERATURE=0.0
G_MAX_TOKENS=512
HORIZON_MIN=60

# Signals / Risk
MIN_PROB=0.60
RR_MIN=1.20
ATR_MULT_TP=1.8
ATR_MULT_SL=1.0
LEVERAGE=5
MARGIN_TYPE=ISOLATED
POSITION_MODE=ONEWAY
MAX_SPREAD_BPS=2.0
ENTRY_MAX_RETRIES=2
RISK_USDT=100
ENTRY_MODE=LIMIT
LIMIT_TTL_SEC=15
LIMIT_POLL_SEC=1.0
LIMIT_MAX_REPRICES=3
LIMIT_MAX_SLIPPAGE_BPS=2.0

# GCS(옵션)
GCS_BUCKET=your-bucket
GCS_PREFIX=trading_bot
```

---

## 👋 메모 / 후속 아이디어

- `/tasks/trader` 응답에 백테스트용 간단 메타(최근 n개 캔들 상승/하락 카운트 등) 추가
- 스프레드/유동성 기반 **사이징**(risk_scalar→수량)에 반영
- Kraken/Bybit 등 거래소 래퍼 플러그인화
