# 🚀 withEarth\_V0 — Crypto Futures Auto Trader (Flask × Binance USDⓈ-M × Gemini)

> **목표**: 다중 타임프레임(5m/1h/4h/1d) 데이터와 오더북/펀딩/간단 심리를 결합해 **신호 생성 →(옵션) 주문 실행**까지 한 번에. GAE(앱엔진)와 로컬 모두 지원.

---

## 📦 프로젝트 구조

```
├─ app.py                # Flask 엔드포인트 (/health, /tasks/trader)
├─ app.yaml              # App Engine 런타임/스케일링/환경변수
├─ cron.yaml             # App Engine 크론 배치 (정기 실행)
├─ requirements.txt      # Python 의존성
└─ helpers/
   ├─ __init__.py
   ├─ binance_client.py  # Binance USDⓈ-M Futures SDK 래퍼 + 체결/필터 유틸
   ├─ data_fetch.py      # OHLCV/오더북/펀딩 수집 + REST 폴백 + 지표 계산
   ├─ predictor.py       # Google Gemini(gemini-1.5-*) JSON 구조 응답 강제
   ├─ sentiment.py       # (옵션) x.ai Grok4 기반 간단 X 트윗 감성 점수
   ├─ signals.py         # 페이로드 생성 → 모델 의사결정 → TP/SL 제안/검증 → (옵션) 주문
   └─ utils.py           # 로깅/비밀키/간단한 GCS 유틸
```

---

## 🧭 전체 플로우

```
[Flask /tasks/trader]
   └─ for symbol in SYMBOLS
       └─ helpers/signals.generate_signal(symbol)
           ├─ helpers/data_fetch.fetch_data(...)
           │   ├─ SDK 시도 실패 시 REST 폴백 (klines/depth/fundingRate)
           │   └─ RSI/SMA20/변동성/ATR/상대거래량/오더북 스프레드·불균형 등 계산
           ├─ (옵션) X 트윗 감성 점수 (sentiment)
           ├─ payload 요약본(payload_preview) 생성
           ├─ Google Gemini로 최종 방향/확률/해설 획득
           └─ ATR×계수 기반 TP/SL 제안 + RR 체크
       └─ (EXECUTE_TRADES=true) helpers/signals.manage_trade
           ├─ 포지션 모드/마진 타입/레버리지 설정
           ├─ 수량 최솟값·티크 정규화(거래필터)
           └─ 진입 → TP/SL(리듀스온리) 브래킷 오더
```

---

## 🔌 API 엔드포인트

### `GET /health`

* 단순 헬스체크. `{ "status": "ok" }`

### `GET|POST /tasks/trader`

* 쿼리/바디

  * `symbols`: 콤마구분 문자열 (예: `BTCUSDT,ETHUSDT`)
  * `debug=1`: 예외 발생 시, 스택트레이스가 결과 JSON에 포함
* 응답 예

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
        "payload_preview": {
          "pair": "BTCUSDT",
          "entry_5m": {"close": 120147.9, "rsi": 63.5, ...},
          "extra": {"RSI_1h": 57.5, "orderbook_imbalance": 0.55, ...}
        }
      },
      "exec": { ... }   // EXECUTE_TRADES=true 일 때만 포함
    }
  ]
}
```

* **주의**: `EXECUTE_TRADES=true` 일 때만 실주문 로직(`manage_trade`)이 실행됩니다. 기본은 dry-run.

---

## ⚙️ 환경 변수 (ENV)

> 자주 쓰는 것만 핵심 표로 정리. 필요 시 `app.yaml` 참고.

### 서버/런타임

| 키                |               기본값 | 설명                        |
| ---------------- | ----------------: | ------------------------- |
| `PORT`           |              8080 | 로컬 실행 포트                  |
| `SYMBOLS`        | `BTCUSDT,ETHUSDT` | /tasks/trader 기본 심볼 목록    |
| `EXECUTE_TRADES` |           `false` | `true`면 주문실행 활성화          |
| `TZ`             |             `UTC` | 로그 타임존 (`Asia/Seoul` 권장)  |
| `LOG_DIR`        |          `./logs` | `bot.log`/페이로드 JSON 저장 루트 |

### 리스크/시그널 파라미터 (`helpers/signals.py`)

| 키               |      기본값 | 의미                      |
| --------------- | -------: | ----------------------- |
| `MIN_PROB`      |     0.60 | 모델 확률 임계값(이상일 때만 진입 고려) |
| `RR_MIN`        |     1.20 | 최소 손익비(미만이면 무조건 홀드)     |
| `ATR_MULT_TP`   |      1.8 | ATR×계수로 TP 제안           |
| `ATR_MULT_SL`   |      1.0 | ATR×계수로 SL 제안           |
| `LEVERAGE`      |        5 | 기본 레버리지                 |
| `MARGIN_TYPE`   | ISOLATED | 교차/격리 설정                |
| `POSITION_MODE` |   ONEWAY | ONEWAY/HEDGE            |

### 바이낸스 (USDⓈ-M Futures)

| 키                         |     기본값 | 설명                        |
| ------------------------- | ------: | ------------------------- |
| `BINANCE_FUTURES_TESTNET` | `false` | `true`면 테스트넷 베이스URL 사용    |
| `BINANCE_HTTP_TIMEOUT_MS` |   10000 | REST 타임아웃(ms)             |
| `BINANCE_HTTP_RETRIES`    |       3 | 재시도 횟수                    |
| `BINANCE_HTTP_BACKOFF_MS` |    1000 | 지수 백오프 기본(ms)             |
| `BINANCE_API_KEY`         |    (없음) | API 키(환경변수 또는 GCP Secret) |
| `BINANCE_API_SECRET`      |    (없음) | API 시크릿                   |

### Gemini (google‑genai)

| 키                |                기본값 | 설명           |
| ---------------- | -----------------: | ------------ |
| `GOOGLE_API_KEY` |               (없음) | Gemini API 키 |
| `GEMINI_MODEL`   | `gemini-1.5-flash` | 모델 이름        |
| `G_TEMPERATURE`  |                0.0 | 생성 온도        |
| `G_MAX_TOKENS`   |                512 | 응답 토큰 상한     |

### 로깅/스토리지

| 키            |           기본값 | 설명                              |
| ------------ | ------------: | ------------------------------- |
| `GCS_BUCKET` |          (없음) | 설정 시, GCS에 트레이드/밸런스 CSV 스냅샷 업로드 |
| `GCS_PREFIX` | `trading_bot` | GCS 경로 프리픽스                     |

---

## 🧩 데이터 수집/지표 (helpers/data\_fetch.py)

* **SDK 우선, 실패 시 REST 폴백**: `klines`(OHLCV)·`depth`(오더북)·`fundingRate` 모두 폴백 경로 준비
* 컬럼 정규화: `timestamp, open, high, low, close, volume` 표준화 및 정렬
* 파생 지표: `RSI(14), SMA_20, returns/std(20) 기반 변동성, ATR(14), 상대거래량` 등
* 오더북 특징: 베스트 스프레드(bps), 상하위 10호가 누적 수량 불균형

### 페이로드에 들어가는 주요 피처(요약)

| 그룹              | 키                                                                                                          | 설명                         |
| --------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------- |
| entry\_5m       | `close, rsi, volatility, sma20, funding_rate_pct, sentiment, timestamp`                                    | 5분봉 기준 핵심 피처               |
| extra(5m)       | `ATR_5m, recent_high_5m, recent_low_5m, relative_volume_5m`                                                | 5분 ATR/직전 고저/상대거래량         |
| extra(1h/4h/1d) | `RSI_1h/4h/1d, ATR_1h/4h/1d, SMA20_1h/4h/1d, recent_high_x, recent_low_x, relative_volume_x, volatility_x` | 다중 타임프레임 피처                |
| 오더북             | `orderbook_imbalance, orderbook_spread`                                                                    | 상하위 10호가 물량 불균형, 스프레드(bps) |
| 보조              | `sr_levels.recent_high/low, trend_filter.daily_uptrend/strength`                                           | 서포트/레지스턴스·간단 추세필터          |

> ⚠️ **테스트넷/프로덕션 전환**: REST 폴백도 `BINANCE_USE_TESTNET/BINANCE_FUTURES_TESTNET` 환경에 맞춰 베이스URL을 고릅니다.

---

## 🧠 의사결정 (helpers/predictor.py)

* **JSON Schema 강제**: `direction ∈ {long, short, hold}`, `prob ∈ [0,1]`, `support/resistance(reasoning)` 포함하도록 강제
* **프롬프트 최소화**: 제공된 **숫자 피처만** 사용하게 설계 (과최적화/환각 감소 목적)
* **디버그 파일**: 입력 페이로드/응답 candidates/토큰 사용량을 `logs/payloads/YYYYMMDD/*.json`으로 보관
* **게이트**: 변동성 등 조건이 부족하면 `hold`로 빠르게 결정하게 유도(신호 과다 방지)

---

## 🎯 시그널 & 리스크 (helpers/signals.py)

1. **페이로드 구성**: 5m 최신 캔들 + MTF(1h/4h/1d) + 오더북/펀딩/감성 → `payload_preview` 축약본 산출
2. **모델 호출**: Gemini로 `direction/prob/...` 결정 수신
3. **TP/SL 제안**: `entry ± ATR×{TP,SL} 계수` + 최근 고저(s/r) 고려
4. **검증**: `prob ≥ MIN_PROB` AND `RR ≥ RR_MIN` AND `스프레드 ≤ MAX_SPREAD_BPS` 등
5. **주문 실행(옵션)**: 설정된 포지션 모드/마진 타입/레버리지로 **진입 + 리듀스온리 TP/SL** 브래킷 세팅

> **건강한 방어적 기본값**: RR, 확률 임계, 스프레드 상한 등은 보수적으로 설정되어 있어, 노이즈구간에서는 `hold`가 많이 나옵니다.

---

## 💱 주문/체결 (helpers/binance\_client.py)

* **심볼 필터 로딩**: `tickSize/stepSize/minNotional` 캐시 → 가격/수량 정규화
* **주문 종류**: 마켓/리밋 + 기본 브래킷(TP/SL reduce-only)
* **계정 설정**: 포지션 모드(ONEWAY/HEDGE), 마진 타입(ISOLATED), 레버리지 변경
* **오버뷰**: 잔고/포지션 요약을 DataFrame으로 제공 (대시보드/로깅용)

---

## 📝 크론/배포 (GAE)

* `app.yaml`: Python 3.12, gunicorn 엔트리포인트, 스케일링, 표준 ENV 포함
* `cron.yaml`: `Asia/Seoul` 타임존 기준, 60분 간격으로 `/tasks/trader` 호출

### 로컬 실행

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# .env에 키/시크릿/모델 등 설정
python app.py  # http://localhost:8080
```

### GAE 배포(요약)

```bash
gcloud app deploy app.yaml cron.yaml
```

> 배포 전 `GOOGLE_API_KEY`, `BINANCE_API_KEY/SECRET`, `EXECUTE_TRADES` 등을 환경에 맞게 설정하세요. (GCP Secret Manager 사용 권장)

---

## 🪵 로깅 & 관측성

* **파일 로그**: `logs/bot.log` (로컬) / 서버리스는 `/tmp/trading_bot/bot.log` 시도
* **페이로드 덤프**: `logs/payloads/YYYYMMDD/*.json` (입력/후보/토큰사용량)
* **GCS 업로드(옵션)**: `GCS_BUCKET` 설정 시 거래/밸런스 CSV 스냅샷을 **작은 파일**로 꾸준히 적재

---

## 🧪 테스트 팁

* **드라이런 확인**: `EXECUTE_TRADES=false` 상태에서 `/tasks/trader?symbols=BTCUSDT,ETHUSDT` 반복 호출
* **디버그**: `?debug=1`을 붙이면 예외 발생 시 trace 포함(개발 중에만)
* **데이터 품질**: `payload_preview`의 `entry_5m.close/RSI/ATR_5m` 등이 비정상(0.0)일 경우 → SDK 실패/REST 폴백 실패 여부를 로그에서 확인

---

## 🧯 트러블슈팅 체크리스트

* **DataFrame 진릿값 에러**: 조건문에서 `if df:` 대신 `if df is not None and not df.empty:` 패턴 유지
* **테스트넷**: `BINANCE_FUTURES_TESTNET=true` + API 키도 테스트넷용인지 확인
* **과도한 HOLD**: `MAX_SPREAD_BPS`, `MIN_PROB`, `RR_MIN`, `ATR_MULT_*` 조정. 변동성이 낮은 장에서는 의도적으로 보수적임
* **Timeout/429**: REST 폴백의 타임아웃/재시도/백오프 파라미터 조정

---

## 📚 버전/레퍼런스

* **Binance USDⓈ-M Futures 공식 모듈형 SDK**: `binance-sdk-derivatives-trading-usds-futures==1.0.0`
* **Gemini**: `google-genai>=1.28.0` (공식 문서의 마이그레이션 가이드에 맞춰 사용)

---

## ✅ 유지보수 메모

* 새 SDK 소버전에서 메서드명이 살짝 달라질 수 있어, 내부 래퍼가 **복수 후보명**을 자동 시도합니다.
* 페이로드·응답 덤프는 회귀 테스트에 유용하니, 문제 발생 시 `logs/payloads` 먼저 확인하세요.

---

## 🧾 예시 .env (샘플)

```dotenv
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

# Gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
G_TEMPERATURE=0.0
G_MAX_TOKENS=512

# Signals / Risk
MIN_PROB=0.60
RR_MIN=1.20
ATR_MULT_TP=1.8
ATR_MULT_SL=1.0
LEVERAGE=5
MARGIN_TYPE=ISOLATED
POSITION_MODE=ONEWAY

# GCS(옵션)
GCS_BUCKET=your-bucket
GCS_PREFIX=trading_bot
```

---

## 👋 문의/후속작업 아이디어

* `/tasks/trader` 응답에 백테스트용 간단 메타(최근 n개 캔들 상승/하락 카운트 등) 추가
* 스프레드/유동성 기반 **사이징**(risk\_scalar→수량)에 반영
* Kraken/Bybit 등 거래소 래퍼 플러그인화
