# withEarth(Backend) — Crypto Futures Quant Backend (proposed README)

> **Status**: Draft • 2025-09-24  
> **Scope**: Back-end (Flask/Gunicorn on App Engine) + TSFM 2.x microservice (FastAPI on Cloud Run)  

This backend automates **Binance USDⓈ-M Futures** trading and risk control. It integrates
a **probabilistic signal engine (TSFM 2.x)**, **Gemini (google-genai)** structured output,
**journal & calibration**, and **kill switch** orchestration. The goal is to reach stable,
scaled monthly PnL while keeping tail risk bounded.

---

## ✨ Highlights

- **Predictor router**: TSFM2 (TimesFM-based), Gemini(JSON Mode), Chronos (local) routing
- **Risk-first execution**: Hedge-safe order wiring (no `BOTH`), idempotent `newClientOrderId`
- **Kill Switch**: Daily loss, max consecutive losses, and MDD tripwires
- **Journal**: Local CSV + **GCS checkpointing** & restore
- **Calibration**: Reliability curve from realized trades → calibrated `prob`
- **Ops**: Cron-driven trader/maintainer tasks; lightweight metrics & logs

> **Patch note**: Recent patch moves **base TF to 15m** for TSFM2 and raises **Cloud Run memory**
for the TSFM microservice. Make sure the **Dockerfile/env** and **schedules** are consistent
(see _Deployment_).

---

## 🏗️ Architecture

```
App Engine (Flask/Gunicorn)                            Cloud Run (FastAPI)
┌────────────────────────────┐                         ┌──────────────────────┐
│ /tasks/trader              │  payload (features)     │  /v1/prob_gate       │
│ /tasks/maintain            ├────────────────────────▶│  /v1/forecast        │
│ /tasks/calibrate           │                         │  /health             │
│ /tasks/journal_sync        │                         └──────────────────────┘
│ /api/* (overview, logs…)   │
└────────────────────────────┘
   └─ helpers/
      ├─ data_fetch.py (OHLCV/OB/indicators)
      ├─ signals.py (sizing+RR+kill switch glue)
      ├─ predictor_router.py (TSFM2 / Gemini / Chronos)
      ├─ binance_client.py (orders/positions/filters)
      ├─ tsfm_predictor.py, tsfm_remote_client.py
      └─ utils.py (GCS, secrets, logging)
```

**Journaling**: `LOG_DIR/trades.csv` (+ GCS snapshots).  
**Calibration**: `tools/retrain_calibration_from_journal.py` → `calibration_tsfm2.json`.

---

## 📦 Repository Map (key files)

- `backend/app.py` – endpoints, cron tasks, metrics, calibration hooks
- `backend/helpers/*` – exchange/predictor/signal utilities
- `backend/cron.yaml` – App Engine Cron (trader/maintainer/calibrator/journal)
- `TSFM_Cloud/` – TimesFM 2.x microservice (FastAPI + Uvicorn)
- `backend/requirements.txt` – pinned: `binance-sdk-derivatives-trading-usds-futures==1.0.0`, `google-genai==1.28.0`

---

## 🔌 Dependencies

- **Exchange**: `binance-sdk-derivatives-trading-usds-futures==1.0.0` (official modular SDK)  
  See Binance USDⓈ-M **New Order** params (`positionSide`, `newClientOrderId`, etc.).
- **LLM**: `google-genai==1.28.0` (new Gemini SDK) — use `from google import genai; client = genai.Client()`

> Docs: Google GenAI SDK migration & usage; Binance USDⓈ-M Futures REST/SDK.  
> See references at the bottom.

---

## ⚙️ Configuration (env)

> **Never commit real secrets.** Prefer **Google Secret Manager** for production.

Common:
- `EXECUTE_TRADES` (`true|false`) — dry-run gate (set **false** locally)
- `PREDICTOR_BACKEND` (`TSFM2|GEMINI|CHRONOS`)
- `PROB_CALIBRATION_PATH` (default: `.../calibration_tsfm2.json`)
- **Risk**: `MAX_DAILY_LOSS_USD`, `MAX_CONSEC_LOSSES`, `MAX_MDD_USD`
- **Sizing**: `BASE_RISK_USD`, `VOL_SIZE_SCALING`, `VOL_SCALAR_MIN`, `VOL_SCALAR_MAX`

TSFM2 (backend side):
- `TSFM2_URL` — Cloud Run endpoint (e.g., `https://<service>-<hash>-<region>.run.app`)
- `TSFM2_BASE_TF_MIN` — **`15`** (recent patch)
- `AUX_TF_MINS` — additional TFs to fetch (comma-separated)

TSFM_Cloud (Cloud Run container env):
- `TSFM2_DT_SEC` — **`900`** (**15m**)  
- `TSFM2_MC_PATHS` — `2000–4000` (MC paths)  
- `TSFM2_MAX_CONTEXT`, `TSFM2_MAX_HORIZON` — context/horizon caps  
- `TSFM2_API_KEY` — optional microservice auth

---

## 🚀 Deployment

### App Engine (backend)
```bash
cd backend
gcloud app deploy app.yaml cron.yaml
```

**Cron**: set trader to **every 15 minutes** to align with 15m base TF.

### Cloud Run (TSFM_Cloud)
Build & deploy:
```bash
cd TSFM_Cloud
gcloud builds submit --tag gcr.io/$PROJECT/tsfm2:2025-09-24
gcloud run deploy tsfm2   --image gcr.io/$PROJECT/tsfm2:2025-09-24   --region=asia-northeast3   --allow-unauthenticated   --memory=2Gi --cpu=2   --min-instances=1 --max-instances=1   --set-env-vars=TSFM2_DT_SEC=900,TSFM2_MC_PATHS=4000
```

> Memory/CPU limits are configured per-revision. Choose values that avoid OOM during MC sampling.

---

## 🔎 Health & Ops

- **Health**: `GET /health` (backend), `GET /health` (TSFM_Cloud)
- **Trader**: `GET /tasks/trader` (manual trigger OK)
- **Metrics**: `GET /api/metrics*`
- **Journaling**: `GET /api/trades`, `/tasks/journal_sync`
- **Calibration** (daily or ad-hoc): `/tasks/calibrate` or via `tools/retrain_calibration_from_journal.py`

---

## 🧪 Local Dev Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# Important: disable real trading
export EXECUTE_TRADES=false

# Run Flask app (dev)
cd backend && python app.py
# or: gunicorn -b :8080 -w 1 app:server
```

**TSFM_Cloud (local):**
```bash
cd TSFM_Cloud
pip install -r requirements.txt
uvicorn main:app --reload --port 8081
```

---

## 🧮 Calibration workflow

1) Collect realized trades (`LOG_DIR/trades.csv`, GCS snapshots)  
2) Run: `python backend/tools/retrain_calibration_from_journal.py`  
3) Output reliability curve → `calibration_tsfm2.json`  
4) Backend loads curve to **map raw prob → calibrated prob**

---

## 🛡️ Risk Controls (overview)

- **Kill switch**: trip on daily loss, consecutive losses, or realized **MDD**
- **Order hygiene**: `positionSide` in Hedge mode, **idempotent client IDs**, strict `tickSize/stepSize` rounding
- **Cool-down**: time-barrier after exits; avoid late-night low-liquidity windows
- **Notional caps**: balance-% sizing + volatility scalers

---

## 📚 References

- **Gemini (google-genai)**: client usage and JSON Mode (Python) — official docs.  
- **GenAI SDK migration** (from legacy `google.generativeai`).  
- **Gemini API libraries (GA)**.  
- **Binance USDⓈ-M Futures New Order API** (Hedge `positionSide`, `newClientOrderId`).  
- **Cloud Run memory limits**.

---

## ⚠️ Security

- Remove real keys from `.env` and rotate keys. Use **Secret Manager** for prod.
- Restrict Cloud Run ingress (if API key-enabled). Enable audit logging.
- Enforce IP allowlist for admin endpoints where feasible.

---

## 🗺️ Roadmap (abridged)

- ✅ 15m base TF & Cloud Run memory bump (this patch)
- 🔁 Converge TF source-of-truth across backend & TSFM_Cloud
- 📈 Pre-trade EV threshold & post-trade MTTD dashboard
- 🧪 Replay harness (N-day) to gate new configs before prod
- 🔐 Secret Manager integration end-to-end
- 🔍 Structured JSON logs + richer `/api/metrics`

