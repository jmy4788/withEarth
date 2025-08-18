from __future__ import annotations
"""
app.py — Backend API for withEarth_V0 (Liquid Glass ready, Option B)

This Flask app exposes the trading APIs and data feeds used by the React SPA.
It is designed to run locally and on Google App Engine (GAE).

Provided endpoints
- GET  /health                          : simple health check
- GET|POST /tasks/trader                : run signal generation (and optionally execute trades)
- GET  /api/overview                    : account balances & positions overview
- GET  /api/trades?limit=200            : recent trade journal rows from logs/trades.csv
- GET  /api/logs?lines=200              : tail of logs/bot.log
- GET  /api/signals                     : list recent model decisions (debug payloads)
- GET  /api/signals/latest?symbol=...   : on-demand latest signal for a symbol (no execution)
- GET  /api/candles?symbol=...&tf=5m    : OHLCV for charting
- GET  /api/orderbook?symbol=...        : top-of-book depth snapshot

Option B (React SPA separate):
- If FRONTEND_BASE_URL is set, GET / and /dashboard redirect to that URL.
- Otherwise, they render a small info page explaining how to serve the SPA.
"""

import csv
import glob
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from helpers.binance_client import get_open_orders  # 이미 다른 것들 임포트되어 있음
from flask import Flask, jsonify, request, redirect, url_for, send_from_directory, make_response

# Optional but recommended for Option B when serving SPA from other origin
try:
    from flask_cors import CORS
except Exception:  # pragma: no cover
    CORS = None  # type: ignore

# Rotating log to file
from logging.handlers import RotatingFileHandler

# -----------------------------
# Environment & Constants
# -----------------------------
TZ = os.getenv("TZ", "Asia/Seoul")
try:
    import tzset  # type: ignore  # if installed
except Exception:
    try:
        if hasattr(time, "tzset"):
            os.environ["TZ"] = TZ
            time.tzset()
    except Exception:
        pass

PORT = int(os.getenv("PORT", "8080"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
EXECUTE_TRADES = str(os.getenv("EXECUTE_TRADES", "false")).lower() in ("1", "true", "yes")

LOG_DIR_ENV = os.getenv("LOG_DIR", "./logs")
DEFAULT_TMP_DIR = "/tmp/trading_bot"  # GAE writable
PAYLOAD_DIR_NAME = "payloads"

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "").strip()  # e.g., https://spa.example.com

# -----------------------------
# App & Logging
# -----------------------------
def _ensure_dir_writable(pref: str) -> str:
    """Ensure directory exists & is writable; fallback to /tmp on GAE."""
    try:
        p = Path(pref)
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)  # type: ignore
        return str(p)
    except Exception:
        pass
    # fallback
    p = Path(DEFAULT_TMP_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

LOG_DIR = _ensure_dir_writable(LOG_DIR_ENV)
LOG_PATH = str(Path(LOG_DIR) / "bot.log")
# app.py — FIX setup_logging() to return a logger and use it safely

def setup_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    logging.getLogger().handlers.clear()
    logging.basicConfig(level=level, format=fmt)

    # 모듈 개별 로거 레벨
    logging.getLogger("helpers.signals").setLevel(level)
    logging.getLogger("helpers.predictor").setLevel(level)

    # 반환: 현재 모듈 로거
    return logging.getLogger(__name__)

# 호출부
logger = setup_logging()  # 이제 logger는 유효


app = Flask(__name__)
# compat alias (some older deployments import server)
server = app

# CORS (only if flask_cors available). By default allow same-origin; if FRONTEND_BASE_URL set, allow that origin.
if CORS is not None:
    cors_origins = []
    if FRONTEND_BASE_URL:
        cors_origins = [FRONTEND_BASE_URL]
    else:
        # during local dev via Vite default port
        cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    CORS(app, resources={r"/api/*": {"origins": cors_origins}}, supports_credentials=False)

# -----------------------------
# Imports from helpers
# -----------------------------
try:
    from helpers.signals import generate_signal, manage_trade, get_overview as sig_get_overview, maintain_positions  # type: ignore
except Exception as e:
    logger.exception("Failed to import helpers.signals: %s", e)
    # Soft stubs for development without full environment
    def generate_signal(symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "result": {"direction": "hold", "entry": 0, "tp": 0, "sl": 0, "prob": 0.5, "risk_ok": False, "rr": 0.0, "payload_preview": {}}}
    def manage_trade(symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "error": "signals_unavailable"}
    def sig_get_overview() -> Dict[str, Any]:
        return {"balances": [], "positions": []}

try:
    from helpers.data_fetch import fetch_ohlcv, fetch_orderbook  # type: ignore
except Exception as e:
    logger.exception("Failed to import helpers.data_fetch: %s", e)
    def fetch_ohlcv(symbol: str, interval: str = "5m", limit: int = 200):
        import pandas as pd
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    def fetch_orderbook(symbol: str, limit: int = 50):
        return {"bids": [], "asks": [], "timestamp": int(time.time()*1000)}

# -----------------------------
# Utilities
# -----------------------------
def _json_ok(**kwargs):
    resp = {"status": "ok"}
    resp.update(kwargs)
    return jsonify(resp)

def _json_err(msg: str, **kwargs):
    resp = {"status": "error", "message": msg}
    resp.update(kwargs)
    return jsonify(resp), 400

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _tail_file(path: str, lines: int = 200) -> List[str]:
    if not Path(path).exists():
        return []
    # simple tail implementation
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 1024
            data = b""
            while size > 0 and lines > 0:
                step = min(block, size)
                size -= step
                f.seek(size)
                chunk = f.read(step)
                data = chunk + data
                lines = lines - chunk.count(b"\n")
            text = data.decode("utf-8", "ignore")
        return text.strip().splitlines()[-lines:]
    except Exception:
        try:
            return Path(path).read_text(encoding="utf-8").splitlines()[-lines:]
        except Exception:
            return []

def _read_trades_csv(limit: int = 200) -> List[Dict[str, Any]]:
    path = Path(LOG_DIR) / "trades.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append({
                    "timestamp": row.get("timestamp", ""),
                    "symbol": row.get("symbol", ""),
                    "side": row.get("side", ""),
                    "qty": _safe_float(row.get("qty", 0)),
                    "entry": _safe_float(row.get("entry", 0)),
                    "tp": _safe_float(row.get("tp", 0)),
                    "sl": _safe_float(row.get("sl", 0)),
                    "exit": _safe_float(row.get("exit", 0)),
                    "pnl": _safe_float(row.get("pnl", 0)),
                    "status": row.get("status", ""),
                    "id": row.get("id", ""),
                })
        return rows[-limit:]
    except Exception as e:
        logger.info("read trades.csv failed: %s", e)
        return []

def _signals_debug_list(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Scan logs/payloads/YYYYMMDD/*_decision.json files and return latest ones.
    """
    base = Path(LOG_DIR) / PAYLOAD_DIR_NAME
    if not base.exists():
        return []
    # Search today and recent 1 day
    patterns = []
    today = datetime.now(tz=timezone.utc)
    for d in [today, today - timedelta(days=1)]:
        p = base / d.strftime("%Y%m%d") / "*_decision.json"
        patterns.append(str(p))
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(files)[-limit:]
    items: List[Dict[str, Any]] = []
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text(encoding="utf-8"))
            items.append({"file": Path(fp).name, "decision": data})
        except Exception:
            continue
    return items[-limit:]

# -----------------------------
# Routes
# -----------------------------
@app.route("/health")
def health():
    return _json_ok()

@app.route("/")
def root():
    if FRONTEND_BASE_URL:
        return redirect(FRONTEND_BASE_URL, code=302)
    # fallback info page
    html = f"""
    <!doctype html>
    <html><head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>withEarth_V0 API</title>
      <style>body{{font-family:ui-sans-serif,system-ui;-webkit-font-smoothing:antialiased;background:#0b1220;color:#cbd5e1;padding:24px}}</style>
    </head><body>
      <h1>withEarth_V0 API</h1>
      <p>This backend is running. Set <code>FRONTEND_BASE_URL</code> to redirect to your React SPA, or open <code>/api/*</code> endpoints directly.</p>
      <ul>
        <li><a href="/health">/health</a></li>
        <li><a href="/api/overview">/api/overview</a></li>
        <li><a href="/api/trades">/api/trades</a></li>
        <li><a href="/api/logs">/api/logs</a></li>
        <li><a href="/api/signals">/api/signals</a></li>
      </ul>
    </body></html>
    """
    return make_response(html, 200)

@app.route("/dashboard")
def dashboard_redirect():
    # Option B: redirect to SPA
    if FRONTEND_BASE_URL:
        return redirect(FRONTEND_BASE_URL, code=302)
    # else same as root()
    return root()

@app.route("/tasks/trader", methods=["GET", "POST"])
def tasks_trader():
    """
    Trigger signal generation (and optionally execution) for given symbols.
    Query/body:
      - symbols: comma-separated (default from env SYMBOLS)
      - debug=1 : include exceptions in result
    """
    debug = str(request.args.get("debug") or request.form.get("debug") or "0") in ("1", "true", "yes")
    symbols_raw = request.args.get("symbols") or request.form.get("symbols") or ",".join(SYMBOLS)
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    results: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            # Triple-barrier/time cleanup before entering new trades
            maint = None
            try:
                maint = maintain_positions(sym)
            except Exception:
                maint = {"action": "none"}
            if EXECUTE_TRADES:
                out = manage_trade(sym)
                results.append({"symbol": sym, "maintenance": maint, "result": out})
            else:
                sig = generate_signal(sym)
                results.append({"symbol": sym, "maintenance": maint, "result": sig.get("result") or sig})
        except Exception as e:
            if debug:
                results.append({"symbol": sym, "error": repr(e)})
            else:
                results.append({"symbol": sym, "error": "failed"})
    return _json_ok(results=results, executed=bool(EXECUTE_TRADES))

@app.route("/api/overview")
def api_overview():
    try:
        ov = sig_get_overview()
        return _json_ok(overview=ov)
    except Exception as e:
        return _json_err("overview_failed", error=str(e))

@app.route("/api/trades")
def api_trades():
    try:
        limit = int(request.args.get("limit", "200"))
    except Exception:
        limit = 200
    rows = _read_trades_csv(limit=limit)
    total = len(rows)
    pnl = sum(float(r.get("pnl", 0)) for r in rows)
    wins = sum(1 for r in rows if float(r.get("pnl", 0)) > 0)
    open_cnt = sum(1 for r in rows if str(r.get("status", "")).lower() == "open")
    summary = {"total": total, "pnl": pnl, "win": wins, "open": open_cnt}
    return _json_ok(rows=rows, summary=summary)

@app.route("/api/logs")
def api_logs():
    try:
        lines = int(request.args.get("lines", "200"))
    except Exception:
        lines = 200
    tail = _tail_file(LOG_PATH, lines=lines)
    return _json_ok(lines=tail)

@app.route("/api/signals")
def api_signals():
    items = _signals_debug_list(limit=50)
    return _json_ok(items=items)

@app.route("/api/signals/latest")
def api_signals_latest():
    symbol = (request.args.get("symbol") or SYMBOLS[0]).upper()
    try:
        sig = generate_signal(symbol)
        # Ensure shape friendly to frontend
        out = sig.get("result") if isinstance(sig, dict) else None
        return _json_ok(symbol=symbol, **({"result": out} if out else {"raw": sig}))
    except Exception as e:
        return _json_err("signal_failed", error=str(e))

@app.route("/api/candles")
def api_candles():
    symbol = (request.args.get("symbol") or SYMBOLS[0]).upper()
    tf = request.args.get("tf", "5m")
    try:
        limit = int(request.args.get("limit", "500"))
    except Exception:
        limit = 500
    try:
        df = fetch_ohlcv(symbol, interval=tf, limit=limit)
        # to list of dicts {t,o,h,l,c,v}
        rows: List[Dict[str, Any]] = []
        if df is not None and len(df) > 0:
            for _, r in df.iterrows():
                rows.append({
                    "t": str(r.get("timestamp", "")),
                    "o": float(r.get("open", 0)),
                    "h": float(r.get("high", 0)),
                    "l": float(r.get("low", 0)),
                    "c": float(r.get("close", 0)),
                    "v": float(r.get("volume", 0)),
                })
        return _json_ok(symbol=symbol, tf=tf, candles=rows[-limit:])
    except Exception as e:
        logger.info("api_candles failed: %s", e)
        return _json_err("candles_failed", error=str(e))

@app.route("/api/orderbook")
def api_orderbook():
    symbol = (request.args.get("symbol") or SYMBOLS[0]).upper()
    try:
        limit = int(request.args.get("limit", "10"))
    except Exception:
        limit = 10
    try:
        ob = fetch_orderbook(symbol, limit=limit)
        return _json_ok(symbol=symbol, orderbook=ob)
    except Exception as e:
        return _json_err("orderbook_failed", error=str(e))
# app.py 상단 import에 추가


# 아래와 같이 라우트 추가
@app.route("/api/open_orders", methods=["GET"])
def api_open_orders():
    symbol = request.args.get("symbol")
    try:
        orders = get_open_orders(symbol)
        return _json_ok(orders=orders)
    except Exception as e:
        return _json_err(f"open_orders_failed: {e}")

# -----------------------------
# WSGI Entrypoint
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
