# app.py — single-file backend (v2 → v2.1 analytics), 2025-08-22
from __future__ import annotations
"""
변경 요약
- /tasks/calibrate: robust import of external loader with fallback (중복 데코레이터 제거)
- /api/metrics, /api/metrics/curve, /api/metrics/calibration 추가
- 기존 /api/*, /tasks/*, /health 그대로 유지
"""

import csv, glob, json, logging, os, time, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

TZ = os.getenv("TZ", "Asia/Seoul")
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
DEFAULT_TMP_DIR = "/tmp/trading_bot"
PAYLOAD_DIR_NAME = "payloads"
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "").strip()

try:
    from flask_cors import CORS  # type: ignore
except Exception:  # pragma: no cover
    CORS = None  # type: ignore

from flask import Flask, jsonify, request, redirect, make_response, Blueprint

def _ensure_dir_writable(pref: str) -> str:
    try:
        p = Path(pref); p.mkdir(parents=True, exist_ok=True)
        t = p / ".write_test"; t.write_text("ok", encoding="utf-8"); t.unlink(missing_ok=True)  # type: ignore
        return str(p)
    except Exception:
        p = Path(DEFAULT_TMP_DIR); p.mkdir(parents=True, exist_ok=True); return str(p)

LOG_DIR = _ensure_dir_writable(LOG_DIR_ENV)
LOG_PATH = str(Path(LOG_DIR) / "bot.log")

def setup_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger("helpers.signals").setLevel(level)
    logging.getLogger("helpers.predictor").setLevel(level)
    return logging.getLogger(__name__)

logger = setup_logging()

# --- imports from helpers ---
try:
    from helpers.signals import generate_signal, manage_trade, get_overview as sig_get_overview, maintain_positions  # noqa
except Exception as e:
    logger.exception("helpers.signals import failed: %s", e)
    def generate_signal(symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "result": {"direction": "hold", "entry": 0, "tp": 0, "sl": 0, "prob": 0.5, "risk_ok": False, "rr": 0.0}}
    def manage_trade(symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "error": "signals_unavailable"}
    def sig_get_overview() -> Dict[str, Any]:
        return {"balances": [], "positions": []}

try:
    from helpers.data_fetch import fetch_ohlcv, fetch_orderbook  # noqa
except Exception as e:
    logger.exception("helpers.data_fetch import failed: %s", e)
    def fetch_ohlcv(symbol: str, interval: str = "5m", limit: int = 200):
        import pandas as pd
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    def fetch_orderbook(symbol: str, limit: int = 50):
        return {"bids": [], "asks": [], "timestamp": int(time.time()*1000)}

try:
    from helpers.binance_client import get_open_orders  # noqa
except Exception as e:
    logger.info("helpers.binance_client.get_open_orders import failed: %s", e)
    def get_open_orders(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

try:
    from helpers.utils import log_event, gcs_enabled  # noqa
except Exception:
    def log_event(event: str, **fields): logger.info("[event]%s %s", event, fields)
    def gcs_enabled() -> bool: return False

# --- small utils for API ---
def _plainify(o):
    if isinstance(o, dict): return {k: _plainify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_plainify(x) for x in o]
    for attr in ("model_dump", "dict"):
        if hasattr(o, attr):
            try: return _plainify(getattr(o, attr)())
            except Exception: pass
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(o): return _plainify(asdict(o))
    except Exception: pass
    if hasattr(o, "__dict__"):
        try: return {k: _plainify(v) for k, v in o.__dict__.items() if not str(k).startswith("_")}
        except Exception: pass
    return o

def _json_ok(**kwargs): return jsonify(_plainify({"status":"ok", **kwargs}))
def _json_err(msg: str, **kwargs): return jsonify(_plainify({"status":"error","message":msg, **kwargs})), 400
def _safe_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

def _tail_file(path: str, lines: int = 200) -> List[str]:
    if not Path(path).exists(): return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell(); block = 1024; data = b""
            while size > 0 and lines > 0:
                step = min(block, size); size -= step; f.seek(size)
                chunk = f.read(step); data = chunk + data; lines = lines - chunk.count(b"\n")
            text = data.decode("utf-8", "ignore")
        return text.strip().splitlines()[-lines:]
    except Exception:
        try: return Path(path).read_text(encoding="utf-8").splitlines()[-lines:]
        except Exception: return []

def _read_trades_csv(limit: int = 200) -> List[Dict[str, Any]]:
    path = Path(LOG_DIR) / "trades.csv"
    if not path.exists(): return []
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
        logger.info("read trades.csv failed: %s", e); return []

def _signals_debug_list(limit: int = 50) -> List[Dict[str, Any]]:
    base = Path(LOG_DIR) / PAYLOAD_DIR_NAME
    if not base.exists(): return []
    patterns = []
    today = datetime.now(tz=timezone.utc)
    for d in [today, today - timedelta(days=1)]:
        patterns.append(str(base / d.strftime("%Y%m%d") / "*_decision.json"))
    files: List[str] = []
    for pat in patterns: files.extend(glob.glob(pat))
    files = sorted(files)[-limit:]
    items: List[Dict[str, Any]] = []
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text(encoding="utf-8"))
            items.append({"file": Path(fp).name, "decision": data})
        except Exception:
            continue
    return items[-limit:]

# ======================================================================
# Blueprint 1: Calibration (/tasks/calibrate) — robust loader + fallback
# ======================================================================
calib_bp = Blueprint("tasks_calibrate", __name__)

@dataclass
class _Sample:
    prob: float
    label: int

def _fallback_load_samples(trades_csv: Path, only_symbols: List[str]) -> List[_Sample]:
    out: List[_Sample] = []
    if not trades_csv.exists(): return out
    try:
        with open(trades_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sym = str(row.get("symbol","")).upper()
                if only_symbols and sym not in only_symbols: continue
                st = str(row.get("status","")).lower()
                if st not in ("closed_tp", "closed_sl"): continue
                try:
                    p = float(row.get("prob", ""))
                except Exception:
                    continue
                out.append(_Sample(prob=p, label=(1 if st == "closed_tp" else 0)))
    except Exception as e:
        logger.info("fallback sample load failed: %s", e)
    return out

def _import_samples_loader():
    """
    Returns callable load_samples(trades_csv: Path, horizon_min: int, only_symbols: List[str]) -> List[_Sample]
    from:
    - CALIB_LOADER_MODULE (env)  e.g., 'backend.tools.calibrate_from_trades'
    - tools.calibrate_from_trades
    - backend.tools.calibrate_from_trades
    - calibrate_from_trades
    """
    import importlib
    prefer = os.getenv("CALIB_LOADER_MODULE", "").strip()
    candidates = [prefer] if prefer else []
    candidates += ["tools.calibrate_from_trades", "backend.tools.calibrate_from_trades", "calibrate_from_trades"]
    for modname in candidates:
        if not modname: continue
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, "load_samples", None)
            if callable(fn):
                def _adapter(trades_csv: Path, horizon_min: int, only_symbols: List[str]) -> List[_Sample]:
                    samples = fn(trades_csv, horizon_min, only_symbols=only_symbols)
                    return [ _Sample(prob=float(getattr(s, "prob")), label=int(getattr(s, "label"))) for s in samples ]
                logger.info("Calibration loader resolved: %s.load_samples", modname)
                return _adapter
        except Exception as e:
            logger.info("Candidate loader miss: %s (%s)", modname, e)
    return None

@calib_bp.route("/tasks/calibrate", methods=["GET", "POST"])
def tasks_calibrate():
    # App Engine Cron 보호: 내부 크론 호출만 허용
    if request.headers.get("X-Appengine-Cron", "").lower() != "true":
        return jsonify({"error": "forbidden"}), 403

    try:
        from helpers.calibration import ProbCalibrator, reload_global  # 저장/핫리로드
    except Exception as e:
        return jsonify({"status": "error", "message": f"calibrator_import_failed: {e}"}), 500

    # 샘플 로더 결정
    loader = _import_samples_loader()
    if loader is None:
        def loader(trades_csv: Path, horizon_min: int, only_symbols: List[str]) -> List[_Sample]:
            return _fallback_load_samples(trades_csv, only_symbols)

    symbols_env = os.getenv("SYMBOLS", "")
    only_symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    horizon_min = int(os.getenv("HORIZON_MIN", "30"))
    min_samples = int(os.getenv("CALIB_MIN_SAMPLES", "150"))
    bins = int(os.getenv("CALIB_BINS", "10"))

    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    trades_csv = log_dir / "trades.csv"

    # 1) collect samples
    samples = loader(trades_csv, horizon_min, only_symbols=only_symbols)
    n = len(samples)
    log_event("calibrate.collect", symbols=only_symbols or "ALL", horizon_min=horizon_min, samples=n)

    if n < min_samples:
        return jsonify({"status": "skipped", "reason": "insufficient_samples", "n": n, "min_samples": min_samples}), 200

    probs = [s.prob for s in samples]
    labels = [s.label for s in samples]

    # 2) fit + save
    calib = ProbCalibrator(bins=bins, min_samples=min_samples)
    ok = calib.fit_from_arrays(probs, labels)
    if not ok:
        return jsonify({"status": "failed", "reason": "fit_failed"}), 200

    calib.save()
    log_event("calibrate.saved", path=calib.path, bins=bins, n=n)

    # 2.5) in-memory 테이블 핫리로드
    try:
        reload_global()
        log_event("calibrate.reloaded")
    except Exception as e:
        log_event("calibrate.reload_failed", error=str(e))

    # 3) optional GCS upload
    if gcs_enabled():
        try:
            from google.cloud import storage  # type: ignore
            client = storage.Client()
            bucket = client.bucket(os.getenv("GCS_BUCKET"))
            prefix = os.getenv("GCS_PREFIX", "trading_bot")
            dst = f"{prefix}/calibration/latest/calibration.json"
            blob = bucket.blob(dst)
            blob.cache_control = "no-cache"
            blob.upload_from_filename(calib.path, content_type="application/json")
            log_event("calibrate.gcs_uploaded", gcs_path=dst)
        except Exception as e:
            log_event("calibrate.gcs_upload_failed", error=str(e))

    return jsonify({"status": "ok", "saved_to": calib.path, "samples_used": n, "bins": bins}), 200

# ======================================================================
# Blueprint 2: Trader (/tasks/trader, /tasks/maintain)
# ======================================================================
trader_bp = Blueprint("tasks_trader", __name__)

def _symbols_from_env() -> List[str]:
    return [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]

def _is_cron(req) -> bool:
    return str(req.headers.get("X-Appengine-Cron", "")).lower() == "true"

@trader_bp.route("/tasks/trader", methods=["GET", "POST"])
def tasks_trader():
    if not _is_cron(request):
        return jsonify({"error": "forbidden"}), 403
    syms = _symbols_from_env()
    exec_live = str(os.getenv("EXECUTE_TRADES", "false")).lower() in ("1", "true", "yes")
    out: List[Dict[str, Any]] = []
    for sym in syms:
        try:
            if exec_live:
                res = manage_trade(sym)
            else:
                sig = generate_signal(sym)
                res = {"symbol": sym, "action": "dryrun", "signal": sig}
        except Exception as e:
            res = {"symbol": sym, "error": str(e)}
        out.append(res)
    log_event("tasks.trader", execute=exec_live, results=len(out))
    return jsonify({"status":"ok","execute":exec_live,"results":out}), 200

@trader_bp.route("/tasks/maintain", methods=["GET", "POST"])
def tasks_maintain():
    if not _is_cron(request):
        return jsonify({"error": "forbidden"}), 403
    syms = _symbols_from_env()
    results: Dict[str, Any] = {}
    for sym in syms:
        try:
            r = maintain_positions(sym)
        except Exception as e:
            r = {"action":"error","error":str(e)}
        results[sym] = r
    log_event("tasks.maintain", symbols=syms, n=len(syms))
    return jsonify({"status":"ok","results":results}), 200

# ======================================================================
# Flask app & API routes
# ======================================================================
app = Flask(__name__)
server = app
app.register_blueprint(calib_bp)
app.register_blueprint(trader_bp)

if CORS is not None:
    cors_origins = [FRONTEND_BASE_URL] if FRONTEND_BASE_URL else ["http://localhost:5173","http://127.0.0.1:5173"]
    CORS(app, resources={r"/api/*": {"origins": cors_origins}}, supports_credentials=False)

@app.route("/health")
def health(): return _json_ok()

@app.route("/")
def root():
    if FRONTEND_BASE_URL: return redirect(FRONTEND_BASE_URL, code=302)
    html = """
    <!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>withEarth_V0 API</title>
    <style>body{font-family:ui-sans-serif,system-ui;-webkit-font-smoothing:antialiased;background:#0b1220;color:#cbd5e1;padding:24px}</style>
    </head><body>
    <h1>withEarth_V0 API</h1>
    <ul>
      <li><a href="/health">/health</a></li>
      <li><a href="/api/overview">/api/overview</a></li>
      <li><a href="/api/trades">/api/trades</a></li>
      <li><a href="/api/logs">/api/logs</a></li>
      <li><a href="/api/signals">/api/signals</a></li>
      <li><a href="/api/metrics">/api/metrics</a></li>
    </ul></body></html>"""
    return make_response(html, 200)

@app.route("/dashboard")
def dashboard_redirect():
    if FRONTEND_BASE_URL: return redirect(FRONTEND_BASE_URL, code=302)
    return root()

@app.route("/api/overview")
def api_overview():
    try:
        ov = sig_get_overview()
        return _json_ok(overview=ov)
    except Exception as e:
        return _json_err("overview_failed", error=str(e))

@app.route("/api/trades")
def api_trades():
    try: limit = int(request.args.get("limit","200"))
    except Exception: limit = 200
    rows = _read_trades_csv(limit=limit)
    total = len(rows); pnl = sum(float(r.get("pnl", 0)) for r in rows)
    wins = sum(1 for r in rows if float(r.get("pnl", 0)) > 0)
    open_cnt = sum(1 for r in rows if str(r.get("status", "")).lower() == "open")
    summary = {"total": total, "pnl": pnl, "win": wins, "open": open_cnt}
    return _json_ok(rows=rows, summary=summary)

@app.route("/api/logs")
def api_logs():
    try: lines = int(request.args.get("lines","200"))
    except Exception: lines = 200
    return _json_ok(lines=_tail_file(LOG_PATH, lines=lines))

@app.route("/api/signals")
def api_signals():
    return _json_ok(items=_signals_debug_list(limit=50))

@app.route("/api/signals/latest")
def api_signals_latest():
    symbol = (request.args.get("symbol") or SYMBOLS[0]).upper()
    try:
        sig = generate_signal(symbol)
        out = sig.get("result") if isinstance(sig, dict) else None
        return _json_ok(symbol=symbol, **({"result": out} if out else {"raw": sig}))
    except Exception as e:
        return _json_err("signal_failed", error=str(e))

@app.route("/api/candles")
def api_candles():
    symbol = (request.args.get("symbol") or SYMBOLS[0]).upper()
    tf = request.args.get("tf", "5m")
    try: limit = int(request.args.get("limit","500"))
    except Exception: limit = 500
    try:
        df = fetch_ohlcv(symbol, interval=tf, limit=limit)
        rows: List[Dict[str, Any]] = []
        if df is not None and len(df) > 0:
            for _, r in df.iterrows():
                rows.append({"t": str(r.get("timestamp","")), "o": float(r.get("open",0)),
                             "h": float(r.get("high",0)), "l": float(r.get("low",0)),
                             "c": float(r.get("close",0)), "v": float(r.get("volume",0))})
        return _json_ok(symbol=symbol, tf=tf, candles=rows[-limit:])
    except Exception as e:
        logger.info("api_candles failed: %s", e)
        return _json_err("candles_failed", error=str(e))

@app.route("/api/orderbook")
def api_orderbook():
    symbol = (request.args.get("symbol") or SYMBOLS[0]).upper()
    try: limit = int(request.args.get("limit","10"))
    except Exception: limit = 10
    try:
        ob = fetch_orderbook(symbol, limit=limit)
        return _json_ok(symbol=symbol, orderbook=ob)
    except Exception as e:
        return _json_err(f"orderbook_failed: {e}")

@app.route("/api/open_orders", methods=["GET"])
def api_open_orders():
    symbol = request.args.get("symbol")
    try:
        orders = get_open_orders(symbol)
        return _json_ok(orders=orders)
    except Exception as e:
        return _json_err(f"open_orders_failed: {e}")

# ==============================
# Analytics: metrics endpoints
# ==============================
def _parse_bool(s: str, default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).lower() in ("1","true","yes","y","on")

def _coerce_label(status: str, pnl: float) -> int:
    """1=TP 승, 0=SL/시간/패 (tools/calibrate_from_trades.py 라벨링과 정렬)"""
    st = str(status or "").strip().lower()
    if "closed_tp" in st or ("tp" in st and "stop" not in st):
        return 1
    if "closed_sl" in st or "stop" in st or "time_exit" in st:
        return 0
    return 1 if float(pnl or 0.0) > 0 else 0

def _read_trades_full(limit: int = 5000) -> list[dict]:
    path = Path(LOG_DIR) / "trades.csv"
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    rows.append({
                        "timestamp": row.get("timestamp",""),
                        "symbol": row.get("symbol","").upper(),
                        "side": row.get("side",""),
                        "qty": _safe_float(row.get("qty",0)),
                        "entry": _safe_float(row.get("entry",0)),
                        "tp": _safe_float(row.get("tp",0)),
                        "sl": _safe_float(row.get("sl",0)),
                        "exit": _safe_float(row.get("exit",0)),
                        "pnl": _safe_float(row.get("pnl",0)),
                        "status": row.get("status",""),
                        "id": row.get("id",""),
                        "prob": _safe_float(row.get("prob",0.0)),
                        "prob_raw": _safe_float(row.get("prob_raw", float("nan"))),
                        "prob_cal": _safe_float(row.get("prob_cal", float("nan"))),
                        "rr": _safe_float(row.get("rr",0.0)),
                        "entry_maker": int(row.get("entry_maker","0") or "0"),
                        "mode": row.get("mode",""),
                        "reprices": int(row.get("reprices","0") or "0"),
                        "used_market_fallback": int(row.get("used_market_fallback","0") or "0"),
                        "post_only": int(row.get("post_only","0") or "0"),
                        "spread_bps": _safe_float(row.get("spread_bps",0.0)),
                        "atr_now": _safe_float(row.get("atr_now",0.0)),
                        "funding_pct": _safe_float(row.get("funding_pct",0.0)),
                        "reasons": row.get("reasons",""),
                    })
                except Exception:
                    continue
        return rows[-limit:]
    except Exception:
        return []

def _filter_trades(rows: list[dict], symbol: str|None, days: int|None, include_open: bool) -> list[dict]:
    if symbol:
        rows = [r for r in rows if r.get("symbol","").upper()==symbol.upper()]
    if days and days>0:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        out = []
        for r in rows:
            ts = r.get("timestamp","")
            try:
                dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                if dt >= cutoff:
                    out.append(r)
            except Exception:
                out.append(r)  # 보수적으로 포함
        rows = out
    if not include_open:
        rows = [r for r in rows if str(r.get("status","")).lower()!="open"]
    return rows

def _equity_curve(rows: list[dict]) -> tuple[list[float], list[float]]:
    eq = []
    dd = []
    s = 0.0
    peak = 0.0
    for r in rows:
        s += float(r.get("pnl",0.0))
        peak = max(peak, s)
        eq.append(s)
        dd.append(peak - s)
    return eq, dd

def _brier(arr: list[tuple[float,int]]) -> float|None:
    if not arr:
        return None
    return sum((p-y)**2 for p,y in arr)/len(arr)

def _logloss(arr: list[tuple[float,int]]) -> float|None:
    if not arr:
        return None
    eps = 1e-12
    return -sum(y*math.log(max(eps,min(1-eps,p))) + (1-y)*math.log(max(eps,min(1-eps,1-p))) for p,y in arr)/len(arr)

def _auc(arr: list[tuple[float,int]]) -> float|None:
    """Mann-Whitney U 기반 AUC (tie 평균 랭크 처리)."""
    if not arr:
        return None
    n1 = sum(1 for _,y in arr if y==1)
    n0 = sum(1 for _,y in arr if y==0)
    if n1==0 or n0==0:
        return None
    arr_sorted = sorted(arr, key=lambda x: x[0])
    i = 0
    rank_sum_pos = 0.0
    n = len(arr_sorted)
    while i < n:
        j = i
        val = arr_sorted[i][0]
        while j < n and arr_sorted[j][0]==val:
            j+=1
        avg_rank = (i+1 + j)/2.0
        pos_in_tie = sum(1 for k in range(i,j) if arr_sorted[k][1]==1)
        rank_sum_pos += avg_rank*pos_in_tie
        i = j
    U = rank_sum_pos - n1*(n1+1)/2.0
    return U/(n1*n0)

def _reliability_bins(arr: list[tuple[float,int]], bins: int=10) -> dict:
    if not arr or bins<3:
        return {"points": [], "ece": None}
    edges = [i/bins for i in range(bins+1)]
    pts = []
    total = len(arr)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        seg = [(p,y) for (p,y) in arr if (p>=lo and (p<hi if i<bins-1 else p<=hi))]
        if len(seg) < 3:
            continue
        mean_p = sum(p for p,_ in seg)/len(seg)
        mean_y = sum(y for _,y in seg)/len(seg)
        w = len(seg)/total
        ece += w*abs(mean_y-mean_p)
        pts.append({"x": mean_p, "y": mean_y, "n": len(seg)})
    return {"points": pts, "ece": ece}

def _basic_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n":0}
    pnl = [float(r.get("pnl",0.0)) for r in rows]
    n = len(rows)
    pos = [x for x in pnl if x>0]
    neg = [x for x in pnl if x<=0]
    total = sum(pnl)
    wins = len(pos)
    losses = len(neg)
    pf = (sum(pos)/max(1e-12, -sum(neg))) if losses>0 else None
    exp = total/n
    mu = exp
    sigma = (sum((x-mu)**2 for x in pnl)/n)**0.5 if n>1 else 0.0
    eq, dd = _equity_curve(rows)
    max_dd = max(dd) if dd else 0.0
    return {
        "n": n, "wins": wins, "losses": losses, "win_rate": wins/max(1,n),
        "total_pnl": total, "avg_pnl": exp, "profit_factor": pf,
        "std_pnl": sigma, "max_drawdown": max_dd,
    }

@app.route("/api/metrics")
def api_metrics():
    try: limit = int(request.args.get("limit","500"))
    except Exception: limit = 500
    symbol = request.args.get("symbol")
    try: days = int(request.args.get("days","0"))
    except Exception: days = 0
    try: bins = int(request.args.get("bins","10"))
    except Exception: bins = 10
    include_open = _parse_bool(request.args.get("include_open","false"))

    rows_all = _read_trades_full(limit=5000)
    rows = _filter_trades(rows_all[-limit:], symbol, days, include_open)
    if not rows:
        return _json_ok(summary={"n":0}, prediction={}, reliability={}, execution={}, per_symbol={})

    # labels & probs
    labels = [_coerce_label(r.get("status",""), float(r.get("pnl",0.0))) for r in rows]
    probs_cal = [float(r.get("prob",0.0)) for r in rows]
    # prob_raw가 없을 수 있음
    probs_raw = []
    labels_for_raw = []
    for i, r in enumerate(rows):
        prx = float(r.get("prob_raw")) if r.get("prob_raw") not in (None, "", "nan") else float("nan")
        if prx == prx:  # not NaN
            probs_raw.append(prx)
            labels_for_raw.append(labels[i])

    arr_cal = list(zip(probs_cal, labels))
    arr_raw = list(zip(probs_raw, labels_for_raw)) if probs_raw else []

    # summary
    summary = _basic_stats(rows)
    # prediction quality
    pred = {
        "brier_cal": _brier(arr_cal),
        "logloss_cal": _logloss(arr_cal),
        "auc_cal": _auc(arr_cal),
    }
    if arr_raw:
        b_raw = _brier(arr_raw)
        b_cal = _brier(arr_cal)
        pred.update({
            "brier_raw": b_raw,
            "logloss_raw": _logloss(arr_raw),
            "auc_raw": _auc(arr_raw),
            "improvement_brier": (b_raw - b_cal) if (b_raw is not None and b_cal is not None) else None,
        })
    # reliability
    rel_cal = _reliability_bins(arr_cal, bins=bins)
    rel_raw = _reliability_bins(arr_raw, bins=bins) if arr_raw else {"points": [], "ece": None}

    # execution quality
    maker_ratio = sum(int(r.get("entry_maker",0)) for r in rows)/max(1,len(rows))
    fallback_ratio = sum(int(r.get("used_market_fallback",0)) for r in rows)/max(1,len(rows))
    avg_spread = sum(float(r.get("spread_bps",0.0)) for r in rows)/max(1,len(rows))
    avg_atr = sum(float(r.get("atr_now",0.0)) for r in rows)/max(1,len(rows))
    execq = {
        "entry_maker_ratio": maker_ratio,
        "market_fallback_ratio": fallback_ratio,
        "avg_spread_bps": avg_spread,
        "avg_atr": avg_atr,
    }

    # per-symbol rollup
    per_sym: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        s = r.get("symbol","")
        per_sym.setdefault(s, {"n":0,"pnl":0.0,"wins":0})
        per_sym[s]["n"] += 1
        per_sym[s]["pnl"] += float(r.get("pnl",0.0))
        per_sym[s]["wins"] += 1 if _coerce_label(r.get("status",""), float(r.get("pnl",0.0)))==1 else 0
    for s,v in per_sym.items():
        v["win_rate"] = v["wins"]/max(1,v["n"])

    return _json_ok(
        summary=summary,
        prediction=pred,
        reliability={"calibrated": rel_cal, "raw": rel_raw},
        execution=execq,
        per_symbol=per_sym,
        sample_size=len(rows),
    )

@app.route("/api/metrics/curve")
def api_metrics_curve():
    try: limit = int(request.args.get("limit","500"))
    except Exception: limit = 500
    symbol = request.args.get("symbol")
    try: days = int(request.args.get("days","0"))
    except Exception: days = 0
    include_open = _parse_bool(request.args.get("include_open","false"))
    rows = _filter_trades(_read_trades_full(limit=5000)[-limit:], symbol, days, include_open)
    if not rows:
        return _json_ok(points=[])
    eq, dd = _equity_curve(rows)
    points = [{"t": rows[i]["timestamp"], "equity": eq[i], "drawdown": dd[i]} for i in range(len(rows))]
    return _json_ok(points=points)

@app.route("/api/metrics/calibration")
def api_metrics_calibration():
    # 현재 저장된 보정 테이블과, 최신 샘플 기준 ECE/Brier를 함께 노출
    try:
        from helpers.calibration import ProbCalibrator
        cal = ProbCalibrator()
        be = getattr(cal, "bin_edges", []) or []
        bm = getattr(cal, "bin_means", []) or []
    except Exception as e:
        be, bm = [], []
    try: limit = int(request.args.get("limit","1000"))
    except Exception: limit = 1000
    try: bins = int(request.args.get("bins","10"))
    except Exception: bins = 10
    rows = _filter_trades(_read_trades_full(limit=5000)[-limit:], symbol=request.args.get("symbol"), days=None, include_open=False)
    labels = [_coerce_label(r.get("status",""), float(r.get("pnl",0.0))) for r in rows]
    probs = [float(r.get("prob",0.0)) for r in rows]
    arr = list(zip(probs, labels))
    return _json_ok(
        calibration_table={"bin_edges": be, "bin_means": bm},
        current_sample={"brier": _brier(arr), "rel": _reliability_bins(arr, bins=bins)},
        n=len(rows),
    )

# ======================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
