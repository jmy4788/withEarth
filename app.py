from __future__ import annotations
"""
app.py — Dashboard Edition (Liquid Glass)

- Keeps existing API: /health, /tasks/trader
- Adds user dashboard at    GET /dashboard
- Adds data APIs for UI:    GET /api/overview, /api/trades, /api/logs, /api/signals

Notes
- Trades journal is read from logs/trades.csv (CSV header created on first write).
  You can append to this file from helpers/signals.py when an order is placed/closed.
  See the comment block near the bottom (INTEGRATION HINTS) for code snippets.
- The UI uses a "Liquid Glass" (glassmorphism) style with subtle gradients and blur.
- No external build tools; Tailwind CDN + Chart.js CDN are used.

Author: GPT-5 Thinking
"""

import os
import io
import csv
import json
import glob
import time
import math
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, Response

# -----------------------------
# Project imports
# -----------------------------
try:
    from helpers.signals import generate_signal, manage_trade  # type: ignore
except Exception as e:
    raise

try:
    from helpers.binance_client import get_overview as _bn_overview  # type: ignore
except Exception:
    _bn_overview = None  # type: ignore

try:
    from helpers.data_fetch import fetch_orderbook, fetch_ohlcv  # type: ignore
except Exception:
    fetch_orderbook = None  # type: ignore
    fetch_ohlcv = None  # type: ignore

# -----------------------------
# App / Logger
# -----------------------------
server = Flask(__name__)


def setup_logging() -> logging.Logger:
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "bot.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when reloaded
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)

    if not any(h.__class__.__name__ == "StreamHandler" for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(sh)

    logging.info("Logging initialized. -> %s", log_path)
    return logger


logger = setup_logging()
server.logger.handlers = []
server.logger.propagate = True

# -----------------------------
# Config helpers
# -----------------------------

def _now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _parse_symbols(param: Optional[str]) -> List[str]:
    if not param:
        param = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    syms = [s.strip().upper() for s in str(param).split(",") if s.strip()]
    return syms or ["BTCUSDT", "ETHUSDT"]


# -----------------------------
# Core routes (existing)
# -----------------------------
@server.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@server.route("/tasks/trader", methods=["GET", "POST"])
def tasks_trader():
    try:
        payload_json: Dict[str, Any] = {}
        if request.method == "POST":
            payload_json = request.get_json(silent=True) or {}

        symbols = _parse_symbols(payload_json.get("symbols") or request.args.get("symbols"))
        do_trade = _bool_env("EXECUTE_TRADES", False)

        results: List[Dict[str, Any]] = []
        for sym in symbols:
            try:
                sig = generate_signal(sym)

                exec_info = None
                if do_trade:
                    try:
                        # Reuse built signal to avoid rebuilding payload
                        exec_info = manage_trade(sym, sig)  # type: ignore[arg-type]
                    except Exception:
                        logging.exception("manage_trade failed for %s", sym)
                        exec_info = {"error": "manage_trade failed"}

                results.append({
                    "symbol": sym,
                    "result": {
                        "action": sig.get("action", sig.get("direction", "hold")),
                        "direction": sig.get("direction", "hold"),
                        "entry": sig.get("entry", 0.0),
                        "tp": sig.get("tp", 0.0),
                        "sl": sig.get("sl", 0.0),
                        "prob": sig.get("prob", 0.5),
                        "risk_ok": sig.get("risk_ok", False),
                        "rr": sig.get("rr", 0.0),
                        "reason": sig.get("reasoning", sig.get("reason", "")),
                        "payload_preview": sig.get("payload_preview", {}),
                    },
                    **({"exec": exec_info} if exec_info else {}),
                })
            except Exception as e:
                logging.exception("tasks_trader symbol=%s failed", sym)
                results.append({"symbol": sym, "error": str(e)})

        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        logging.exception("/tasks/trader failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# -----------------------------
# Data helpers for dashboard
# -----------------------------
LOG_DIR = os.getenv("LOG_DIR", "logs")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")
PAYLOADS_DIR = os.path.join(LOG_DIR, "payloads")


def tail_lines(path: str, n: int = 200) -> List[str]:
    if not os.path.exists(path):
        return []
    avg_line = 120
    to_read = n * avg_line
    with open(path, "rb") as f:
        try:
            f.seek(max(f.tell() - to_read, 0), os.SEEK_SET)
        except Exception:
            pass
        data = f.read().decode("utf-8", errors="ignore")
    lines = data.splitlines()
    return lines[-n:]


def _read_trades(limit: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(TRADES_CSV):
        return rows
    try:
        with open(TRADES_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        logging.warning("read trades.csv failed: %s", e)
        return []
    rows = rows[-limit:]
    # Normalize types
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            out.append({
                "timestamp": r.get("timestamp") or r.get("time") or "",
                "symbol": r.get("symbol", ""),
                "side": r.get("side", ""),
                "qty": float(r.get("qty", 0) or 0),
                "entry": float(r.get("entry", 0) or 0),
                "exit": float(r.get("exit", 0) or 0),
                "tp": float(r.get("tp", 0) or 0),
                "sl": float(r.get("sl", 0) or 0),
                "pnl": float(r.get("pnl", 0) or 0),
                "status": r.get("status", "") or ("closed" if (r.get("exit") and float(r.get("exit") or 0) > 0) else "open"),
                "id": r.get("id") or r.get("order_id") or "",
            })
        except Exception:
            continue
    return out


def _last_price(symbol: str) -> Optional[float]:
    # Try orderbook mid; fallback to last close
    try:
        if fetch_orderbook is not None:
            ob = fetch_orderbook(symbol, limit=5)
            if ob and ob.get("bids") and ob.get("asks"):
                bid = float(ob["bids"][0][0])
                ask = float(ob["asks"][0][0])
                return (bid + ask) / 2.0
    except Exception:
        pass
    try:
        if fetch_ohlcv is not None:
            df = fetch_ohlcv(symbol, interval="5m", limit=2)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
    except Exception:
        pass
    return None


def _compute_live_pnl(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_realized = 0.0
    total_unrealized = 0.0
    win = 0
    loss = 0
    sym_prices: Dict[str, float] = {}

    for r in rows:
        side = (r.get("side") or "").lower()
        entry = float(r.get("entry") or 0)
        exitp = float(r.get("exit") or 0)
        qty = float(r.get("qty") or 0)
        symbol = r.get("symbol") or ""
        status = r.get("status") or ""

        if exitp > 0:
            # realized
            pnl = (exitp - entry) * qty if side == "long" else (entry - exitp) * qty
            total_realized += pnl
            if pnl >= 0:
                win += 1
            else:
                loss += 1
        else:
            # unrealized
            if symbol and symbol not in sym_prices:
                lp = _last_price(symbol)
                if lp is not None:
                    sym_prices[symbol] = lp
            lp = sym_prices.get(symbol)
            if lp is not None and qty > 0 and entry > 0:
                upnl = (lp - entry) * qty if side == "long" else (entry - lp) * qty
                total_unrealized += upnl

    total = total_realized + total_unrealized
    trades = win + loss if (win + loss) > 0 else 0
    wr = (win / trades) if trades else 0.0
    return {
        "realized": total_realized,
        "unrealized": total_unrealized,
        "total": total,
        "win_rate": wr,
        "wins": win,
        "losses": loss,
    }


# -----------------------------
# Dashboard data APIs
# -----------------------------
@server.get("/api/overview")
def api_overview() -> Any:
    try:
        if _bn_overview is None:
            return jsonify({"status": "error", "error": "overview unavailable"}), 503
        ov = _bn_overview()
        # Expecting dicts/lists; pass-through
        return jsonify({"status": "ok", "overview": ov})
    except Exception as e:
        logging.exception("/api/overview failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@server.get("/api/trades")
def api_trades() -> Any:
    try:
        limit = int(request.args.get("limit", 200))
        rows = _read_trades(limit=limit)
        pnl = _compute_live_pnl(rows) if rows else {"realized": 0, "unrealized": 0, "total": 0, "win_rate": 0, "wins": 0, "losses": 0}
        return jsonify({"status": "ok", "rows": rows, "pnl": pnl})
    except Exception as e:
        logging.exception("/api/trades failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@server.get("/api/logs")
def api_logs() -> Any:
    try:
        lines = int(request.args.get("lines", 200))
        path = os.path.join(LOG_DIR, "bot.log")
        tail = tail_lines(path, n=lines)
        return jsonify({"status": "ok", "lines": tail})
    except Exception as e:
        logging.exception("/api/logs failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@server.get("/api/signals")
def api_signals() -> Any:
    # Read latest payload preview snapshots if any were dumped to logs/payloads
    try:
        items: List[Dict[str, Any]] = []
        if os.path.isdir(PAYLOADS_DIR):
            candidates = sorted(glob.glob(os.path.join(PAYLOADS_DIR, "**", "*.json")), reverse=True)
            for p in candidates[:100]:
                try:
                    with open(p, encoding="utf-8") as f:
                        j = json.load(f)
                    # Normalize minimal fields if present
                    items.append({
                        "path": p,
                        "symbol": j.get("symbol") or j.get("pair") or j.get("payload", {}).get("pair"),
                        "direction": j.get("direction") or j.get("prediction", {}).get("direction"),
                        "prob": j.get("prob") or j.get("prediction", {}).get("prob"),
                        "ts": j.get("timestamp") or j.get("created_at") or _now_iso(),
                    })
                except Exception:
                    continue
        return jsonify({"status": "ok", "items": items})
    except Exception as e:
        logging.exception("/api/signals failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# -----------------------------
# Dashboard page (Liquid Glass Theme)
# -----------------------------
GLASS_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>withEarth — Liquid Glass Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --glass-bg: rgba(255, 255, 255, 0.10);
      --glass-border: rgba(255, 255, 255, 0.25);
      --glass-shadow: 0 8px 30px rgba(0,0,0,0.20);
    }
    body {
      min-height: 100vh;
      background: radial-gradient(1200px 600px at 10% 10%, #92A8FF33, transparent 60%),
                  radial-gradient(1000px 800px at 90% 20%, #8EF7D733, transparent 60%),
                  linear-gradient(180deg, #0b1220 0%, #060a14 100%);
      color: #eef2ff;
    }
    .glass {
      background: var(--glass-bg);
      backdrop-filter: blur(18px) saturate(140%);
      -webkit-backdrop-filter: blur(18px) saturate(140%);
      border: 1px solid var(--glass-border);
      box-shadow: var(--glass-shadow);
      border-radius: 1.25rem;
    }
    .pill {
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      backdrop-filter: blur(10px);
      border-radius: 9999px;
      padding: 0.25rem 0.75rem;
    }
    .glow {
      box-shadow: 0 0 0 2px rgba(255,255,255,0.06), 0 10px 40px rgba(37, 99, 235, 0.25);
    }
    .divider {
      height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
    }
    .grid-auto {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1rem;
    }
    .table-wrap { overflow: auto; }
    .table { width: 100%; border-collapse: collapse; }
    .table th, .table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.08); white-space: nowrap; }
    .badge { padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; }
  </style>
</head>
<body class="antialiased">
  <div class="max-w-7xl mx-auto px-4 py-8">
    <header class="flex items-center justify-between mb-6">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-2xl glass flex items-center justify-center glow">
          <span class="text-xl">🪩</span>
        </div>
        <div>
          <h1 class="text-2xl font-semibold tracking-tight">withEarth — Auto‑Trade Dashboard</h1>
          <div class="text-xs text-indigo-200/70">Liquid Glass • visionOS‑inspired</div>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span id="statusDot" class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="lastUpdated" class="pill text-xs">업데이트 대기…</span>
        <button id="refreshBtn" class="pill text-sm hover:bg-white/20">즉시 새로고침</button>
        <label class="pill text-sm flex items-center gap-2 cursor-pointer">
          <input id="autoRefresh" type="checkbox" class="accent-indigo-400" checked>
          Auto Refresh (5s)
        </label>
      </div>
    </header>

    <!-- KPI Cards -->
    <section class="grid-auto mb-6">
      <div class="glass p-5">
        <div class="text-sm text-indigo-200/80">총 PnL</div>
        <div id="pnlTotal" class="text-3xl font-semibold mt-1">—</div>
        <div class="text-xs mt-1 text-indigo-200/70">실현: <span id="pnlRealized">—</span> • 미실현: <span id="pnlUnrealized">—</span></div>
      </div>
      <div class="glass p-5">
        <div class="text-sm text-indigo-200/80">승률</div>
        <div id="winRate" class="text-3xl font-semibold mt-1">—</div>
        <div class="text-xs mt-1 text-indigo-200/70">승 <span id="winCount">0</span> / 패 <span id="lossCount">0</span></div>
      </div>
      <div class="glass p-5">
        <div class="text-sm text-indigo-200/80">오버뷰 (계정 요약)</div>
        <div id="balanceUSDT" class="text-3xl font-semibold mt-1">—</div>
        <div class="text-xs mt-1 text-indigo-200/70">마진/포지션/잔고 실시간</div>
      </div>
      <div class="glass p-5">
        <div class="text-sm text-indigo-200/80">시그널 히스토리</div>
        <div id="signalsCount" class="text-3xl font-semibold mt-1">—</div>
        <div class="text-xs mt-1 text-indigo-200/70">최근 저장된 payloads 기반</div>
      </div>
    </section>

    <div class="grid md:grid-cols-3 gap-6">
      <!-- Trades Table -->
      <section class="md:col-span-2 glass p-5">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-lg font-semibold">📊 최근 거래</h2>
          <div class="flex items-center gap-2 text-sm">
            <span class="pill">최대 200건</span>
          </div>
        </div>
        <div class="table-wrap">
          <table class="table text-sm">
            <thead class="text-indigo-200/70">
              <tr>
                <th>시간(UTC)</th>
                <th>심볼</th>
                <th>사이드</th>
                <th>수량</th>
                <th>진입</th>
                <th>청산</th>
                <th>TP</th>
                <th>SL</th>
                <th>PNL</th>
                <th>상태</th>
              </tr>
            </thead>
            <tbody id="tradesBody"></tbody>
          </table>
        </div>
      </section>

      <!-- Logs & Mini Chart -->
      <section class="glass p-5">
        <h2 class="text-lg font-semibold mb-3">🪵 로그 (최근 200줄)</h2>
        <pre id="logBox" class="text-xs whitespace-pre-wrap h-72 overflow-auto bg-black/20 p-3 rounded-xl"></pre>
        <div class="divider my-4"></div>
        <h3 class="text-sm font-medium mb-2">수익 곡선</h3>
        <canvas id="pnlChart" height="140"></canvas>
      </section>
    </div>
  </div>

  <script>
    const fmt = new Intl.NumberFormat('en-US', {maximumFractionDigits: 6});
    const fmt2 = new Intl.NumberFormat('en-US', {maximumFractionDigits: 2});

    async function fetchJSON(url) {
      const r = await fetch(url);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.json();
    }

    function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

    function renderTrades(rows) {
      const tbody = document.getElementById('tradesBody');
      if (!tbody) return;
      tbody.innerHTML = '';
      rows.forEach(r => {
        const tr = document.createElement('tr');
        const badgeSide = r.side === 'long' ? 'bg-emerald-500/20 text-emerald-300' : (r.side === 'short' ? 'bg-rose-500/20 text-rose-300' : 'bg-slate-500/20 text-slate-300');
        const badgeStatus = r.status === 'open' ? 'bg-amber-500/20 text-amber-300' : 'bg-indigo-500/20 text-indigo-300';
        tr.innerHTML = `
          <td class="text-indigo-100/90">${r.timestamp ?? ''}</td>
          <td>${r.symbol ?? ''}</td>
          <td><span class="badge ${badgeSide}">${r.side ?? ''}</span></td>
          <td>${fmt.format(r.qty ?? 0)}</td>
          <td>${fmt.format(r.entry ?? 0)}</td>
          <td>${r.exit ? fmt.format(r.exit) : '—'}</td>
          <td>${r.tp ? fmt.format(r.tp) : '—'}</td>
          <td>${r.sl ? fmt.format(r.sl) : '—'}</td>
          <td class="${(r.pnl ?? 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}">${r.pnl ? fmt2.format(r.pnl) : '—'}</td>
          <td><span class="badge ${badgeStatus}">${r.status ?? ''}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    let pnlChart;
    function renderPnlChart(rows) {
      const ctx = document.getElementById('pnlChart');
      if (!ctx) return;
      // Build equity curve: realized cumulative; include unrealized as last point shadow
      let cum = 0; const xs = []; const ys = [];
      rows.forEach(r => {
        const realized = (r.exit && r.entry && r.qty) ? ((r.side === 'long' ? (r.exit - r.entry) : (r.entry - r.exit)) * r.qty) : 0;
        cum += realized;
        xs.push(r.timestamp || '');
        ys.push(cum);
      });
      if (pnlChart) pnlChart.destroy();
      pnlChart = new Chart(ctx, {
        type: 'line',
        data: { labels: xs, datasets: [{ label: 'Realized PnL', data: ys, fill: false }] },
        options: { responsive: true, plugins: { legend: { labels: { color: '#c7d2fe' } } }, scales: { x: { ticks: { color: '#c7d2fe' } }, y: { ticks: { color: '#c7d2fe' } } } }
      });
    }

    async function refreshAll() {
      document.getElementById('statusDot').classList.remove('bg-rose-400');
      document.getElementById('statusDot').classList.add('bg-emerald-400');
      setText('lastUpdated', new Date().toLocaleString());

      try {
        const [trades, logs, overview, signals] = await Promise.all([
          fetchJSON('/api/trades?limit=200'),
          fetchJSON('/api/logs?lines=200'),
          fetchJSON('/api/overview'),
          fetchJSON('/api/signals')
        ]);

        // Trades
        if (trades.status === 'ok') {
          renderTrades(trades.rows || []);
          renderPnlChart(trades.rows || []);
          const p = trades.pnl || {realized:0, unrealized:0, total:0, win_rate:0, wins:0, losses:0};
          setText('pnlRealized', fmt2.format(p.realized));
          setText('pnlUnrealized', fmt2.format(p.unrealized));
          setText('pnlTotal', fmt2.format(p.total));
          setText('winRate', (p.win_rate*100).toFixed(1) + '%');
          setText('winCount', p.wins);
          setText('lossCount', p.losses);
        }

        // Logs
        if (logs.status === 'ok') {
          document.getElementById('logBox').textContent = (logs.lines || []).join('\n');
        }

        // Overview
        if (overview.status === 'ok') {
          const ov = overview.overview || {}; // expects {balances:[...], positions:[...]} etc.
          const bal = (() => {
            try {
              const b = (ov.balances || []).find(x => x and (x.asset === 'USDT' || x['asset'] === 'USDT'));
              if (b) return parseFloat(b.balance || b['balance'] || 0);
            } catch (e) {}
            return '—';
          })();
          setText('balanceUSDT', typeof bal === 'number' ? fmt2.format(bal) + ' USDT' : bal);
        }

        // Signals
        if (signals.status === 'ok') {
          const cnt = (signals.items || []).length;
          setText('signalsCount', cnt);
        }
      } catch (e) {
        document.getElementById('statusDot').classList.remove('bg-emerald-400');
        document.getElementById('statusDot').classList.add('bg-rose-400');
        console.error(e);
      }
    }

    document.getElementById('refreshBtn').addEventListener('click', refreshAll);
    const auto = document.getElementById('autoRefresh');
    let timer = setInterval(refreshAll, 5000);
    auto.addEventListener('change', () => {
      if (auto.checked) { timer = setInterval(refreshAll, 5000); }
      else { clearInterval(timer); }
    });

    refreshAll();
  </script>
</body>
</html>
"""

# +++ ADD BELOW THIS LINE IN app.py +++
from flask import redirect

@server.get("/")
def root_redirect():
    # 브라우저가 루트(/)로 접속하면 대시보드로 바로 이동
    return redirect("/dashboard", code=302)

@server.errorhandler(404)
def handle_404(e):
    # 브라우저(html 요청)면 대시보드로 보내고, 그 외(API 등)는 JSON 404 유지
    accept = (request.headers.get("Accept") or "").lower()
    if "text/html" in accept:
        return redirect("/dashboard", code=302)
    return jsonify({"status": "error", "error": "not_found", "path": request.path}), 404
# +++ END ADD +++


@server.get("/dashboard")
def dashboard() -> Response:
    return Response(GLASS_HTML, mimetype="text/html")


# -----------------------------
# INTEGRATION HINTS: trade journaling
# -----------------------------
"""
To populate logs/trades.csv automatically, add a line inside helpers/signals.py -> manage_trade()
when an order is PLACED or CLOSED. Example snippet (append after order filled or immediately after placing bracket):

from datetime import datetime, timezone
import csv, os

LOG_DIR = os.getenv("LOG_DIR", "logs")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")
os.makedirs(LOG_DIR, exist_ok=True)

headers = ["timestamp", "symbol", "side", "qty", "entry", "tp", "sl", "exit", "pnl", "status", "id"]
new_row = {
    "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    "symbol": symbol,
    "side": direction,   # "long" | "short"
    "qty": qty,
    "entry": entry,
    "tp": tp,
    "sl": sl,
    "exit": 0.0,
    "pnl": 0.0,
    "status": "open",
    "id": str(order_id_or_client_order_id),
}

exists = os.path.exists(TRADES_CSV)
with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    if not exists:
        w.writeheader()
    w.writerow(new_row)

# On close: write a new row with exit/pnl/status=closed (or maintain a DB/CSV update logic).
"""


# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    server.run(host="0.0.0.0", port=port, debug=True)
