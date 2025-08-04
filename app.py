"""
Dash dashboard for trading_bot – robust version
- Wallet/Positions table
- Candlestick with SMA20
- Manual prediction & (optionally) manual trade
- /tasks/trader endpoint for App Engine Cron (GET/POST both)
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from flask import jsonify, request

# --- Project helpers ---
from helpers.utils import TZ
from helpers.data_fetch import fetch_ohlcv
from helpers.binance_client import load_symbol_filters
from helpers.signals import get_overview, generate_signal, manage_trade

# -------- Config --------
SYMBOLS = [s.strip().upper() for s in os.getenv("DASH_SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
DEFAULT_SYMBOL = SYMBOLS[0] if SYMBOLS else "BTCUSDT"
DEFAULT_INTERVAL = os.getenv("DASH_INTERVAL", "5m")
ALLOW_MANUAL_TRADE = os.getenv("ALLOW_MANUAL_TRADE", "").lower() == "true"
MODEL_BALANCE = float(os.getenv("MODEL_BALANCE", "1000"))
REFRESH_MS = int(os.getenv("REFRESH_MS", "15000"))

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------- App / Server --------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app: Dash = Dash(__name__, external_stylesheets=external_stylesheets, title="ToTheMoon Dashboard")
server = app.server  # Flask

# -------- Flask routes (health & cron) --------
@server.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "tz": TZ, "symbols": SYMBOLS})

@server.route("/tasks/trader", methods=["GET", "POST"])
def tasks_trader():
    """
    Called by App Engine cron.yaml (GET by default), or manual POST
    Body(optional, POST): {"symbols": ["BTCUSDT","ETHUSDT"]}
    """
    try:
        body = request.get_json(silent=True) or {}
        symbols: List[str] = body.get("symbols") or SYMBOLS
        results = []
        for sym in symbols:
            try:
                res = manage_trade(sym)
                # 변경: 직렬화 보강 - dict만 유지, 비JSON 요소 문자열화
                if isinstance(res, dict):
                    clean_res = {k: str(v) if not isinstance(v, (int, float, str, list, dict, type(None))) else v for k, v in res.items()}
                    results.append({"symbol": sym, "result": clean_res})
                else:
                    results.append({"symbol": sym, "result": str(res)})
            except Exception as exc:
                logging.error(f"auto_trade error {sym}: {exc}")
                results.append({"symbol": sym, "error": str(exc)})
        return jsonify({"status": "ok", "results": results})
    except Exception as exc:
        logging.exception("/tasks/trader failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

# -------- UI helpers --------
def _metric_card(title: str, value: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([html.Div(title, className="text-muted"), html.H4(value, className=f"text-{color}", style={"margin": 0})]),
        className="mb-2",
    )

def _wallet_card(balance_df: Optional[pd.DataFrame]) -> dbc.Card:
    usdt = 0.0
    try:
        if isinstance(balance_df, pd.DataFrame) and not balance_df.empty:
            df = balance_df
            if "asset" in df.columns:
                row = df[df["asset"] == "USDT"]
                if not row.empty:
                    for col in ("availableBalance", "walletBalance", "balance", "crossWalletBalance"):
                        if col in row.columns:
                            usdt = float(row[col].iloc[0]); break
    except Exception:
        logging.exception("wallet parse failed")
    return _metric_card("Available (USDT)", f"{usdt:,.2f} USDT", color="primary")

def _positions_table(positions_df: Optional[pd.DataFrame]) -> dash_table.DataTable:
    if not isinstance(positions_df, pd.DataFrame) or positions_df.empty:
        data, columns = [], []
    else:
        show = [c for c in ("symbol","positionAmt","entryPrice","markPrice","leverage","unrealizedProfit","roe_pct") if c in positions_df.columns]
        df = positions_df[show].copy()
        for c in ("positionAmt","entryPrice","markPrice","unrealizedProfit","roe_pct"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        data = df.to_dict("records"); columns = [{"name": c, "id": c, "type": "numeric"} for c in df.columns]
    return dash_table.DataTable(
        data=data, columns=columns,
        style_table={"height": "340px", "overflowY": "auto"},
        style_cell={"fontSize": 13, "textAlign": "right"},
        page_size=20,
    )

def _candlestick(df: Optional[pd.DataFrame], symbol: str, interval: str):
    import plotly.graph_objs as go
    fig = go.Figure()
    if isinstance(df, pd.DataFrame) and not df.empty:
        if "timestamp" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        sma20 = None
        try: sma20 = df["close"].rolling(20).mean()
        except Exception: pass
        fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=f"{symbol} {interval}"))
        if sma20 is not None: fig.add_trace(go.Scatter(x=df["timestamp"], y=sma20, mode="lines", name="SMA20"))
        fig.update_layout(margin=dict(l=30, r=20, t=30, b=30), height=420)
    else:
        fig.update_layout(title="No data", margin=dict(l=30, r=20, t=30, b=30), height=420)
    return fig

# -------- Layout --------
controls = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Symbol"), dcc.Dropdown(id="symbol", options=[{"label": s, "value": s} for s in SYMBOLS], value=DEFAULT_SYMBOL, clearable=False)], md=3),
                    dbc.Col([dbc.Label("Interval"), dcc.Dropdown(id="interval", options=[{"label": l, "value": v} for l, v in [("5m","5m"),("15m","15m"),("1h","1h"),("4h","4h"),("1d","1d")]], value=DEFAULT_INTERVAL, clearable=False)], md=3),
                    dbc.Col([dbc.Label("Auto refresh (sec)"), dcc.Slider(id="refresh_sec", min=5, max=60, step=5, value=max(5, min(60, REFRESH_MS // 1000)), tooltip={"always_visible": True})], md=6),
                ], align="center",
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Predict (Gemini)", id="btn-predict", n_clicks=0, color="secondary"), md="auto"),
                    dbc.Col(dbc.Button("Trade Now", id="btn-trade", n_clicks=0, color="danger", disabled=not ALLOW_MANUAL_TRADE, title="Set ALLOW_MANUAL_TRADE=true to enable"), md="auto"),
                    dbc.Col(html.Div(f"TZ: {TZ}"), md="auto", className="text-muted"),
                ], align="center", className="g-2",
            ),
        ]
    ), className="mb-3",
)

app.layout = dbc.Container(
    [
        html.H3("ToTheMoon – Trading Dashboard", className="mt-3"),
        controls,
        dcc.Interval(id="timer", interval=REFRESH_MS, n_intervals=0),
        dbc.Row(
            [
                dbc.Col([html.Div(id="wallet-card"), html.Hr(), html.H6("Positions"), html.Div(id="pos-table")], md=4),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id="candlestick-chart", config={"displayModeBar": False})),
                        html.Br(),
                        html.H6("Last Prediction"),
                        dash_table.DataTable(id="pred-table", style_cell={"fontSize": 13}),
                        html.Div(id="pred-note", className="text-muted", style={"fontSize": 12, "marginTop": "6px"}),
                        html.Br(),
                        html.H6("Last Orders (Entry/TP/SL)"),
                        dash_table.DataTable(id="order-table", style_cell={"fontSize": 12, "whiteSpace": "pre-wrap"}, style_table={"height": "180px", "overflowY": "auto"}),
                    ], md=8,
                ),
            ]
        ),
    ], fluid=True,
)

# -------- Helpers --------
def _safe_fetch_overview() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        bal_df, pos_df = get_overview(); return bal_df, pos_df
    except Exception as exc:
        logging.exception(f"get_overview failed: {exc}"); return None, None

def _safe_fetch_ohlcv(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        return fetch_ohlcv(symbol=symbol, interval=interval, ohlcv_limit=250)
    except TypeError:
        try: return fetch_ohlcv(symbol, interval)
        except Exception as exc:
            logging.exception(f"fetch_ohlcv fallback failed: {exc}"); return None
    except Exception as exc:
        logging.exception(f"fetch_ohlcv failed: {exc}"); return None

# -------- Callbacks --------
@app.callback(
    Output("wallet-card", "children"),
    Output("pos-table", "children"),
    Output("candlestick-chart", "figure"),
    Input("timer", "n_intervals"),
    Input("symbol", "value"),
    Input("interval", "value"),
)
def update_dashboard(n: int, symbol: str, interval: str):
    bal_df, pos_df = _safe_fetch_overview()
    ohlcv = _safe_fetch_ohlcv(symbol or DEFAULT_SYMBOL, interval or DEFAULT_INTERVAL)
    wallet = _wallet_card(bal_df)
    pos_table = _positions_table(pos_df)
    fig = _candlestick(ohlcv, symbol or DEFAULT_SYMBOL, interval or DEFAULT_INTERVAL)
    return wallet, pos_table, fig

@app.callback(Output("timer", "interval"), Input("refresh_sec", "value"))
def update_interval(refresh_sec: int):
    try:
        sec = int(refresh_sec); return max(5, min(60, sec)) * 1000
    except Exception:
        return REFRESH_MS

@app.callback(
    Output("pred-table", "data"),
    Output("pred-table", "columns"),
    Output("pred-note", "children"),
    Output("order-table", "data"),
    Output("order-table", "columns"),
    Input("btn-predict", "n_clicks"),
    State("symbol", "value"),
    State("interval", "value"),
    prevent_initial_call=True,
)
def do_predict(n_clicks: int, symbol: str, interval: str):
    pred_data: List[Dict[str, Any]] = []
    pred_cols: List[Dict[str, str]] = []
    note = ""
    orders_data: List[Dict[str, Any]] = []
    orders_cols: List[Dict[str, str]] = []
    try:
        sig = generate_signal(symbol=symbol, balance=MODEL_BALANCE)
        if sig.get("action") == "hold":
            pred_data = [sig]
        else:
            pred_data = [{"action": sig.get("action"), "qty": sig.get("qty"), "entry": sig.get("entry"),
                          "tp": sig.get("tp"), "sl": sig.get("sl"), "prob": sig.get("prob"), "risk_scalar": sig.get("risk_scalar")}]
        pred_cols = [{"name": k, "id": k} for k in (pred_data[0].keys() if pred_data else ["info"])]
        note = "Prediction done. (No trade executed)"
        try:
            filters = load_symbol_filters(__import__("helpers").binance_client.client(), symbol)
            note += f" | tickSize={filters.get('tickSize')} stepSize={filters.get('stepSize')}"
        except Exception:
            pass
    except Exception as exc:
        logging.exception(f"prediction failed: {exc}")
        pred_data = [{"error": str(exc)}]; pred_cols = [{"name": "error", "id": "error"}]; note = "Prediction error – see logs."
    orders_cols = [{"name": c, "id": c} for c in ["type", "detail"]]
    return pred_data, pred_cols, note, orders_data, orders_cols

@app.callback(
    Output("order-table", "data"),
    Output("order-table", "columns"),
    Input("btn-trade", "n_clicks"),
    State("symbol", "value"),
    prevent_initial_call=True,
)
def do_trade(n_clicks: int, symbol: str):
    if not ALLOW_MANUAL_TRADE:
        cols = [{"name": "type", "id": "type"}, {"name": "detail", "id": "detail"}]
        return [{"type": "blocked", "detail": "ALLOW_MANUAL_TRADE=false"}], cols
    try:
        res = manage_trade(symbol)
        rows = []
        if isinstance(res, dict):
            for k in ("action","qty","entry","tp","sl","prob","risk_scalar","entry_order","entry_fill","bracket_order"):
                if k in res: rows.append({"type": k, "detail": str(res[k])})
        if not rows: rows = [{"type": "info", "detail": str(res)}]
        cols = [{"name": "type", "id": "type"}, {"name": "detail", "id": "detail"}]
        return rows, cols
    except Exception as exc:
        logging.exception("manual trade failed")
        cols = [{"name": "type", "id": "type"}, {"name": "detail", "id": "detail"}]
        return [{"type": "error", "detail": str(exc)}], cols

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)