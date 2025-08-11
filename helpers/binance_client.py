from __future__ import annotations

"""
helpers/binance_client.py – refactor #1 (2025-08-11, KST)

Target SDK: binance-sdk-derivatives-trading-usds-futures==1.0.0
Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
Repo path for this SDK family: binance-connector-python/clients/derivatives_trading_usds_futures

This module wraps the USDⓈ-M Futures REST API with a thin, robust layer that:
- Initializes the official modular SDK with prod/testnet switching
- Normalizes common trade ops (cancel-all, position mode, margin type, leverage)
- Places market/limit and basic "bracket" (TP/SL reduce-only) orders
- Exposes utility helpers used by signals.py

Design notes
- The SDK is auto-generated; method names occasionally differ across minor versions.
  We defensively try multiple candidate method names per operation (see _call and
  _pick for implementation). This keeps compatibility across 1.0.0 … 1.x.
- All prices/qty are quantized to symbol filters (tickSize/stepSize/minNotional).
- Logging is friendly to the existing log style seen in your traces.
"""

import logging
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---- Optional utils integration ------------------------------------------------
try:  # Prefer project utils if available
    from .utils import get_secret, LOG_DIR  # type: ignore
except Exception:  # pragma: no cover
    def get_secret(name: str) -> Optional[str]:  # minimal fallback
        return os.getenv(name)

    LOG_DIR = os.path.join(os.getcwd(), "logs")

# ---- Binance SDK (modular) -----------------------------------------------------
# pip install binance-sdk-derivatives-trading-usds-futures==1.0.0
from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

# --------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================
# Client bootstrap & utilities
# =============================
_api_key = get_secret("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY")
_api_secret = get_secret("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET")
if not _api_key or not _api_secret:
    logger.warning("Binance API key/secret not provided; running in unauthenticated mode.")

_USE_TESTNET = os.getenv("BINANCE_FUTURES_TESTNET", "").lower() in ("1", "true", "yes")
_BASE_PATH = (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
    if _USE_TESTNET
    else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
)

_TIMEOUT_MS = int(os.getenv("BINANCE_HTTP_TIMEOUT_MS", "10000"))
_RETRIES = int(os.getenv("BINANCE_HTTP_RETRIES", "3"))
_BACKOFF_MS = int(os.getenv("BINANCE_HTTP_BACKOFF_MS", "1000"))

_config = ConfigurationRestAPI(
    api_key=_api_key or "",
    api_secret=_api_secret or "",
    base_path=_BASE_PATH,
    timeout=_TIMEOUT_MS,
    retries=_RETRIES,
    backoff=_BACKOFF_MS,
)
_client: Optional[DerivativesTradingUsdsFutures] = None


def _get_client() -> DerivativesTradingUsdsFutures:
    global _client
    if _client is None:
        _client = DerivativesTradingUsdsFutures(config_rest_api=_config)
        logger.info(
            "Binance config initialized: base_path=%s, api_key_set=%s",
            _BASE_PATH,
            bool(_api_key),
        )
        logger.info("Binance client initialized successfully.")
    return _client



def get_client() -> DerivativesTradingUsdsFutures:
    """Public accessor for other modules (e.g., data_fetch)."""
    return _get_client()


# ===============
# Call helpers
# ===============

def _pick(obj: Any, names: List[str]) -> Optional[Any]:
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn
    return None


def _call(obj: Any, cand_names: List[str], /, **kwargs):
    fn = _pick(obj, cand_names)
    if not fn:
        raise AttributeError(f"None of {cand_names} available on {type(obj).__name__}")
    return fn(**kwargs)


# ==================================
# Exchange info + symbol filter util
# ==================================
_symbol_filters_cache: Dict[str, Dict[str, Any]] = {}


def _to_decimal(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    # floor to step (Binance rejects rounding up)
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step


def load_symbol_filters(symbol: str) -> Dict[str, Any]:
    """Return {tickSize, stepSize, minQty, maxQty, minNotional, raw} with cache."""
    symbol = symbol.upper()
    if symbol in _symbol_filters_cache:
        return _symbol_filters_cache[symbol]

    client = _get_client()
    # exchange_information covers filters
    resp = _call(client.rest_api, ["exchange_information", "exchangeInformation"])  # SDK naming guard
    data = resp.data() if hasattr(resp, "data") else resp
    # data.symbols is usually a list of dicts
    symbols = getattr(data, "symbols", None) or data.get("symbols", [])
    sym = next((s for s in symbols if (s.get("symbol") or s.get("symbolName")) == symbol), None)
    if not sym:
        raise ValueError(f"Symbol {symbol} not found in exchange info")

    filters_list = sym.get("filters", [])
    fmap = {f.get("filterType"): f for f in filters_list}

    tick = _to_decimal((fmap.get("PRICE_FILTER") or {}).get("tickSize", "0.01"))
    step = _to_decimal((fmap.get("LOT_SIZE") or {}).get("stepSize", "0.001"))
    min_qty = _to_decimal((fmap.get("LOT_SIZE") or {}).get("minQty", "0.0"))
    max_qty = _to_decimal((fmap.get("LOT_SIZE") or {}).get("maxQty", "0.0"))
    min_notional = _to_decimal((fmap.get("NOTIONAL") or {}).get("notional", "5"))

    result = {
        "tickSize": tick,
        "stepSize": step,
        "minQty": min_qty,
        "maxQty": max_qty,
        "minNotional": min_notional,
        "raw": sym,
    }
    _symbol_filters_cache[symbol] = result
    return result


def ensure_min_notional(symbol: str, qty: float, price: float, filters: Optional[Dict[str, Any]] = None) -> float:
    """Return a quantity meeting minNotional and stepSize requirements."""
    filters = filters or load_symbol_filters(symbol)
    step: Decimal = filters["stepSize"]
    min_notional: Decimal = filters["minNotional"]

    q = _to_decimal(qty)
    p = _to_decimal(price)
    # enforce step and notional
    q = round_to_step(q, step)
    if (q * p) < min_notional:
        # bump to meet min notional
        need = (min_notional / max(p, Decimal("1e-12")))
        q = round_to_step(need, step)
    return float(max(q, step))


def normalize_price_with_mode(symbol: str, price: float) -> float:
    filters = load_symbol_filters(symbol)
    tick: Decimal = filters["tickSize"]
    return float(round_to_step(_to_decimal(price), tick))


# ==========================
# Account & overview helpers
# ==========================

def _df_safe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    try:
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def get_overview() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (balances_df, positions_df) for the dashboard.
    We try V3 → V2 → position risk; fall back to empty frames if all fail.
    """
    client = _get_client()

    # balances
    balances: List[Dict[str, Any]] = []
    try:
        # account_information may contain assets; some SDKs split it
        acct = _call(client.rest_api, ["account_information_v3", "account_information_v2", "account_information", "futures_account_balance"])  # naming guard
        data = acct.data() if hasattr(acct, "data") else acct
        assets = getattr(data, "assets", None) or data.get("assets", []) or getattr(data, "balance", None) or data.get("balance", [])
        for a in assets:
            balances.append({
                "asset": a.get("asset") or a.get("assetName"),
                "balance": float(a.get("balance") or a.get("walletBalance") or 0.0),
                "unrealizedPnL": float(a.get("unrealizedProfit", 0.0) or a.get("unrealizedPnL", 0.0)),
            })
    except Exception as e:
        logger.info("overview balances fetch failed: %s", e)

    # positions
    positions: List[Dict[str, Any]] = []
    try:
        pos = _call(client.rest_api, [
            "position_information_v3",
            "position_information_v2",
            "position_information",
            "position_risk",
        ])
        pdata = pos.data() if hasattr(pos, "data") else pos
        items = getattr(pdata, "positions", None) or pdata if isinstance(pdata, list) else pdata.get("positions", [])
        for p in items:
            try:
                amt = float(p.get("positionAmt") or p.get("positionAmount") or 0)
                if abs(amt) < 1e-12:
                    continue
                positions.append({
                    "symbol": p.get("symbol"),
                    "positionAmt": amt,
                    "entryPrice": float(p.get("entryPrice") or 0.0),
                    "unRealizedProfit": float(p.get("unRealizedProfit") or p.get("unrealizedPnL") or 0.0),
                    "leverage": float(p.get("leverage") or 0.0),
                    "isolated": bool((p.get("isolated") if isinstance(p.get("isolated"), bool) else str(p.get("isolated")).lower() == "true")),
                })
            except Exception:
                continue
    except Exception as e:
        logger.info("overview positions fetch failed: %s", e)

    return _df_safe(balances), _df_safe(positions)


# ==================
# Account mutators
# ==================

def cancel_open_orders(symbol: str) -> Dict[str, Any]:
    """DELETE /fapi/v1/allOpenOrders"""
    client = _get_client()
    resp = _call(client.rest_api, ["cancel_all_open_orders", "cancelAllOpenOrders"], symbol=symbol)
    data = resp.data() if hasattr(resp, "data") else resp
    msg = (data.get("msg") if isinstance(data, dict) else getattr(data, "msg", "")) or ""
    code = (data.get("code") if isinstance(data, dict) else getattr(data, "code", None))
    logger.info("Bulk-cancel %s: code=%s msg=%r additional_properties=%s", symbol, code or "", msg, getattr(resp, "additional_properties", {}))
    return data if isinstance(data, dict) else {"code": code, "msg": msg}


def set_position_mode(mode: str = "ONEWAY") -> None:
    """POST /fapi/v1/positionSide/dual  dualSidePosition=true → HEDGE, false → ONEWAY"""
    client = _get_client()
    mode = (mode or "").upper()
    dual = True if mode == "HEDGE" else False
    try:
        _call(client.rest_api, ["change_position_mode", "changePositionMode", "change_position_side_dual"], dual_side_position=str(dual).lower())
        logger.info("Position mode set to %s", mode)
    except Exception as e:
        logger.error("Position mode set error: %s", getattr(e, "message", str(e)))


def set_margin_type(symbol: str, margin_type: str = "ISOLATED") -> None:
    """POST /fapi/v1/marginType"""
    client = _get_client()
    mt = (margin_type or "").upper()
    try:
        _call(client.rest_api, ["change_margin_type", "changeMarginType"], symbol=symbol, margin_type=mt)
        logger.info("Margin type set to %s for %s", mt, symbol)
    except Exception as e:
        logger.error("Margin type set error: %s", getattr(e, "message", str(e)))


def set_leverage(symbol: str, leverage: int) -> None:
    """POST /fapi/v1/leverage"""
    client = _get_client()
    try:
        resp = _call(client.rest_api, ["change_initial_leverage", "changeInitialLeverage"], symbol=symbol, leverage=int(leverage))
        data = resp.data() if hasattr(resp, "data") else resp
        logger.info("Leverage set to %sx for %s: %s", leverage, symbol, data)
    except Exception as e:
        logger.error("Leverage set error: %s", getattr(e, "message", str(e)))


# ==================
# Order placement
# ==================

def _position_side() -> str:
    # For ONEWAY we can omit, but keeping BOTH is safe.
    return os.getenv("POSITION_SIDE", "BOTH").upper()


def place_market_order(symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
    client = _get_client()
    side = side.upper()
    ps = _position_side()
    filters = load_symbol_filters(symbol)
    qty = ensure_min_notional(symbol, quantity, price=_to_decimal(1.0), filters=filters)  # notional checked later with real price

    try:
        resp = _call(
            client.rest_api,
            ["new_order", "newOrder"],
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=str(qty),
            reduce_only=reduce_only,
            position_side=ps,
            new_order_resp_type="RESULT",
        )
        return resp.data() if hasattr(resp, "data") else resp
    except Exception as e:
        logger.exception("place_market_order failed: %s", e)
        raise


def place_limit_order(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    time_in_force: str = "GTC",
    reduce_only: bool = False,
) -> Dict[str, Any]:
    client = _get_client()
    side = side.upper()
    ps = _position_side()

    f = load_symbol_filters(symbol)
    price_q = normalize_price_with_mode(symbol, price)
    qty_q = ensure_min_notional(symbol, quantity, price_q, f)

    try:
        resp = _call(
            client.rest_api,
            ["new_order", "newOrder"],
            symbol=symbol,
            side=side,
            type="LIMIT",
            time_in_force=time_in_force,
            price=str(price_q),
            quantity=str(qty_q),
            reduce_only=reduce_only,
            position_side=ps,
        )
        return resp.data() if hasattr(resp, "data") else resp
    except Exception as e:
        logger.exception("place_limit_order failed: %s", e)
        raise


def place_bracket_orders(symbol: str, side: str, quantity: float, take_profit: float, stop_loss: float) -> Dict[str, Any]:
    """Create reduce-only TP and SL market orders (independent)."""
    client = _get_client()
    side = side.upper()
    ps = _position_side()

    # TP is opposite side, SL is opposite side as well (reduce-only)
    opp = "SELL" if side == "BUY" else "BUY"

    tp_price = normalize_price_with_mode(symbol, take_profit)
    sl_price = normalize_price_with_mode(symbol, stop_loss)

    f = load_symbol_filters(symbol)
    qty_q = ensure_min_notional(symbol, quantity, price=tp_price, filters=f)

    out: Dict[str, Any] = {"take_profit": None, "stop_loss": None}
    # TAKE_PROFIT_MARKET
    try:
        tp_resp = _call(
            client.rest_api,
            ["new_order", "newOrder"],
            symbol=symbol,
            side=opp,
            type="TAKE_PROFIT_MARKET",
            stop_price=str(tp_price),
            close_position=False,
            reduce_only=True,
            position_side=ps,
            quantity=str(qty_q),
        )
        out["take_profit"] = tp_resp.data() if hasattr(tp_resp, "data") else tp_resp
    except Exception as e:
        logger.info("place_bracket_orders TP failed: %s", e)

    # STOP_MARKET
    try:
        sl_resp = _call(
            client.rest_api,
            ["new_order", "newOrder"],
            symbol=symbol,
            side=opp,
            type="STOP_MARKET",
            stop_price=str(sl_price),
            close_position=False,
            reduce_only=True,
            position_side=ps,
            quantity=str(qty_q),
        )
        out["stop_loss"] = sl_resp.data() if hasattr(sl_resp, "data") else sl_resp
    except Exception as e:
        logger.info("place_bracket_orders SL failed: %s", e)

    return out


def build_entry_and_brackets(
    symbol: str,
    side: str,
    quantity: float,
    target_price: float,
    stop_price: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    entry = place_market_order(symbol, side, quantity)
    bracket = place_bracket_orders(symbol, side, quantity, target_price, stop_price)
    return entry, bracket


# ==================
# Position helpers
# ==================

def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    client = _get_client()
    try:
        pos = _call(client.rest_api, [
            "position_information_v3",
            "position_information_v2",
            "position_information",
            "position_risk",
        ], symbol=symbol)
        pdata = pos.data() if hasattr(pos, "data") else pos
        items = getattr(pdata, "positions", None) or (pdata if isinstance(pdata, list) else pdata.get("positions", []))
        for p in items:
            if p.get("symbol") == symbol:
                return p
        return None
    except Exception as e:
        logger.info("get_position failed: %s", e)
        return None


__all__ = [
    "get_client",
    # overview
    "get_overview",
    # filters & utils
    "load_symbol_filters",
    "ensure_min_notional",
    "normalize_price_with_mode",
    # account mutators
    "cancel_open_orders",
    "set_position_mode",
    "set_margin_type",
    "set_leverage",
    # orders
    "place_market_order",
    "place_limit_order",
    "place_bracket_orders",
    "build_entry_and_brackets",
    # readbacks
    "get_position",
]
