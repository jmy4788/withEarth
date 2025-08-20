from __future__ import annotations
"""
helpers/binance_client.py – refactor #3 (2025-08-18, KST)

Target SDK: binance-sdk-derivatives-trading-usds-futures==1.0.0
Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info

핵심
- 모듈형 SDK 래핑(호출명 가드)
- 심볼 필터 기반 가격/수량 정규화(tickSize/stepSize/minNotional)
- 계정 설정(포지션 모드/마진/레버리지)
- 주문: MARKET/LIMIT/브래킷(TP=LIMIT|MARKET, SL=STOP_MARKET)
- **부분 취소**: 개별 주문 취소(cancel_order), 타입별 일괄 취소(cancel_orders_by_type)
- **tickSize 엄격화**: 사이드별 라운딩 + Decimal.quantize(문자열 자리 고정)
- 리밋 실패 시(옵션) 마켓 폴백을 위한 보조
"""

import logging
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---- Optional utils integration ------------------------------------------------
try:
    from .utils import get_secret, log_event  # type: ignore
except Exception:  # pragma: no cover
    def get_secret(name: str) -> Optional[str]:  # 최소 폴백
        return os.getenv(name)
    def log_event(*args, **kwargs):
        pass

# ---- Binance SDK (modular) -----------------------------------------------------
from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================
# Env
# =============================
_api_key = get_secret("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY")
_api_secret = get_secret("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET")
if not _api_key or not _api_secret:
    logger.warning("Binance API key/secret not provided; running in unauthenticated mode.")

_USE_TESTNET = (
    os.getenv("BINANCE_FUTURES_TESTNET", "") or os.getenv("BINANCE_USE_TESTNET", "")
).lower() in ("1", "true", "yes")
_BASE_PATH = (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
    if _USE_TESTNET
    else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
)

_TIMEOUT_MS = int(os.getenv("BINANCE_HTTP_TIMEOUT_MS", "10000"))
_RETRIES = int(os.getenv("BINANCE_HTTP_RETRIES", "3"))
_BACKOFF_MS = int(os.getenv("BINANCE_HTTP_BACKOFF_MS", "1000"))

# TP order type (LIMIT | MARKET)
_TP_ORDER_TYPE = os.getenv("TP_ORDER_TYPE", "LIMIT").strip().upper()
# limit error fallback to market
_LIMIT_FAILOVER_TO_MARKET = str(os.getenv("LIMIT_FAILOVER_TO_MARKET", "true")).lower() in ("1", "true", "yes")
_SL_ORDER_TYPE = os.getenv("SL_ORDER_TYPE", "STOP_MARKET").strip().upper()  # STOP_MARKET | STOP
_SL_LIMIT_SLIPPAGE_BPS = float(os.getenv("SL_LIMIT_SLIPPAGE_BPS", "10.0"))  # STOP_LIMIT 시 리밋 오프셋(bps)

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
            _BASE_PATH, bool(_api_key),
        )
        logger.info("Binance client initialized successfully.")
    return _client


def get_client() -> DerivativesTradingUsdsFutures:
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
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step

def round_up_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_UP)
    return q * step

def load_symbol_filters(symbol: str) -> Dict[str, Any]:
    """심볼 필터: {tickSize, stepSize, minQty, maxQty, minNotional, raw}"""
    symbol = symbol.upper()
    if symbol in _symbol_filters_cache:
        return _symbol_filters_cache[symbol]

    client = _get_client()
    resp = _call(client.rest_api, ["exchange_information", "exchangeInformation"])
    data = resp.data() if hasattr(resp, "data") else resp

    def to_dict(obj):
        if isinstance(obj, dict): return obj
        if hasattr(obj, "model_dump"): return obj.model_dump()
        if hasattr(obj, "dict"): return obj.dict()
        return obj

    def val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    D = to_dict(data)
    symbols = D.get("symbols") or getattr(data, "symbols", None)
    if not symbols:
        raise ValueError("exchange information returned no symbols")

    found = None
    for s in symbols:
        sd = to_dict(s)
        name = sd.get("symbol") or sd.get("symbolName") or val(s, "symbol") or val(s, "symbolName")
        if name == symbol:
            found = s
            break
    if found is None:
        raise ValueError(f"Symbol {symbol} not found in exchange info")

    f_list = to_dict(found).get("filters") or getattr(found, "filters", []) or []
    fmap: Dict[str, Any] = {}
    for f in (f_list or []):
        fd = to_dict(f)
        ftype = fd.get("filterType") or getattr(f, "filterType", None)
        if ftype:
            fmap[ftype] = fd

    price_f = fmap.get("PRICE_FILTER") or {}
    lot_f   = fmap.get("LOT_SIZE") or {}
    not_f = fmap.get("MIN_NOTIONAL") or fmap.get("NOTIONAL") or {}

    tick = _to_decimal(price_f.get("tickSize") or getattr(price_f, "tickSize", "0.01"))
    step = _to_decimal(lot_f.get("stepSize")   or getattr(lot_f,   "stepSize", "0.001"))
    min_qty = _to_decimal(lot_f.get("minQty")  or getattr(lot_f,   "minQty",   "0.0"))
    max_qty = _to_decimal(lot_f.get("maxQty")  or getattr(lot_f,   "maxQty",   "0.0"))
    min_notional = _to_decimal(
    (not_f.get("minNotional") if isinstance(not_f, dict) else getattr(not_f, "minNotional", None))
    or (not_f.get("notional") if isinstance(not_f, dict) else getattr(not_f, "notional", None))
    or "5")
    
    result = {
        "tickSize": tick,
        "stepSize": step,
        "minQty": min_qty,
        "maxQty": max_qty,
        "minNotional": min_notional,
        "raw": to_dict(found),
    }
    _symbol_filters_cache[symbol] = result
    return result

def _format_to_tick_str(symbol: str, price: float) -> str:
    """Decimal.quantize로 tick 자리 고정 문자열 반환"""
    f = load_symbol_filters(symbol)
    tick: Decimal = f["tickSize"]
    p = _to_decimal(price).quantize(tick, rounding=ROUND_DOWN)
    # 문자열 끝 0/소수점 유지(서버 파싱에 안전)
    return format(p, 'f')

def ensure_min_notional(symbol: str, qty: float, price: float, filters: Optional[Dict[str, Any]] = None) -> float:
    filters = filters or load_symbol_filters(symbol)
    step: Decimal = filters["stepSize"]
    min_notional: Decimal = filters["minNotional"]

    q = _to_decimal(qty)
    p = _to_decimal(price)
    q = round_to_step(q, step)
    if (q * p) < min_notional:
        need = (min_notional / max(p, Decimal("1e-12")))
        q = round_to_step(need, step)
    return float(max(q, step))

def normalize_price_with_mode(symbol: str, price: float) -> float:
    """(호환) tickSize 내림"""
    filters = load_symbol_filters(symbol)
    tick: Decimal = filters["tickSize"]
    return float(round_to_step(_to_decimal(price), tick))

def normalize_price_for_side(symbol: str, price: float, side: str) -> float:
    """BUY는 내림(더 보수적), SELL은 올림(증분 미충족 방지)"""
    filters = load_symbol_filters(symbol)
    tick: Decimal = filters["tickSize"]
    v = _to_decimal(price)
    if str(side).upper() == "SELL":
        vq = round_up_to_step(v, tick)
    else:
        vq = round_to_step(v, tick)
    return float(vq)

# ==========================
# Account & overview helpers
# ==========================
def _df_safe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    try:
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def get_overview() -> Dict[str, List[Dict[str, Any]]]:
    client = _get_client()
    balances: List[Dict[str, Any]] = []
    try:
        acct = _call(
            client.rest_api,
            ["account_information_v3", "account_information_v2", "account_information", "futures_account_balance"],
        )
        data = acct.data() if hasattr(acct, "data") else acct
        assets = (
            getattr(data, "assets", None)
            or (data.get("assets", []) if isinstance(data, dict) else None)
            or getattr(data, "balance", None)
            or (data.get("balance", []) if isinstance(data, dict) else [])
        )
        for a in assets or []:
            try:
                balances.append({
                    "asset": a.get("asset") or a.get("assetName"),
                    "balance": float(a.get("balance") or a.get("walletBalance") or 0.0),
                    "unrealizedPnL": float(a.get("unrealizedProfit", 0.0) or a.get("unrealizedPnL", 0.0)),
                })
            except Exception:
                continue
    except Exception as e:
        logger.info("overview balances fetch failed: %s", e)

    positions: List[Dict[str, Any]] = []
    try:
        pos = _call(
            client.rest_api,
            ["position_information_v3", "position_information_v2", "position_information", "position_risk"],
        )
        pdata = pos.data() if hasattr(pos, "data") else pos
        items = (
            getattr(pdata, "positions", None)
            or (pdata if isinstance(pdata, list) else pdata.get("positions", []))
        )
        for p in items or []:
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
                    "isolated": bool( p.get("isolated") if isinstance(p.get("isolated"), bool)
                                      else str(p.get("isolated")).lower() == "true"),
                })
            except Exception:
                continue
    except Exception as e:
        logger.info("overview positions fetch failed: %s", e)

    return {"balances": balances, "positions": positions}

# ==================
# Account mutators
# ==================
def cancel_open_orders(symbol: str) -> Dict[str, Any]:
    client = _get_client()
    resp = _call(client.rest_api, ["cancel_all_open_orders", "cancelAllOpenOrders"], symbol=symbol)
    data = resp.data() if hasattr(resp, "data") else resp
    msg = (data.get("msg") if isinstance(data, dict) else getattr(data, "msg", "")) or ""
    code = (data.get("code") if isinstance(data, dict) else getattr(data, "code", None))
    logger.info("Bulk-cancel %s: code=%s msg=%r additional_properties=%s", symbol, code or "", msg, getattr(resp, "additional_properties", {}))
    return data if isinstance(data, dict) else {"code": code, "msg": msg}

def cancel_order(symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
    """개별 주문 취소"""
    client = _get_client()
    kwargs: Dict[str, Any] = {"symbol": symbol}
    if order_id is not None:
        kwargs["order_id"] = int(order_id)
    if orig_client_order_id:
        kwargs["orig_client_order_id"] = orig_client_order_id
    try:
        resp = _call(client.rest_api,
                     ["cancel_order", "cancelOrder", "cancel_specific_order", "cancelSpecificOrder"],
                     **kwargs)
        data = resp.data() if hasattr(resp, "data") else resp
        return data if isinstance(data, dict) else {"raw": data}
    except Exception as e:
        logger.info("cancel_order failed for %s: %s", symbol, e)
        return {}

def cancel_orders_by_type(symbol: str, types: List[str]) -> int:
    """열린 주문 중 특정 타입(TAKE_PROFIT, STOP, STOP_MARKET 등)만 취소"""
    typesU = {t.upper() for t in types}
    cnt = 0
    try:
        orders = get_open_orders(symbol)
    except Exception:
        orders = []
    for o in orders or []:
        t = str((o.get("type") if isinstance(o, dict) else getattr(o, "type", "")) or "").upper()
        oid = o.get("orderId") if isinstance(o, dict) else getattr(o, "orderId", None)
        if t in typesU and oid is not None:
            try:
                cancel_order(symbol, order_id=int(oid))
                cnt += 1
            except Exception:
                pass
    return cnt

def set_position_mode(mode: str = "ONEWAY") -> None:
    client = _get_client()
    mode = (mode or "").upper()
    dual = True if mode == "HEDGE" else False
    try:
        # 1차: snake_case
        _call(client.rest_api, ["change_position_mode", "changePositionMode", "change_position_side_dual"],
              dual_side_position=str(dual).lower())
        logger.info("Position mode set to %s (snake_case)", mode)
        return
    except Exception as e1:
        msg1 = str(getattr(e1, "message", str(e1)))
        if "no need to change" in msg1.lower():
            logger.info("Position mode already %s", mode)
            return
        logger.info("Position mode snake_case failed: %s", msg1)
    try:
        # 2차: camelCase
        _call(client.rest_api, ["change_position_mode", "changePositionMode", "change_position_side_dual"],
              dualSidePosition=str(dual).lower())
        logger.info("Position mode set to %s (camelCase)", mode)
    except Exception as e2:
        logger.error("Position mode set error: %s", str(getattr(e2, "message", str(e2))))

def set_margin_type(symbol: str, margin_type: str = "ISOLATED") -> None:
    client = _get_client()
    mt = (margin_type or "").upper()
    try:
        _call(client.rest_api, ["change_margin_type", "changeMarginType"], symbol=symbol, margin_type=mt)
        logger.info("Margin type set to %s for %s", mt, symbol)
    except Exception as e:
        msg = str(getattr(e, "message", str(e)))
        if "no need to change" in msg.lower():
            logger.info("Margin type already %s for %s", mt, symbol)
        else:
            logger.error("Margin type set error: %s", msg)

def set_leverage(symbol: str, leverage: int) -> None:
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
    return os.getenv("POSITION_SIDE", "BOTH").upper()

def _last_price(symbol: str) -> float:
    client = _get_client()
    # 1) mark/idx/last
    try:
        r = _call(client.rest_api, ["premium_index", "mark_price", "premiumIndex"], symbol=symbol)
        d = r.data() if hasattr(r, "data") else r
        for k in ("markPrice", "indexPrice", "lastPrice", "price"):
            v = (d.get(k) if isinstance(d, dict) else getattr(d, k, None))
            if v is not None:
                return float(v)
    except Exception:
        pass
    # 2) ticker
    try:
        r = _call(client.rest_api, ["ticker_price", "tickerPrice"], symbol=symbol)
        d = r.data() if hasattr(r, "data") else r
        v = d.get("price") if isinstance(d, dict) else getattr(d, "price", None)
        if v is not None:
            return float(v)
    except Exception:
        pass
    # 3) orderbook mid
    try:
        r = _call(client.rest_api, ["depth", "order_book"], symbol=symbol, limit=5)
        d = r.data() if hasattr(r, "data") else r
        bids = d.get("bids") or []
        asks = d.get("asks") or []
        if bids and asks:
            return (float(bids[0][0]) + float(asks[0][0])) / 2.0
    except Exception:
        pass
    return 0.0

def get_last_price(symbol: str) -> float:
    """외부 공개"""
    return _last_price(symbol)

def _quantize_qty(symbol: str, qty: float, at_price: float) -> float:
    f = load_symbol_filters(symbol)
    return ensure_min_notional(symbol, qty, price=at_price, filters=f)

def place_market_order(symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
    client = _get_client()
    side = side.upper()
    ps = _position_side()

    try:
        ref = _last_price(symbol)
    except Exception:
        ref = 0.0
    try:
        if ref > 0:
            q = _quantize_qty(symbol, quantity, at_price=ref)
        else:
            f = load_symbol_filters(symbol)
            q = float(round_to_step(_to_decimal(quantity), f["stepSize"]))
    except Exception:
        q = float(quantity)

    if q <= 0:
        raise ValueError("Normalized quantity <= 0")

    payload = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": str(q),
        "reduce_only": bool(reduce_only),
        "position_side": ps,
    }
    log_event("binance.order.request", **payload)
    try:
        resp = _call(client.rest_api, ["new_order", "newOrder"], **payload)
        raw = resp.data() if hasattr(resp, "data") else resp
        data = _to_plain(raw)
        log_event("binance.order.response",
                  symbol=symbol, side=side, type="MARKET",
                  orderId=(data.get("orderId") if isinstance(data, dict) else None),
                  status=(data.get("status") if isinstance(data, dict) else None),
                  price=(data.get("avgPrice") or data.get("price") if isinstance(data, dict) else None),
                  qty=(data.get("executedQty") or data.get("origQty") if isinstance(data, dict) else None),
                  raw=data)
        return data if isinstance(data, dict) else {"raw": data}
    except Exception as e:
        logger.error("place_market_order error: %s", e)
        raise

def place_limit_order(symbol: str, side: str, quantity: float, price: float,
                      time_in_force: str = "GTC", reduce_only: bool = False,
                      post_only: bool = False) -> Dict[str, Any]:
    client = _get_client()
    side = side.upper()
    ps = _position_side()

    px = normalize_price_for_side(symbol, price, side)
    q = _quantize_qty(symbol, quantity, at_price=px)

    payload = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "time_in_force": time_in_force,  # GTC | IOC | FOK | GTX(POST-ONLY)
        "price": _format_to_tick_str(symbol, px),
        "quantity": str(q),
        "reduce_only": bool(reduce_only),
        "position_side": ps,
    }

    log_event("binance.order.request", **payload)
    try:
        resp = _call(client.rest_api, ["new_order", "newOrder"], **payload)
        raw = resp.data() if hasattr(resp, "data") else resp
        data = _to_plain(raw)
        log_event("binance.order.response",
                  symbol=symbol, side=side, type="LIMIT",
                  orderId=(data.get("orderId") if isinstance(data, dict) else None),
                  status=(data.get("status") if isinstance(data, dict) else None),
                  price=(data.get("price") if isinstance(data, dict) else None),
                  qty=(data.get("executedQty") or data.get("origQty") if isinstance(data, dict) else None),
                  raw=data)
        return data if isinstance(data, dict) else {"raw": data}
    except Exception as e:
        logger.error("place_limit_order error: %s", e)
        # POST-ONLY 의도일 때는 MARKET 폴백 금지
        if _LIMIT_FAILOVER_TO_MARKET and (not post_only):
            logger.info("Falling back to MARKET due to limit error for %s: %s", symbol, e)
            return place_market_order(symbol, side, quantity, reduce_only=reduce_only)
        raise

def place_take_profit(symbol: str, opp_side: str, quantity: float, tp_price: float, order_type: str = "LIMIT") -> Dict[str, Any]:
    client = _get_client()
    ps = _position_side()
    order_type = (order_type or _TP_ORDER_TYPE).upper()

    if order_type == "MARKET":
        tp_payload = {
            "symbol": symbol,
            "side": opp_side,
            "type": "TAKE_PROFIT_MARKET",
            "stop_price": _format_to_tick_str(symbol, tp_price),
            "working_type": "MARK_PRICE",
            "quantity": str(quantity),
            "reduce_only": True,
            "position_side": ps,
        }
    else:
        tp_p = normalize_price_for_side(symbol, tp_price, opp_side)
        tp_payload = {
            "symbol": symbol,
            "side": opp_side,
            "type": "TAKE_PROFIT",
            "time_in_force": "GTC",
            "price": _format_to_tick_str(symbol, tp_p),
            "quantity": str(quantity),
            "stop_price": _format_to_tick_str(symbol, tp_p),
            "working_type": "MARK_PRICE",
            "reduce_only": True,
            "position_side": ps,
        }
    log_event("binance.order.request", **tp_payload)
    resp = _call(client.rest_api, ["new_order", "newOrder"], **tp_payload)
    raw = resp.data() if hasattr(resp, "data") else resp
    data = _to_plain(raw)
    return data if isinstance(data, dict) else {"raw": data}

def place_stop_market(symbol: str, opp_side: str, quantity: float, sl_price: float) -> Dict[str, Any]:
    client = _get_client()
    ps = _position_side()
    sl_payload = {
        "symbol": symbol,
        "side": opp_side,
        "type": "STOP_MARKET",
        "stop_price": _format_to_tick_str(symbol, sl_price),
        "working_type": "MARK_PRICE",
        "quantity": str(quantity),
        "reduce_only": True,
        "position_side": ps,
    }
    log_event("binance.order.request", **sl_payload)
    sl_resp = _call(client.rest_api, ["new_order", "newOrder"], **sl_payload)
    raw = sl_resp.data() if hasattr(sl_resp, "data") else sl_resp
    data = _to_plain(raw)
    return data if isinstance(data, dict) else {"raw": data}

def place_stop_limit(symbol: str, opp_side: str, quantity: float, stop_price: float,
                     limit_slippage_bps: float = _SL_LIMIT_SLIPPAGE_BPS) -> Dict[str, Any]:
    """
    STOP_LIMIT(SL=STOP) 생성.
    SELL(롱 청산): limit = stop * (1 - ε)
    BUY (숏 청산): limit = stop * (1 + ε)
    """
    client = _get_client()
    ps = _position_side()
    opp_side = opp_side.upper()
    eps = float(limit_slippage_bps) / 1e4
    if opp_side == "SELL":
        limit_px = stop_price * (1.0 - eps)
    else:
        limit_px = stop_price * (1.0 + eps)
    limit_px = normalize_price_for_side(symbol, limit_px, opp_side)

    payload = {
        "symbol": symbol,
        "side": opp_side,
        "type": "STOP",
        "time_in_force": "GTC",
        "price": _format_to_tick_str(symbol, limit_px),
        "stop_price": _format_to_tick_str(symbol, stop_price),
        "working_type": "MARK_PRICE",
        "quantity": str(quantity),
        "reduce_only": True,
        "position_side": ps,
    }
    log_event("binance.order.request", **payload)
    resp = _call(client.rest_api, ["new_order", "newOrder"], **payload)
    raw = resp.data() if hasattr(resp, "data") else resp
    data = _to_plain(raw)
    return data if isinstance(data, dict) else {"raw": data}

def place_bracket_orders(symbol: str, side: str, quantity: float, take_profit: float, stop_loss: float) -> Dict[str, Any]:
    """엔트리 직후 브래킷(RO) 생성: TP(LIMIT|MARKET), SL(STOP_MARKET|STOP)"""
    side = side.upper()
    opp = "SELL" if side == "BUY" else "BUY"

    tp_price = float(take_profit)
    sl_price = float(stop_loss)

    f = load_symbol_filters(symbol)
    qty_q = ensure_min_notional(symbol, float(quantity), price=max(tp_price, sl_price), filters=f)

    out: Dict[str, Any] = {"take_profit": None, "stop_loss": None}
    try:
        tp_data = place_take_profit(symbol, opp, qty_q, tp_price, order_type=_TP_ORDER_TYPE)
        tp_plain = _to_plain(tp_data)
        log_event("binance.order.response",
                  symbol=symbol, side=opp, type=("TAKE_PROFIT_MARKET" if _TP_ORDER_TYPE=="MARKET" else "TAKE_PROFIT"),
                  orderId=(tp_plain.get("orderId") if isinstance(tp_plain, dict) else None),
                  status=(tp_plain.get("status") if isinstance(tp_plain, dict) else None),
                  price=tp_price, qty=qty_q, raw=tp_plain)
        out["take_profit"] = tp_plain if isinstance(tp_plain, dict) else {"raw": tp_plain}
    except Exception as e:
        logger.info("place_bracket_orders TP failed: %s", e)

    try:
        if _SL_ORDER_TYPE == "STOP":
            sl_data = place_stop_limit(symbol, opp, qty_q, sl_price, limit_slippage_bps=_SL_LIMIT_SLIPPAGE_BPS)
        else:
            sl_data = place_stop_market(symbol, opp, qty_q, sl_price)
        sl_plain = _to_plain(sl_data)
        log_event("binance.order.response",
                  symbol=symbol, side=opp, type=("STOP" if _SL_ORDER_TYPE=="STOP" else "STOP_MARKET"),
                  orderId=(sl_plain.get("orderId") if isinstance(sl_plain, dict) else None),
                  status=(sl_plain.get("status") if isinstance(sl_plain, dict) else None),
                  price=None, qty=qty_q, raw=sl_plain)
        out["stop_loss"] = sl_plain if isinstance(sl_plain, dict) else {"raw": sl_plain}
    except Exception as e:
        logger.info("place_bracket_orders SL failed: %s", e)

    return out

def build_entry_and_brackets(symbol: str, side: str, quantity: float, target_price: float, stop_price: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    entry = place_market_order(symbol, side, quantity)
    bracket = place_bracket_orders(symbol, side, quantity, target_price, stop_price)
    return (_to_plain(entry) if entry is not None else None,
            _to_plain(bracket) if bracket is not None else None)

# ==================
# Readbacks
# ==================
def get_open_orders(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """현재 미체결 주문 나열(타입 정규화 없음)"""
    client = _get_client()
    try:
        cand_names = [
            "open_orders", "openOrders", "allOpenOrders", "get_open_orders", "getAllOpenOrders",
            "current_all_open_orders", "query_current_all_open_orders",
            "currentAllOpenOrders", "queryCurrentAllOpenOrders",
            "current_open_orders", "query_current_open_orders",
            "currentOpenOrders", "queryCurrentOpenOrders",
        ]
        fn = _pick(client.rest_api, cand_names)
        if not fn:
            candidates = [n for n in dir(client.rest_api)
                          if ("order" in n.lower() and "open" in n.lower() and callable(getattr(client.rest_api, n)))]
            fn = getattr(client.rest_api, candidates[0], None) if candidates else None
            if not fn:
                raise AttributeError("No method for open orders")

        r = fn(symbol=symbol) if symbol else fn()
        data = r.data() if hasattr(r, "data") else r

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "orders" in data and isinstance(data["orders"], list):
                return data["orders"]
            return [data]
        if hasattr(data, "__iter__"):
            return list(data)

        return [_as_plain_dict(data)]
    except Exception as e:
        logger.error(f"get_open_orders failed: {e}")
        return []

def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for attr in ("to_dict", "as_dict"):
        if hasattr(obj, attr):
            try:
                d = getattr(obj, attr)()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    if hasattr(obj, "model_dump"):
        try:
            d = obj.model_dump()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
        except Exception:
            pass
    out: Dict[str, Any] = {}
    try:
        for k in dir(obj):
            if k.startswith("_"): continue
            try:
                v = getattr(obj, k)
            except Exception:
                continue
            if callable(v): continue
            out[k] = v
    except Exception:
        pass
    return out

def _to_plain(obj):
    """Binance SDK 응답 객체(Pydantic 등)를 JSON-가능한 순수 구조로 재귀 변환"""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    # pydantic BaseModel 호환
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return _to_plain(getattr(obj, attr)())
            except Exception:
                pass
    # 일반 객체
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_plain(v) for k, v in obj.__dict__.items() if not str(k).startswith("_")}
        except Exception:
            pass
    return obj

def _positions_from_response(resp: Any) -> List[Any]:
    data = resp.data() if hasattr(resp, "data") else resp
    if hasattr(data, "positions"):
        try:
            items = getattr(data, "positions")
            if isinstance(items, (list, tuple)):
                return list(items)
        except Exception:
            pass
    if isinstance(data, dict):
        items = data.get("positions")
        if isinstance(items, list):
            return items
    if isinstance(data, list):
        return data
    return [data]

def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    """심볼 포지션 단건 반환. 심볼 호출 실패→전체 조회 후 필터링."""
    client = _get_client()
    sym_up = (symbol or "").upper()

    def _try_call(with_symbol: bool):
        return _call(
            client.rest_api,
            ["position_information_v3", "position_information_v2", "position_information", "position_risk"],
            **({"symbol": symbol} if with_symbol else {})
        )

    try:
        resp = _try_call(with_symbol=True)
        items = _positions_from_response(resp)
    except Exception as e1:
        logger.info("get_position first call failed (with symbol): %s", e1)
        try:
            resp = _try_call(with_symbol=False)
            items = _positions_from_response(resp)
        except Exception as e2:
            logger.info("get_position second call failed (without symbol): %s", e2)
            return None

    for p in items or []:
        d = _as_plain_dict(p)
        sym = (d.get("symbol") or d.get("s") or getattr(p, "symbol", None) or getattr(p, "pair", None) or "")
        if str(sym).upper() == sym_up:
            try:
                amt = float(d.get("positionAmt") or d.get("positionAmount") or 0.0)
                if abs(amt) < 1e-12:
                    continue
            except Exception:
                pass
            return d
    return None

# ==================
# SL 교체 유틸
# ==================
def replace_stop_loss_to_price(symbol: str, is_long: bool, quantity: float, new_stop_price: float) -> Dict[str, Any]:
    """
    기존 STOP/STOP_MARKET만 취소 후, 지정 가격으로 STOP_MARKET 재배치.
    TP는 유지.
    """
    try:
        cancel_orders_by_type(symbol, ["STOP", "STOP_MARKET"])
    except Exception:
        pass
    opp = "SELL" if is_long else "BUY"
    return place_stop_market(symbol, opp, quantity, new_stop_price)

__all__ = [
    "get_client",
    # overview
    "get_overview",
    # filters & utils
    "load_symbol_filters",
    "ensure_min_notional",
    "normalize_price_with_mode",
    "normalize_price_for_side",
    "get_last_price",
    # account mutators
    "cancel_open_orders",
    "cancel_order",
    "cancel_orders_by_type",
    "set_position_mode",
    "set_margin_type",
    "set_leverage",
    # orders
    "place_market_order",
    "place_limit_order",
    "place_take_profit",
    "place_stop_market",
    "place_bracket_orders",
    "build_entry_and_brackets",
    # readbacks & updates
    "get_open_orders",
    "get_position",
    "replace_stop_loss_to_price",
]
