from __future__ import annotations

import logging
import os
import time
import uuid
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Any, Dict, Optional, Tuple, List, Union

import pandas as pd

# === New modular SDK (USDS-M Futures) ===
# pip install binance-sdk-derivatives-trading-usds-futures==1.0.0
from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

from .utils import get_secret

getcontext().prec = 28

__all__ = [
    "get_client",
    "set_leverage",
    "set_margin_type",
    "set_position_mode",
    "get_overview",
    "get_position",
    "place_market_order",
    "place_limit_order",
    "wait_order_filled",
    "place_bracket_orders",
    "build_entry_and_brackets",
    "normalize_price",
    "normalize_price_with_mode",
    "normalize_qty",
    "ensure_min_notional",
    "load_symbol_filters",
    "cancel_open_orders",
]

# --- API keys ---
_api_key = get_secret("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY")
_api_secret = get_secret("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET")
if not _api_key or not _api_secret:
    logging.warning("Binance API key/secret not provided; running in unauthenticated mode.")

# --- Env / endpoints ---
_USE_TESTNET = os.getenv("BINANCE_FUTURES_TESTNET", "").lower() in ("1", "true", "yes")
_BASE_PATH = (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
    if _USE_TESTNET
    else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
)

_timeout_ms = int(os.getenv("BINANCE_HTTP_TIMEOUT_MS", "10000"))
_retries = int(os.getenv("BINANCE_HTTP_RETRIES", "3"))
_backoff_ms = int(os.getenv("BINANCE_HTTP_BACKOFF_MS", "1000"))

_config = ConfigurationRestAPI(
    api_key=_api_key,
    api_secret=_api_secret,
    base_path=_BASE_PATH,
    timeout=_timeout_ms,
    retries=_retries,
    backoff=_backoff_ms,
    keep_alive=True,
    compression=True,
)
logging.info(f"Binance config initialized: base_path={_config.base_path}, api_key_set={bool(_api_key)}")

# Client (lazy init 함수로 변경)
_client: Optional[DerivativesTradingUsdsFutures] = None

def get_client() -> DerivativesTradingUsdsFutures:
    global _client
    if _client is None:
        try:
            _client = DerivativesTradingUsdsFutures(config_rest_api=_config)
            logging.info("Binance client initialized successfully.")
            # 테스트 호출: 서버 시간으로 init 확인 (메서드 이름 수정)
            try:
                _client.rest_api.get_server_time()  # SDK에서 가능한 메서드 이름 (문서 기반)
            except AttributeError:
                logging.warning("get_server_time method not found; skipping init test.")
            except Exception as exc:
                logging.warning(f"Init test failed: {exc}; proceeding anyway.")
        except Exception as exc:
            logging.error(f"Client init error: {exc}")
            raise
    return _client

# --- Utilities ---
def _data(resp: Any) -> Any:
    """Return SDK Response.data() if available, else resp itself."""
    try:
        return resp.data() if hasattr(resp, "data") else resp
    except Exception:
        return resp

def _try_methods(obj: Any, candidates: List[str]) -> Optional[Any]:
    """Return first callable attribute found in candidates, else None."""
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None

# ------- Exchange Info & Filters (cached) -------
_symbol_filters_cache: Dict[str, Dict[str, Any]] = {}

def _to_decimal(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))

def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step == 0:
        return value
    return (value // step) * step

def load_symbol_filters(_: Any, symbol: str) -> Dict[str, Any]:
    """
    Fetch exchange info and extract filters for `symbol` (cached).
    """
    symbol = symbol.upper()
    if symbol in _symbol_filters_cache:
        return _symbol_filters_cache[symbol]

    fn = _try_methods(
        get_client().rest_api,
        ["exchange_information", "exchange_info", "get_exchange_information"],
    )
    if not fn:
        raise RuntimeError("exchange_information method not found in SDK.")

    ex = _data(fn(symbol=symbol))
    if isinstance(ex, dict):
        syms = ex.get("symbols") or []
    else:
        syms = []

    sym = None
    for s in syms:
        if str(s.get("symbol", "")).upper() == symbol:
            sym = s
            break
    if not sym and syms:
        sym = syms[0]
    if not sym:
        raise KeyError(f"Symbol {symbol} not found in exchange info.")

    filters_map = {f.get("filterType"): f for f in sym.get("filters", [])}

    tick = _to_decimal(filters_map.get("PRICE_FILTER", {}).get("tickSize", "0"))
    lot = filters_map.get("LOT_SIZE") or {}
    market_lot = filters_map.get("MARKET_LOT_SIZE") or {}
    step_size = _to_decimal((market_lot or lot).get("stepSize", "0"))
    min_qty = _to_decimal((market_lot or lot).get("minQty", "0"))
    max_qty = _to_decimal((market_lot or lot).get("maxQty", "0"))
    notional_min = _to_decimal(
        (filters_map.get("NOTIONAL") or filters_map.get("MIN_NOTIONAL") or {}).get("notional", "0")
    )

    filters = {
        "tickSize": tick,
        "stepSize": step_size,
        "minQty": min_qty,
        "maxQty": max_qty,
        "minNotional": notional_min,
        "raw": sym,
    }
    _symbol_filters_cache[symbol] = filters
    return filters

def normalize_price(price: Decimal, tick: Decimal) -> Decimal:
    return round_to_step(price, tick)

def normalize_price_with_mode(price: Decimal, tick: Decimal, mode: str = "floor") -> Decimal:
    if tick <= 0:
        return price
    q, r = divmod(price, tick)
    if mode == "ceil" and r > 0:
        return (q + 1) * tick
    return q * tick

def normalize_qty(qty: Decimal, step: Decimal, min_qty: Decimal) -> Decimal:
    q = round_to_step(qty, step)
    if q < min_qty:
        q = min_qty
    return q

def ensure_min_notional(qty: Decimal, price: Decimal, step: Decimal, min_notional: Decimal) -> Decimal:
    if min_notional <= 0:
        return qty
    notional = qty * price
    if notional >= min_notional:
        return qty
    need = (min_notional / price)
    k = (need / step).to_integral_value(rounding=ROUND_DOWN)
    if k * step < need:
        k += 1
    return k * step

# ------- Trading params -------
def set_leverage(symbol: str, leverage: int = 10) -> None:
    try:
        fn = _try_methods(
            get_client().rest_api,
            ["change_initial_leverage", "change_leverage", "initial_leverage"],
        )
        if not fn:
            logging.warning("change_initial_leverage not found in SDK.")
            return
        resp = fn(symbol=symbol, leverage=leverage)
        logging.info(f"Leverage set to {leverage}x for {symbol}: {_data(resp)}")
    except Exception as exc:
        msg = str(exc)
        if any(code in msg for code in ("-4059", "-4046")):
            logging.info(f"Leverage already set/no change: {exc}")
        else:
            logging.error(f"Leverage set error: {exc}")

def set_margin_type(symbol: str, margin_type: str = "ISOLATED") -> None:
    try:
        fn = _try_methods(get_client().rest_api, ["change_margin_type", "change_margin"])
        if not fn:
            logging.warning("change_margin_type not found in SDK.")
            return
        mt = margin_type.upper()
        if mt not in ("ISOLATED", "CROSSED"):
            mt = "ISOLATED"
        kw = {"margin_type": mt} if "margin_type" in fn.__code__.co_varnames else {"marginType": mt}
        resp = fn(symbol=symbol, **kw)
        logging.info(f"Margin type set to {mt} for {symbol}: {_data(resp)}")
    except Exception as exc:
        msg = str(exc)
        if any(code in msg for code in ("-4059", "-4046")):
            logging.info(f"Margin type already set/no change: {exc}")
        else:
            logging.error(f"Margin type set error: {exc}")

def set_position_mode(dual_side: bool = False) -> None:
    try:
        fn = _try_methods(get_client().rest_api, ["change_position_mode", "position_side_dual"])
        if not fn:
            logging.warning("change_position_mode not found in SDK.")
            return
        # v1.0.0 → snake_case
        kw = {"dual_side_position": dual_side} if "dual_side_position" in fn.__code__.co_varnames else {"dualSidePosition": dual_side}
        resp = fn(**kw)
        logging.info(f"Position mode set dual_side={dual_side}: {_data(resp)}")

    except Exception as exc:
        msg = str(exc)
        if any(code in msg for code in ("-4059", "-4046")):
            logging.info(f"Position mode already set/no change: {exc}")
        else:
            logging.error(f"Position mode set error: {exc}")

# ------- Orders -------
def _gen_client_id(prefix: str = "tm") -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

def place_market_order(symbol: str, side: str, quantity: float, new_client_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not _api_key or not _api_secret:
        logging.warning("Cannot place order: missing API credentials.")
        return None
    try:
        if not new_client_id:
            new_client_id = _gen_client_id("mkt")
        fn = _try_methods(get_client().rest_api, ["new_order", "order", "post_order"])
        if not fn:
            raise RuntimeError("new_order method not found in SDK.")
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": str(quantity),
            "newOrderRespType": "RESULT",
            "newClientOrderId": new_client_id,
        }
        response = fn(**params)
        unwrapped = _data(response)
        logging.info(f"Placed {side.upper()} MARKET {quantity} {symbol}: {unwrapped}")
        return unwrapped if isinstance(unwrapped, dict) else {"__raw__": unwrapped}
    except Exception as exc:
        logging.error(f"Order error: {exc}")
        return None

def place_limit_order(symbol: str, side: str, quantity: float, price: float, new_client_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not _api_key or not _api_secret:
        logging.warning("Cannot place order: missing API credentials.")
        return None
    try:
        if not new_client_id:
            new_client_id = _gen_client_id("lmt")
        fn = _try_methods(get_client().rest_api, ["new_order", "order", "post_order"])
        if not fn:
            raise RuntimeError("new_order method not found in SDK.")
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": str(quantity),
            "price": str(price),
            "timeInForce": "GTC",
            "newOrderRespType": "RESULT",
            "newClientOrderId": new_client_id,
        }
        response = fn(**params)
        unwrapped = _data(response)
        logging.info(f"Placed {side.upper()} LIMIT {quantity} {symbol} @ {price}: {unwrapped}")
        return unwrapped if isinstance(unwrapped, dict) else {"__raw__": unwrapped}
    except Exception as exc:
        logging.error(f"Order error: {exc}")
        return None

def _cancel_order(symbol: str, order_id: Optional[Union[int, str]] = None, client_order_id: Optional[str] = None) -> Any:
    fn = _try_methods(get_client().rest_api, ["cancel_order", "delete_order"])  # DELETE /fapi/v1/order
    if not fn:
        raise RuntimeError("cancel_order not found in SDK.")
    if order_id is not None:
        return _data(fn(symbol=symbol, orderId=order_id))
    if client_order_id:
        return _data(fn(symbol=symbol, origClientOrderId=client_order_id))
    raise ValueError("Either order_id or client_order_id required.")

def wait_order_filled(
    symbol: str,
    order_id: Optional[Union[int, str]] = None,
    client_order_id: Optional[str] = None,
    max_checks: int = 30,
    sleep_sec: float = 1.0,
    timeout_cancel: bool = False,
) -> Optional[Dict[str, Any]]:
    last = None
    q_fn = _try_methods(get_client().rest_api, ["query_order", "get_order"])
    if not q_fn:
        logging.error("query_order not found in SDK.")
        return None

    for _ in range(max_checks):
        try:
            if order_id is not None:
                res = q_fn(symbol=symbol, orderId=order_id)
            elif client_order_id:
                res = q_fn(symbol=symbol, origClientOrderId=client_order_id)
            else:
                break
            data = _data(res)
            last = data
            status = str((data or {}).get("status", "")).upper()
            if status in ("FILLED", "PARTIALLY_FILLED"):
                return data if isinstance(data, dict) else {"__raw__": data}
            elif status in ("CANCELED", "REJECTED", "EXPIRED"):
                return data if isinstance(data, dict) else {"__raw__": data}
        except Exception as exc:
            logging.error(f"query_order error: {exc}")
        time.sleep(sleep_sec)

    if timeout_cancel and (order_id is not None or client_order_id):
        try:
            _cancel_order(symbol, order_id, client_order_id)
            return {"status": "CANCELLED", "reason": "Timeout"}
        except Exception as exc:
            logging.error(f"Cancel on timeout error: {exc}")
            return {"status": "CANCEL_FAILED", "reason": str(exc)}
    return last if isinstance(last, dict) else ({"__raw__": last} if last else None)

def cancel_open_orders(symbol: str) -> bool:
    try:
        # 1) Bulk cancel
        bulk = _try_methods(get_client().rest_api, ["cancel_all_open_orders", "delete_all_open_orders"])  # DELETE /fapi/v1/allOpenOrders
        if bulk:
            try:
                resp = _data(bulk(symbol=symbol))
                logging.info(f"Bulk-cancel {symbol}: {resp}")
                return True
            except Exception as bulk_exc:
                logging.warning(f"Bulk-cancel failed for {symbol}: {bulk_exc} → per-order fallback")

        # 2) Per-order fallback
        get_open = _try_methods(
            get_client().rest_api,
            ["query_current_all_open_orders", "get_open_orders", "current_all_open_orders"],
        )
        if not get_open:
            logging.info("Open orders fetch method not found; nothing to cancel.")
            return True
        orders = _data(get_open(symbol=symbol))
        if not orders or not isinstance(orders, list):
            logging.info(f"No open orders for {symbol}")
            return True
        for od in orders:
            try:
                oid = od.get("orderId")
                coid = od.get("clientOrderId") or od.get("origClientOrderId")
                _cancel_order(symbol, oid, coid)
            except Exception as per_exc:
                logging.error(f"Failed to cancel order for {symbol}: {per_exc}")
        return True
    except Exception as exc:
        logging.error(f"Error cancelling open orders for {symbol}: {exc}")
        return False

def place_bracket_orders(
    symbol: str,
    side: str,
    quantity: float,
    target_price: float,
    stop_price: float,
    working_type: str = "MARK_PRICE",
) -> Optional[Dict[str, Any]]:
    if not _api_key or not _api_secret:
        logging.warning("Cannot place bracket orders: missing API credentials.")
        return None
    try:
        if side.upper() == "BUY":
            tp_side = "SELL"; sl_side = "SELL"; tp_mode = "ceil"; sl_mode = "floor"
        else:
            tp_side = "BUY";  sl_side = "BUY";  tp_mode = "floor"; sl_mode = "ceil"

        filters = load_symbol_filters(None, symbol)
        tick = filters["tickSize"]
        tp_trigger = normalize_price_with_mode(Decimal(str(target_price)), tick, tp_mode)
        sl_trigger = normalize_price_with_mode(Decimal(str(stop_price)),  tick, sl_mode)

        new_order_fn = _try_methods(get_client().rest_api, ["new_order", "order", "post_order"])
        if not new_order_fn:
            raise RuntimeError("new_order not found in SDK.")

        # closePosition=True uses reduce-only close for the entire position; quantity omitted
        tp_params = dict(
            symbol=symbol, side=tp_side, type="TAKE_PROFIT_MARKET",
            stopPrice=str(tp_trigger), closePosition=True, workingType=working_type,
        )
        sl_params = dict(
            symbol=symbol, side=sl_side, type="STOP_MARKET",
            stopPrice=str(sl_trigger), closePosition=True, workingType=working_type,
        )

        tp_order = _data(new_order_fn(**tp_params))
        sl_order = _data(new_order_fn(**sl_params))
        return {"tp": tp_order, "sl": sl_order}
    except Exception as exc:
        logging.error(f"Bracket orders error: {exc}")
        return None

# ------- Account overview -------
def get_overview() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch current futures balances and positions.

    The USDS‑M modular SDK introduced new method names for account and balance
    endpoints (e.g. ``account_information_v2``, ``futures_account_balance_v3``),
    whereas previous versions exposed simple names like ``account`` and
    ``balance``.  Calling an undefined method on the SDK object can lead to
    unpredictable runtime errors (e.g. ``list index out of range``).  To
    remain compatible across versions, this function attempts a prioritized list
    of method names for both balances and account information and falls back
    gracefully if one fails.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of (balances_df,
        positions_df).  Both dataframes may be empty if API credentials are
        missing or the endpoints return no data.
    """
    bal_df: pd.DataFrame = pd.DataFrame()
    pos_df: pd.DataFrame = pd.DataFrame()

    # -- Balance retrieval --
    balance_methods: List[str] = [
        "futures_account_balance_v3",
        "futures_account_balance_v2",
        "futures_account_balance",
        "balance",
        "account_balance_v3",
        "account_balance_v2",
    ]
    try:
        bal_fn = _try_methods(get_client().rest_api, balance_methods)
        if bal_fn:
            raw = _data(bal_fn())
            # Normalize list of objects (dicts or model instances)
            if isinstance(raw, list):
                records: List[Dict[str, Any]] = []
                for item in raw:
                    if isinstance(item, dict):
                        records.append(item)
                    elif hasattr(item, "to_dict") and callable(getattr(item, "to_dict", None)):
                        try:
                            records.append(item.to_dict())
                        except Exception:
                            pass
                    elif hasattr(item, "__dict__"):
                        # Filter out private attributes
                        d = {
                            k: v
                            for k, v in vars(item).items()
                            if not k.startswith("_")
                        }
                        records.append(d)
                if records:
                    bal_df = pd.json_normalize(records)
            elif isinstance(raw, dict):
                # Some responses wrap data inside a 'data' field
                data_section = raw.get("data")
                if isinstance(data_section, list):
                    records: List[Dict[str, Any]] = []
                    for item in data_section:
                        if isinstance(item, dict):
                            records.append(item)
                        elif hasattr(item, "to_dict") and callable(getattr(item, "to_dict", None)):
                            try:
                                records.append(item.to_dict())
                            except Exception:
                                pass
                        elif hasattr(item, "__dict__"):
                            d = {
                                k: v
                                for k, v in vars(item).items()
                                if not k.startswith("_")
                            }
                            records.append(d)
                    if records:
                        bal_df = pd.json_normalize(records)
                else:
                    balances = raw.get("balances") or raw.get("assets") or []
                    if isinstance(balances, list):
                        records: List[Dict[str, Any]] = []
                        for item in balances:
                            if isinstance(item, dict):
                                records.append(item)
                            elif hasattr(item, "to_dict") and callable(getattr(item, "to_dict", None)):
                                try:
                                    records.append(item.to_dict())
                                except Exception:
                                    pass
                            elif hasattr(item, "__dict__"):
                                d = {
                                    k: v
                                    for k, v in vars(item).items()
                                    if not k.startswith("_")
                                }
                                records.append(d)
                        if records:
                            bal_df = pd.json_normalize(records)
    except Exception as exc:
        logging.error(f"Error fetching balances: {exc}")

    # -- Position retrieval --
    account_methods: List[str] = [
        "account_information_v3",
        "account_information_v2",
        "account_information",
        "account_v3",
        "account_v2",
        "account",
    ]
    try:
        acct_fn = _try_methods(get_client().rest_api, account_methods)
        if acct_fn:
            acct_raw = _data(acct_fn())
            # Account information may be a model instance with to_dict
            if hasattr(acct_raw, "to_dict") and callable(getattr(acct_raw, "to_dict", None)):
                try:
                    acct_raw = acct_raw.to_dict()
                except Exception:
                    pass
            elif not isinstance(acct_raw, dict) and hasattr(acct_raw, "__dict__"):
                acct_raw = {
                    k: v for k, v in vars(acct_raw).items() if not k.startswith("_")
                }
            if isinstance(acct_raw, dict):
                pos_data = (
                    acct_raw.get("positions")
                    or acct_raw.get("positionRisk")
                    or acct_raw.get("positionsInformation")
                    or acct_raw.get("data")
                    or []
                )
                # Normalize list of position objects
                if isinstance(pos_data, list):
                    records: List[Dict[str, Any]] = []
                    for item in pos_data:
                        if isinstance(item, dict):
                            records.append(item)
                        elif hasattr(item, "to_dict") and callable(getattr(item, "to_dict", None)):
                            try:
                                records.append(item.to_dict())
                            except Exception:
                                pass
                        elif hasattr(item, "__dict__"):
                            records.append({
                                k: v for k, v in vars(item).items() if not k.startswith("_")
                            })
                    if records:
                        pos_df = pd.json_normalize(records)
        # NOTE: We intentionally avoid calling position_information_* methods here.
        # In some SDK versions these endpoints may throw internal errors such as
        # "list index out of range" when invoked without a symbol.  Account
        # information already provides position details, so we skip the extra
        # position_information calls to improve stability.
    except Exception as exc:
        logging.error(f"Error fetching positions: {exc}")

    # -- Post-processing --
    try:
        if not pos_df.empty and "positionAmt" in pos_df.columns:
            pos_df["positionAmt"] = pd.to_numeric(pos_df.get("positionAmt"), errors="coerce").fillna(0)
            pos_df = pos_df[pos_df["positionAmt"] != 0]

        needed = {"unrealizedProfit", "entryPrice", "positionAmt"}
        if not pos_df.empty and needed.issubset(set(pos_df.columns)):
            upnl = pd.to_numeric(pos_df["unrealizedProfit"], errors="coerce").fillna(0)
            entry = pd.to_numeric(pos_df["entryPrice"], errors="coerce").fillna(0)
            qty = pos_df["positionAmt"].abs()
            denom = (entry * qty).replace(0, pd.NA).fillna(1e-9)
            pos_df["roe_pct"] = (upnl / denom) * 100
        else:
            pos_df["roe_pct"] = 0.0
    except Exception as exc:
        logging.error(f"Post-processing positions failed: {exc}")

    return bal_df, pos_df

def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    _, positions = get_overview()
    if positions.empty:
        return None
    pos = positions[positions["symbol"] == symbol]
    if pos.empty:
        return None
    return pos.iloc[0].to_dict()

def build_entry_and_brackets(
    symbol: str, side: str, quantity: float, target_price: float, stop_price: float
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    entry = place_market_order(symbol, side, quantity)
    bracket = place_bracket_orders(symbol, side, quantity, target_price, stop_price)
    return entry, bracket