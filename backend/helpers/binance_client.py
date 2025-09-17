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
import time, random, string
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple
import inspect
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

# =====================
# Hedge/idempotent utils
# =====================
def _is_hedge_mode() -> bool:
    try:
        return str(os.getenv("POSITION_MODE", "ONEWAY")).upper() == "HEDGE"
    except Exception:
        return False

def _position_side_for(entry_side: str, reduce_only: bool) -> str:
    """
    ONEWAY  : 'BOTH'
    HEDGE   : 진입/청산 방향에 따라 LONG/SHORT 자동 결정
      - 진입(BUY)->LONG, 진입(SELL)->SHORT
      - reduce_only(SELL)->LONG 청산, reduce_only(BUY)->SHORT 청산
    """
    if not _is_hedge_mode():
        return "BOTH"
    s = str(entry_side).upper()
    if reduce_only:
        return "LONG" if s == "SELL" else "SHORT"
    return "LONG" if s == "BUY" else "SHORT"

def _should_send_reduce_only(order_type: str, reduce_only: bool) -> bool:
    """
    HEDGE 모드에서 MARKET/LIMIT 즉시 체결 주문은 positionSide=LONG/SHORT만으로
    '청산 전용' 의도가 충분히 전달되므로 reduceOnly는 불필요/거부될 수 있다.
    - ONEWAY(BOTH) 모드: reduceOnly 허용 (true일 때만 전송)
    - HEDGE 모드:
        * order_type in {"MARKET","LIMIT"} -> reduceOnly 전송하지 않음
        * 그 외(예: STOP_MARKET/TAKE_PROFIT[_MARKET])은 기존대로 전송 허용
    """
    if not reduce_only:
        return False
    try:
        hedge = (os.getenv("POSITION_MODE", "ONEWAY").upper() == "HEDGE")
    except Exception:
        hedge = False
    if hedge and str(order_type).upper() in ("MARKET", "LIMIT"):
        return False
    return True

def _new_client_id(tag: str = "E") -> str:
    ms = int(time.time() * 1000)
    rnd = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    return f"{tag}{ms}{rnd}"

def _safe_new_order(client, **payload):
    """
    모듈러 SDK의 new_order/newOrder 시그니처 차이를 흡수:
    - new_client_order_id vs newClientOrderId
    - position_side vs positionSide
    - reduce_only vs reduceOnly
    + 간단 재시도/백오프(네트워크/429/5xx) 2회
    """
    fn = _pick(client.rest_api, ["new_order", "newOrder"])
    if not fn:
        raise AttributeError("new_order/newOrder not found")
    # 서명 필터링
    try:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    def _filter(d: Dict[str, Any]) -> Dict[str, Any]:
        if not params:
            return d
        return {k: v for k, v in d.items() if k in params}

    # 케멀/스네이크 동시 주입 후 필터
    base = dict(payload)
    if "new_client_order_id" not in base and "newClientOrderId" not in base:
        cid = _new_client_id("E")
        base["new_client_order_id"] = cid
        base["newClientOrderId"] = cid
    if "position_side" in base and "positionSide" not in base:
        base["positionSide"] = base["position_side"]
    if "reduce_only" in base and "reduceOnly" not in base:
        base["reduceOnly"] = base["reduce_only"]
    if "time_in_force" in base and "timeInForce" not in base:
        base["timeInForce"] = base["time_in_force"]
    if "stop_price" in base and "stopPrice" not in base:
        base["stopPrice"] = base["stop_price"]
    if "working_type" in base and "workingType" not in base:
        base["workingType"] = base["working_type"]

    # 경미한 재시도(2회)
    back = [0.25, 0.75]
    last_err = None
    for i in range(1 + len(back)):
        try:
            return fn(**_filter(base))
        except Exception as e:
            last_err = e
            if i < len(back):
                time.sleep(back[i])
                continue
            raise last_err

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
# --- replace this function in helpers/binance_client.py ---
import inspect
def list_all_orders(symbol: str, limit: int = 100, start_time_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    최근 주문 히스토리(심볼별). 모듈러 SDK의 메서드명이 혼재하므로
    소수 후보만 시도하고, **시그니처에 존재하는 인자만** 전달한다.
    """
    client = _get_client()
    candidates = ["all_orders", "allOrders", "get_all_orders", "query_all_orders", "list_all_orders"]
    fn = None
    for name in candidates:
        cand = getattr(client.rest_api, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        # best-effort fallback
        names = [n for n in dir(client.rest_api) if "order" in n.lower() and "all" in n.lower()]
        fn = getattr(client.rest_api, names[0], None) if names else None
        if fn is None:
            return []

    # --- 시그니처 기반 안전 kwargs 구성 ---
    try:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    kwargs: Dict[str, Any] = {}
    if "symbol" in params: kwargs["symbol"] = symbol
    if "limit"  in params: kwargs["limit"] = int(limit)
    if start_time_ms:
        # camel/snake 중 존재하는 쪽만 선택
        if "start_time" in params:
            kwargs["start_time"] = int(start_time_ms)
        elif "startTime" in params:
            kwargs["startTime"] = int(start_time_ms)

    try:
        r = fn(**kwargs)
        data = r.data() if hasattr(r, "data") else r
        if isinstance(data, list):   return [_as_plain_dict(x) for x in data]
        if isinstance(data, dict):
            if isinstance(data.get("orders"), list): return data["orders"]
            return [_as_plain_dict(data)]
        if hasattr(data, "__iter__"): return list(data)
        return [_as_plain_dict(data)]
    except Exception as e:
        logger.info("list_all_orders failed for %s: %s", symbol, e)
        return []

def find_recent_exit_fill(symbol: str, since_ms: int, *, back_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
    try:
        tol = int(os.getenv("EXIT_SEARCH_BACK_MS", str(back_ms if back_ms is not None else 15 * 60_000)))
    except Exception:
        tol = 15 * 60_000  # 15분
    start = max(0, int(since_ms) - int(tol))

    orders = list_all_orders(symbol, limit=200, start_time_ms=start)
    if not orders:
        return None

    def _get(o: Dict[str, Any], *keys: str):
        for k in keys:
            if k in o and o[k] is not None:
                return o[k]
        return None

    def _ms(o: Dict[str, Any]) -> int:
        for k in ("updateTime", "transactTime", "time", "workingTime",
                  "update_time", "transact_time", "working_time"):
            v = _get(o, k)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
        return 0

    def _is_exit(o: Dict[str, Any]) -> bool:
        t = str(_get(o, "type") or "").upper()
        st = str(_get(o, "status") or "").upper()
        ro = _get(o, "reduceOnly", "reduce_only")
        std_exit = t in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET")
        fb_exit = (t in ("LIMIT", "MARKET") and (str(ro).lower() == "true"))
        return (st in ("FILLED", "PARTIALLY_FILLED")) and (std_exit or fb_exit)

    cands = [o for o in orders if _is_exit(o) and _ms(o) >= start]
    if not cands:
        return None

    cands.sort(key=_ms)
    o = cands[-1]
    typ = str(_get(o, "type") or "").upper()
    px = _get(o, "avgPrice", "avg_price", "price", "stopPrice", "stop_price",
              "activatePrice", "triggerPrice", "activate_price", "trigger_price")
    try:
        pxf = float(px) if px not in (None, "") else 0.0
    except Exception:
        pxf = 0.0

    return {"type": typ, "price": pxf, "time": _ms(o)}

# --- user trades fallback (Plan B) --------------------------------------------
def list_user_trades(
    symbol: str,
    limit: int = 1000,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    USDS-M Futures 사용자 체결 이력 조회 (메서드 명칭 차이 흡수).
    - 반환: dict 리스트 (SDK/REST 응답을 list[dict]로 정규화)
    """
    client = _get_client()
    candidates = [
        "user_trades", "get_user_trades", "my_trades",
        "account_trades", "get_account_trades",
        "userTrades", "get_userTrades", "get_myTrades",
    ]
    fn = _pick(getattr(client, "rest_api", client), candidates)
    if not fn:
        return []

    # 시그니처에 존재하는 키만 전달
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except Exception:
        params = {}
    kwargs: Dict[str, Any] = {}
    if "symbol" in params:
        kwargs["symbol"] = symbol
    if "limit" in params:
        kwargs["limit"] = int(limit)
    if start_time_ms is not None:
        if "start_time" in params:
            kwargs["start_time"] = int(start_time_ms)
        if "startTime" in params:
            kwargs["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        if "end_time" in params:
            kwargs["end_time"] = int(end_time_ms)
        if "endTime" in params:
            kwargs["endTime"] = int(end_time_ms)

    try:
        resp = fn(**kwargs)
        data = resp.data() if hasattr(resp, "data") else resp
        if isinstance(data, (list, tuple)):
            return [(_as_plain_dict(x) if not isinstance(x, dict) else x) for x in data]
        return []
    except Exception as e:
        logger.info("list_user_trades failed for %s: %s", symbol, e)
        return []


def find_recent_exit_trade(symbol: str, since_ms: int, *, back_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    주문 히스토리 기반 탐색이 실패했을 때를 대비한 '체결 이력' 폴백.
    - 전략: 엔트리 이후 [since_ms, since_ms + tol + 3분] 사이 체결 중
            realizedPnL(또는 동의어)이 '0이 아닌' 첫/마지막 체결을 청산으로 간주
    - 반환: {"type": "USER_TRADE", "price": float, "time": int(ms), "realized": float}
    """
    try:
        tol = int(os.getenv("EXIT_SEARCH_BACK_MS", str(back_ms if back_ms is not None else 15 * 60_000)))
    except Exception:
        tol = 15 * 60_000
    start = max(0, int(since_ms))
    end = int(since_ms) + int(tol) + 180_000  # +3분 여유

    trades = list_user_trades(symbol, start_time_ms=start, end_time_ms=end, limit=1000)
    if not trades:
        return None

    def _get(o: Dict[str, Any], *keys: str):
        for k in keys:
            if k in o and o[k] is not None:
                return o[k]
        return None

    closings: List[Dict[str, Any]] = []
    for t in trades:
        rp = _get(t, "realizedPnl", "realized_pnl", "realizedPNL", "realizedProfit", "realizedPnlUSDT")
        try:
            rp_f = float(rp) if rp not in (None, "") else 0.0
        except Exception:
            rp_f = 0.0
        if abs(rp_f) <= 1e-12:
            continue  # 실현 PnL 0이면 청산으로 보지 않음

        px = _get(t, "price", "avgPrice", "avg_price")
        try:
            px_f = float(px) if px not in (None, "") else 0.0
        except Exception:
            px_f = 0.0

        tm = _get(t, "time", "T", "transactTime", "transact_time")
        try:
            tm_i = int(tm)
        except Exception:
            try:
                tm_i = int(float(tm))
            except Exception:
                tm_i = None

        closings.append({"type": "USER_TRADE", "price": px_f, "time": tm_i, "realized": rp_f})

    if not closings:
        return None
    closings = [c for c in closings if c["time"] is not None and c["time"] >= start]
    if not closings:
        return None
    closings.sort(key=lambda x: x["time"])
    return closings[-1]

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
# helpers/binance_client.py

def cancel_orders_by_type(symbol: str, types: List[str]) -> int:
    """
    열린 주문 중 특정 타입(TAKE_PROFIT, STOP, STOP_MARKET 등)만 취소.
    - ID 키: orderId | order_id | (orig_)clientOrderId 모두 지원.
    """
    typesU = {t.upper() for t in types}
    cnt = 0
    try:
        orders = get_open_orders(symbol)
    except Exception:
        orders = []
    for o in orders or []:
        d = o if isinstance(o, dict) else _as_plain_dict(o)
        t = str((d.get("type") or "")).upper()
        if t not in typesU:
            continue
        oid = d.get("orderId") or d.get("order_id")
        ocid = d.get("origClientOrderId") or d.get("clientOrderId") or d.get("orig_client_order_id") or d.get("client_order_id")
        try:
            if oid is not None:
                cancel_order(symbol, order_id=int(oid))
                cnt += 1
            elif ocid:
                cancel_order(symbol, orig_client_order_id=str(ocid))
                cnt += 1
        except Exception:
            continue
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

def place_market_order(symbol: str, side: str, quantity: float, reduce_only: bool = False,
                       position_side_override: Optional[str] = None) -> Dict[str, Any]:
    client = _get_client()
    side = side.upper()
    ps = (position_side_override or _position_side_for(side, reduce_only=reduce_only))

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

    order_type = "MARKET"
    payload = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": str(q),
        "position_side": ps,
    }
    if _should_send_reduce_only(order_type, reduce_only):
        payload["reduce_only"] = True
    log_event("binance.order.request", **payload)
    try:
        resp = _safe_new_order(client, **payload)
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
        msg = str(getattr(e, "message", str(e))).lower()
        # 서버가 reduceonly가 불필요/금지라고 응답하면 reduce_only 제거 후 1회 재시도
        if "reduceonly" in msg and "not required" in msg:
            try:
                payload.pop("reduce_only", None)
                log_event("binance.order.retry_no_reduceonly", symbol=symbol, side=side, type=order_type)
                resp = _safe_new_order(client, **payload)
                raw = resp.data() if hasattr(resp, "data") else resp
                data = _to_plain(raw)
                return data if isinstance(data, dict) else {"raw": data}
            except Exception:
                pass
        logger.error("place_market_order error: %s", e)
        raise

def place_limit_order(symbol: str, side: str, quantity: float, price: float,
                      time_in_force: str = "GTC", reduce_only: bool = False,
                      post_only: bool = False) -> Dict[str, Any]:
    client = _get_client()
    side = side.upper()
    ps = _position_side_for(side, reduce_only=reduce_only)

    px = normalize_price_for_side(symbol, price, side)
    q = _quantize_qty(symbol, quantity, at_price=px)

    order_type = "LIMIT"
    payload = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,  # GTC | IOC | FOK | GTX(POST-ONLY)
        "price": _format_to_tick_str(symbol, px),
        "quantity": str(q),
        "position_side": ps,
    }
    if _should_send_reduce_only(order_type, reduce_only):
        payload["reduce_only"] = True

    log_event("binance.order.request", **payload)
    try:
        resp = _safe_new_order(client, **payload)
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
        msg = str(getattr(e, "message", str(e))).lower()
        if "reduceonly" in msg and "not required" in msg:
            try:
                payload.pop("reduce_only", None)
                log_event("binance.order.retry_no_reduceonly", symbol=symbol, side=side, type=order_type)
                resp = _safe_new_order(client, **payload)
                raw = resp.data() if hasattr(resp, "data") else resp
                data = _to_plain(raw)
                return data if isinstance(data, dict) else {"raw": data}
            except Exception:
                pass
        logger.error("place_limit_order error: %s", e)
        # POST-ONLY 의도일 때는 MARKET 폴백 금지
        if _LIMIT_FAILOVER_TO_MARKET and (not post_only):
            logger.info("Falling back to MARKET due to limit error for %s: %s", symbol, e)
            return place_market_order(symbol, side, quantity, reduce_only=reduce_only)
        raise

def place_take_profit(symbol: str, opp_side: str, quantity: float, tp_price: float, order_type: str = "LIMIT") -> Dict[str, Any]:
    client = _get_client()
    opp_side = opp_side.upper()
    ps = _position_side_for(opp_side, reduce_only=True)
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
    resp = _safe_new_order(client, **tp_payload)
    raw = resp.data() if hasattr(resp, "data") else resp
    data = _to_plain(raw)
    return data if isinstance(data, dict) else {"raw": data}

def place_stop_market(symbol: str, opp_side: str, quantity: float, sl_price: float) -> Dict[str, Any]:
    client = _get_client()
    opp_side = opp_side.upper()
    ps = _position_side_for(opp_side, reduce_only=True)
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
    sl_resp = _safe_new_order(client, **sl_payload)
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
    opp_side = opp_side.upper()
    ps = _position_side_for(opp_side, reduce_only=True)
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
    resp = _safe_new_order(client, **payload)
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
    """
    현재 미체결 주문 나열. **심볼 지정이 되면 심볼별 엔드포인트 우선**.
    SDK의 일부 구현은 빈 결과에서 IndexError를 던지므로 try/except로 우회.
    """
    client = _get_client()
    # 1) 심볼별 우선
    per_symbol_candidates = ["open_orders", "openOrders", "current_open_orders", "currentOpenOrders", "get_open_orders"]
    # 2) 전체(모든 심볼)
    all_candidates = ["current_all_open_orders", "allOpenOrders", "getAllOpenOrders", "query_current_all_open_orders", "queryCurrentAllOpenOrders"]

    # helper
    def _try(fn, with_symbol: bool):
        try:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()
        kwargs = {}
        if with_symbol and "symbol" in params and symbol:
            kwargs["symbol"] = symbol
        r = fn(**kwargs) if kwargs else fn()
        data = r.data() if hasattr(r, "data") else r
        if isinstance(data, list): return data
        if isinstance(data, dict):
            if isinstance(data.get("orders"), list): return data["orders"]
            return [data]
        if hasattr(data, "__iter__"): return list(data)
        return [_as_plain_dict(data)]

    # 1) 심볼 지정이 있으면 per-symbol 먼저
    if symbol:
        for name in per_symbol_candidates:
            fn = getattr(client.rest_api, name, None)
            if callable(fn):
                try:
                    out = _try(fn, with_symbol=True)
                    if isinstance(out, list): return out
                except Exception as e:
                    # noise 줄이기: debug 로그만
                    logger.info("get_open_orders %s(%s) failed: %s", name, symbol, e)
                    continue

    # 2) 전체 오더(메서드에 버그가 있으면 스킵)
    for name in all_candidates:
        fn = getattr(client.rest_api, name, None)
        if callable(fn):
            try:
                out = _try(fn, with_symbol=False)
                if isinstance(out, list): return out
            except Exception as e:
                msg = str(e)
                if "list index out of range" in msg.lower():
                    logger.debug("get_open_orders %s(all) benign fallback: %s", name, msg)
                else:
                    logger.info("get_open_orders %s(all) failed: %s", name, msg)
                continue
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
            # 가장 보수적인 엔드포인트부터 시도
            ["position_risk", "position_information", "position_information_v2", "position_information_v3"],
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
    # + add
    "list_all_orders",
    "find_recent_exit_fill",
]
