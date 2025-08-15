from __future__ import annotations

"""
helpers/binance_client.py – refactor #2 (2025-08-12, KST)

Target SDK: binance-sdk-derivatives-trading-usds-futures==1.0.0
Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
Repo path for this SDK family: binance-connector-python/clients/derivatives_trading_usds_futures

이 모듈은 USDⓈ-M Futures 모듈형 SDK를 래핑합니다.

핵심 기능
- 프로덕션/테스트넷 스위치 및 타임아웃/재시도/백오프 설정
- 심볼 필터 기반 가격/수량 정규화(tickSize/stepSize/minNotional)
- 계정 설정(포지션 모드/마진 타입/레버리지) 및 전체취소
- 마켓/리밋/브래킷(RO TP/SL) 주문 유틸
- 대시보드용 오버뷰(잔고/포지션) 조회

디자인 메모
- 모듈형 SDK는 자동생성 특성상 버전에 따라 메서드명이 달라질 수 있습니다.
  _call()에서 복수 후보를 시도해 1.0.0~1.x 사이 호환성을 높였습니다.
- 반환값은 JSON 직렬화 가능한 dict/list 위주로 통일했습니다(특히 get_overview()).
"""

import logging
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---- Optional utils integration ------------------------------------------------
try:  # 프로젝트 utils 우선 사용
    from .utils import get_secret  # type: ignore
except Exception:  # pragma: no cover
    def get_secret(name: str) -> Optional[str]:  # 최소 폴백
        return os.getenv(name)

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
    """내부 전용: 싱글톤 클라이언트 생성/반환."""
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
    """외부용 액세서(예: data_fetch에서 사용)."""
    return _get_client()


# ===============
# Call helpers
# ===============
def _pick(obj: Any, names: List[str]) -> Optional[Any]:
    """여러 후보 메서드명 중 사용 가능한 첫 번째를 선택."""
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn
    return None


def _call(obj: Any, cand_names: List[str], /, **kwargs):
    """_pick으로 찾은 메서드를 호출. 실패 시 AttributeError."""
    fn = _pick(obj, cand_names)
    if not fn:
        raise AttributeError(f"None of {cand_names} available on {type(obj).__name__}")
    return fn(**kwargs)

def get_open_orders(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    client = _get_client()
    try:
        if symbol:
            r = _call(client.rest_api, ["open_orders", "openOrders"], symbol=symbol)
        else:
            r = _call(client.rest_api, ["open_orders", "openOrders"])
        data = r.data() if hasattr(r, "data") else r
        if isinstance(data, list):
            return data
        return data.get("orders", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.error("get_open_orders error: %s", e)
        return []



# ==================================
# Exchange info + symbol filter util
# ==================================
_symbol_filters_cache: Dict[str, Dict[str, Any]] = {}


def _to_decimal(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Binance 규칙상 올림 금지 → step 배수로 내림."""
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step


def load_symbol_filters(symbol: str) -> Dict[str, Any]:
    """심볼 필터 로딩: {tickSize, stepSize, minQty, maxQty, minNotional, raw} (캐시 포함)."""
    symbol = symbol.upper()
    if symbol in _symbol_filters_cache:
        return _symbol_filters_cache[symbol]

    client = _get_client()
    resp = _call(client.rest_api, ["exchange_information", "exchangeInformation"])  # SDK naming guard
    data = resp.data() if hasattr(resp, "data") else resp

    def to_dict(obj):
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):  # pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):        # pydantic v1
            return obj.dict()
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
    not_f   = fmap.get("NOTIONAL") or {}

    tick = _to_decimal(price_f.get("tickSize") or getattr(price_f, "tickSize", "0.01"))
    step = _to_decimal(lot_f.get("stepSize")   or getattr(lot_f,   "stepSize", "0.001"))
    min_qty = _to_decimal(lot_f.get("minQty")  or getattr(lot_f,   "minQty",   "0.0"))
    max_qty = _to_decimal(lot_f.get("maxQty")  or getattr(lot_f,   "maxQty",   "0.0"))
    min_notional = _to_decimal(not_f.get("notional") or getattr(not_f, "notional", "5"))

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


def ensure_min_notional(symbol: str, qty: float, price: float, filters: Optional[Dict[str, Any]] = None) -> float:
    """최소 명목가치(minNotional)와 수량 스텝(stepSize)을 만족하도록 수량을 보정."""
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
    """tickSize에 맞춰 가격을 내림 정규화."""
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


def get_overview() -> Dict[str, List[Dict[str, Any]]]:
    """
    대시보드용 오버뷰를 반환합니다(JSON 직렬화 가능).
    {
      "balances": [{"asset":"USDT","balance":123.4,"unrealizedPnL":0.0}, ...],
      "positions": [{"symbol":"BTCUSDT","positionAmt":0.01,"entryPrice":..., ...}, ...]
    }
    """
    client = _get_client()

    # balances
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

    # positions
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
    """DELETE /fapi/v1/allOpenOrders — 심볼 내 미체결 전량 취소."""
    client = _get_client()
    resp = _call(client.rest_api, ["cancel_all_open_orders", "cancelAllOpenOrders"], symbol=symbol)
    data = resp.data() if hasattr(resp, "data") else resp
    msg = (data.get("msg") if isinstance(data, dict) else getattr(data, "msg", "")) or ""
    code = (data.get("code") if isinstance(data, dict) else getattr(data, "code", None))
    logger.info("Bulk-cancel %s: code=%s msg=%r additional_properties=%s", symbol, code or "", msg, getattr(resp, "additional_properties", {}))
    return data if isinstance(data, dict) else {"code": code, "msg": msg}


def set_position_mode(mode: str = "ONEWAY") -> None:
    """POST /fapi/v1/positionSide/dual — dualSidePosition=true → HEDGE, false → ONEWAY."""
    client = _get_client()
    mode = (mode or "").upper()
    dual = True if mode == "HEDGE" else False
    try:
        _call(
            client.rest_api,
            ["change_position_mode", "changePositionMode", "change_position_side_dual"],
            dual_side_position=str(dual).lower(),
        )
        logger.info("Position mode set to %s", mode)
    except Exception as e:
        logger.error("Position mode set error: %s", getattr(e, "message", str(e)))


def set_margin_type(symbol: str, margin_type: str = "ISOLATED") -> None:
    """POST /fapi/v1/marginType — 심볼별 교차/격리."""
    client = _get_client()
    mt = (margin_type or "").upper()
    try:
        _call(client.rest_api, ["change_margin_type", "changeMarginType"], symbol=symbol, margin_type=mt)
        logger.info("Margin type set to %s for %s", mt, symbol)
    except Exception as e:
        logger.error("Margin type set error: %s", getattr(e, "message", str(e)))


def set_leverage(symbol: str, leverage: int) -> None:
    """POST /fapi/v1/leverage — 심볼별 레버리지."""
    client = _get_client()
    try:
        resp = _call(
            client.rest_api,
            ["change_initial_leverage", "changeInitialLeverage"],
            symbol=symbol,
            leverage=int(leverage),
        )
        data = resp.data() if hasattr(resp, "data") else resp
        logger.info("Leverage set to %sx for %s: %s", leverage, symbol, data)
    except Exception as e:
        logger.error("Leverage set error: %s", getattr(e, "message", str(e)))


# ==================
# Order placement
# ==================
def _position_side() -> str:
    """
    포지션 사이드 파라미터.
    - ONEWAY 모드에선 생략 가능하지만 BOTH 사용이 무난.
    - HEDGE 모드에서는 BUY/SELL 구분하여 LONG/SHORT 지정할 수도 있음(여기선 BOTH 유지).
    """
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


def _quantize_qty(symbol: str, qty: float, at_price: float) -> float:
    """수량을 stepSize 및 minNotional에 맞춰 보정."""
    f = load_symbol_filters(symbol)
    return ensure_min_notional(symbol, qty, price=at_price, filters=f)


def place_market_order(symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
    """
    마켓 주문(BUY/SELL).
    - reduce_only: True면 청산 전용
    - position_side=BOTH (원웨이)
    """
    client = _get_client()
    side = side.upper()
    ps = _position_side()

    # 수량 정규화 (실시장가 기반)
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
    try:
        resp = _call(client.rest_api, ["new_order", "newOrder"], **payload)
        return resp.data() if hasattr(resp, "data") else resp
    except Exception as e:
        logger.error("place_market_order error: %s", e)
        raise

def place_limit_order(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    time_in_force: str = "GTC",
    reduce_only: bool = False,
) -> Dict[str, Any]:
    """
    리밋 주문.
    - 가격 tickSize 내림 정규화
    - 수량 stepSize/minNotional 충족
    """
    client = _get_client()
    side = side.upper()
    ps = _position_side()

    px = normalize_price_with_mode(symbol, price)
    q = _quantize_qty(symbol, quantity, at_price=px)

    payload = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "time_in_force": time_in_force,
        "price": str(px),
        "quantity": str(q),
        "reduce_only": bool(reduce_only),
        "position_side": ps,
    }
    try:
        resp = _call(client.rest_api, ["new_order", "newOrder"], **payload)
        return resp.data() if hasattr(resp, "data") else resp
    except Exception as e:
        logger.error("place_limit_order error: %s", e)
        raise
# helpers/binance_client.py
def place_bracket_orders(
    symbol: str,
    side: str,            # 엔트리 방향 (BUY=롱, SELL=숏)
    quantity: float,      # 체결 수량(신규 포지션 수량)
    take_profit: float,   # TP 레벨(가격)
    stop_loss: float,     # SL 레벨(가격)
) -> Dict[str, Any]:
    """
    엔트리 직후 브래킷(리듀스온리) 생성.
      - TP: TAKE_PROFIT (LIMIT)  ← 기존 TAKE_PROFIT_MARKET에서 변경
      - SL: STOP_MARKET (권장: 미충족 위험 줄이기)
    주의: reduce_only=True, position_side=BOTH(원웨이) 유지.
    """
    client = _get_client()
    side = side.upper()
    opp = "SELL" if side == "BUY" else "BUY"
    ps = _position_side()

    # 가격을 거래소 tickSize에 맞게 정규화
    tp_price = normalize_price_with_mode(symbol, float(take_profit))
    sl_price = normalize_price_with_mode(symbol, float(stop_loss))

    # 수량을 stepSize/minNotional에 맞게 보정(보수적으로 TP 가격 기준)
    f = load_symbol_filters(symbol)
    qty_q = ensure_min_notional(symbol, float(quantity), price=tp_price, filters=f)

    out: Dict[str, Any] = {"take_profit": None, "stop_loss": None}

    # --- TAKE_PROFIT (LIMIT) ---
    # Futures의 LIMIT형 TP는 price(발주시점 지정가) + stop_price(트리거)가 모두 필요합니다.
    # working_type을 MARK_PRICE로 주면 마크가격 기준으로 트리거됩니다(지원 환경에서만 적용).
    try:
        tp_payload = {
            "symbol": symbol,
            "side": opp,
            "type": "TAKE_PROFIT",          # ★ LIMIT TP
            "time_in_force": "GTC",
            "price": str(tp_price),         # 체결을 시도할 지정가
            "stop_price": str(tp_price),    # 트리거 가격(=TP 레벨과 동일하게 세팅)
            "quantity": str(qty_q),
            "reduce_only": True,
            "position_side": ps,
            # 선택 필드(지원되는 SDK/엔드포인트에서만 유효)
            "working_type": "MARK_PRICE",
            "price_protect": True,
        }
        tp_resp = _call(client.rest_api, ["new_order", "newOrder"], **tp_payload)
        out["take_profit"] = tp_resp.data() if hasattr(tp_resp, "data") else tp_resp
    except Exception as e:
        logger.info("TP LIMIT failed: %s", e)

    # --- STOP_MARKET ---
    try:
        sl_payload = {
            "symbol": symbol,
            "side": opp,
            "type": "STOP_MARKET",
            "stop_price": str(sl_price),
            "quantity": str(qty_q),
            "reduce_only": True,
            "position_side": ps,
            "working_type": "MARK_PRICE",
            "price_protect": True,
        }
        sl_resp = _call(client.rest_api, ["new_order", "newOrder"], **sl_payload)
        out["stop_loss"] = sl_resp.data() if hasattr(sl_resp, "data") else sl_resp
    except Exception as e:
        logger.info("SL MARKET failed: %s", e)

    return out



def build_entry_and_brackets(
    symbol: str,
    side: str,
    quantity: float,
    target_price: float,
    stop_price: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(호환 유지) 마켓 진입 → 브래킷(RO) 생성 일괄 호출."""
    entry = place_market_order(symbol, side, quantity)
    bracket = place_bracket_orders(symbol, side, quantity, target_price, stop_price)
    return entry, bracket


# ==================
# Position helpers
# ==================
def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    """현재 심볼 포지션 1개(있는 경우)만 리턴. 없으면 None."""
    client = _get_client()
    try:
        pos = _call(
            client.rest_api,
            ["position_information_v3", "position_information_v2", "position_information", "position_risk"],
            symbol=symbol,
        )
        pdata = pos.data() if hasattr(pos, "data") else pos
        items = getattr(pdata, "positions", None) or (pdata if isinstance(pdata, list) else pdata.get("positions", []))
        for p in items or []:
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
