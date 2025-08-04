from __future__ import annotations

import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np  # 시뮬/백테스트용

# NOTE:
# 본 파일은 Binance 모듈형 SDK(USDS-M)로의 마이그레이션을 고려해
# helpers/binance_client.py의 래퍼 함수를 사용하는 구조를 유지했습니다.
# => signals.py는 거래 로직만 담당하고, 실제 REST 호출은 binance_client.py가
#    binance-sdk-derivatives-trading-usds-futures==1.0.0을 사용하도록 변경하세요.

from .binance_client import (
    _client as _get_client,
    ensure_min_notional,
    get_overview,
    get_position,
    load_symbol_filters,
    normalize_price_with_mode,
    normalize_qty,
    place_bracket_orders,
    place_limit_order,
    place_market_order,  # (FIX) market 진입 사용
    wait_order_filled,
    set_leverage,
    set_margin_type,
    set_position_mode,
    cancel_open_orders,
)
from .data_fetch import fetch_data, fetch_multitime_indicators, fetch_funding_rate, fetch_orderbook
from .predictor import get_gemini_prediction, should_predict
from .utils import LOG_DIR, gcs_append_csv_row, gcs_enabled, gcs_read_recent_csvs

__all__ = ["generate_signal", "manage_trade", "get_trade_history", "auto_trade", "backtest_strategy"]

# --- Constants ---
MIN_PROB: float = float(os.getenv("MIN_PROB", "0.65"))
RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.1"))  # 10% 노출 per trade
PAPER_TRADING: bool = os.getenv("EXECUTE_TRADES", "").lower() != "true"
TIMEFRAMES: list[str] = ["5m", "1h", "4h", "1d"]
TRADE_LOG_FILE: Path = LOG_DIR / "trades.csv"
BALANCE_LOG_FILE: Path = LOG_DIR / "balance.csv"

# === 진입 관련 파라미터 (환경변수로 조절 가능) ===
ENTRY_TIMEOUT_SEC: int = int(os.getenv("ENTRY_TIMEOUT_SEC", "8"))
ENTRY_MAX_RETRIES: int = int(os.getenv("ENTRY_MAX_RETRIES", "1"))
ENTRY_MAX_SLIPPAGE_BPS: float = float(os.getenv("ENTRY_MAX_SLIPPAGE_BPS", "5"))  # 1bp=0.01%
# fallback: "skip" | "market" | "marketable_limit" | "adaptive"
ENTRY_FALLBACK_MODE: str = os.getenv("ENTRY_FALLBACK_MODE", "adaptive").lower()

TRADES_HEADER: List[str] = [
    "timestamp",
    "symbol",
    "side",
    "qty",
    "entry_price",
    "exit_price",
    "realized_pnl",
    "prob",
    "target_price",
    "stop_loss",
    "reason",
    "fee",
    "status",
    "entry_type",
    "attempts",
    "slippage_bp",
    "book_spread_bp",
]
BALANCE_HEADER: List[str] = ["timestamp", "balance"]


def _calc_tp_sl(price_now: float, atr: float, support: float, resistance: float, side: str) -> Tuple[float, float]:
    """Return (tp, sl) based on support/resistance + ATR."""
    if side == "long":
        tp = resistance * 0.99 if resistance > 0 else price_now + max(1.5 * atr, 0.015 * price_now)
        sl = support * 1.01 if support > 0 else price_now - max(atr, 0.010 * price_now)
    elif side == "short":
        tp = support * 1.01 if support > 0 else price_now - max(1.5 * atr, 0.015 * price_now)
        sl = resistance * 0.99 if resistance > 0 else price_now + max(atr, 0.010 * price_now)
    else:
        tp = sl = 0.0
    return float(tp), float(sl)


def simulate_fill_price(
    side: str,
    qty: float,
    orderbook: Dict[str, Any],
    max_levels: int = 5
) -> Optional[float]:
    """
    간단한 체결 시뮬레이션: 지정한 수량을 상위 N레벨로 소진했을 때
    가중 평균 체결가를 추정. 부족하면 None.
    """
    try:
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return None
        levels = orderbook["asks"] if side.lower() in ("buy", "long") else orderbook["bids"]
        # 매수는 asks 오름차순, 매도는 bids 내림차순이 이미 정렬되어 있다고 가정
        remaining = float(qty)
        notional = 0.0
        filled = 0.0
        for i, (px, q) in enumerate(levels[:max_levels]):
            price = float(px)
            avail = float(q)
            take = min(remaining, avail)
            notional += take * price
            filled += take
            remaining -= take
            if remaining <= 1e-12:
                break
        if remaining > 1e-12:
            return None
        return notional / max(filled, 1e-12)
    except Exception:
        return None


def place_entry_with_fallback(
    symbol: str,
    action: str,            # "buy" | "sell"
    qty: float,
    preferred_price: float  # 모델/전략이 제안한 엔트리
) -> Dict[str, Any]:
    """
    1) 우선 지정가로 시도 → timeout 시 취소
    2) ENTRY_FALLBACK_MODE 에 따라 재시도:
       - skip: 중단
       - marketable_limit: 슬리피지 상한 내에서 교차 지정가
       - market: 시장가 (단, 슬리피지 상한 체크 후)
       - adaptive(권장): 오더북/스프레드/잔량 조건 충족 시 marketable_limit, 아니면 skip
    """
    result: Dict[str, Any] = {
        "entry_type": "limit",
        "attempts": 0,
        "filled": None,
        "entry_resp": None,
        "error": None,
        "slippage_bp": None,
        "book_spread_bp": None,
    }

    if PAPER_TRADING:
        # 드라이런에서는 체결 시나리오만 기록
        result["entry_type"] = "paper"
        result["attempts"] = 1
        result["filled"] = {"status": "FILLED", "price": preferred_price}
        return result

    # 0) 사전 계산
    side = "buy" if action.lower() == "buy" else "sell"
    max_slip_frac = ENTRY_MAX_SLIPPAGE_BPS / 10000.0  # bps → frac

    # 1) 지정가
    try:
        entry_resp = place_limit_order(symbol, action, qty, preferred_price)
        result["entry_resp"] = entry_resp
    except Exception as exc:
        logging.error(f"[entry] limit error: {exc}")
        result["error"] = f"limit_error:{exc}"
        return result

    result["attempts"] += 1
    # 대기/취소 (wait_order_filled는 타임아웃 시 cancel까지 시도)
    order_id, client_id = _extract_ids_from_entry(entry_resp)
    filled = wait_order_filled(
        symbol,
        order_id=order_id,
        client_order_id=client_id,
        max_checks=max(1, int(ENTRY_TIMEOUT_SEC)),  # 1s 간격 가정 시
        sleep_sec=1.0,
        timeout_cancel=True,
    )
    result["filled"] = filled
    if filled and str(filled.get("status", "")).upper() in ("FILLED", "PARTIALLY_FILLED"):
        result["entry_type"] = "limit"
        return result

    # 2) Fallback 결정
    mode = ENTRY_FALLBACK_MODE
    if mode not in ("skip", "market", "marketable_limit", "adaptive"):
        mode = "adaptive"

    # 오더북/스프레드 측정
    ob = fetch_orderbook(symbol, limit=10) or {}
    try:
        best_bid = float(ob["bids"][0][0]) if ob.get("bids") else 0.0
        best_ask = float(ob["asks"][0][0]) if ob.get("asks") else 0.0
        mid = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else 0.0
        spread_bp = (best_ask - best_bid) / mid * 10000 if mid > 0 else None
        result["book_spread_bp"] = spread_bp
    except Exception:
        best_bid = best_ask = mid = 0.0
        spread_bp = None

    def within_slippage_cap(sim_px: Optional[float]) -> bool:
        if sim_px is None or mid <= 0:
            return False
        slip_bp = abs(sim_px - preferred_price) / preferred_price * 10000
        result["slippage_bp"] = slip_bp
        return slip_bp <= ENTRY_MAX_SLIPPAGE_BPS + 1e-9

    # 모드별 분기
    do_marketable_limit = False
    do_market = False

    if mode == "skip":
        return result

    elif mode == "market":
        # 시장가도 상한 위반 시 Skip
        sim_px = simulate_fill_price(side, qty, ob, max_levels=5)
        if within_slippage_cap(sim_px):
            do_market = True

    elif mode == "marketable_limit":
        sim_px = simulate_fill_price(side, qty, ob, max_levels=5)
        if within_slippage_cap(sim_px):
            do_marketable_limit = True

    else:  # adaptive
        # 스프레드가 너무 넓으면 Skip
        if spread_bp is None or spread_bp > max(ENTRY_MAX_SLIPPAGE_BPS, 7.0):
            return result
        sim_px = simulate_fill_price(side, qty, ob, max_levels=5)
        if within_slippage_cap(sim_px):
            # 우선 marketable-limit (가격 상한 보호)
            do_marketable_limit = True
        else:
            return result

    # 3) 재시도
    attempts = 0
    while attempts < ENTRY_MAX_RETRIES:
        attempts += 1
        result["attempts"] += 1
        try:
            if do_marketable_limit:
                if side == "buy":
                    # 매수: 현재 호가 위로 상한 캡 (가격 상한)
                    price_cap = preferred_price * (1.0 + max_slip_frac)
                    # 스프레드 기반 보정(캡이 호가보다 너무 낮지 않게)
                    if best_ask > 0:
                        price_cap = max(price_cap, best_ask)
                    entry_resp = place_limit_order(symbol, action, qty, price_cap)
                    result["entry_type"] = "marketable_limit"
                else:
                    # 매도: 현재 호가 아래로 상한 캡 (가격 하한)
                    price_cap = preferred_price * (1.0 - max_slip_frac)
                    if best_bid > 0:
                        price_cap = min(price_cap, best_bid)
                    entry_resp = place_limit_order(symbol, action, qty, price_cap)
                    result["entry_type"] = "marketable_limit"

            elif do_market:
                entry_resp = place_market_order(symbol, action, qty)
                result["entry_type"] = "market"

            else:
                break  # 안전 장치

            result["entry_resp"] = entry_resp
            order_id, client_id = _extract_ids_from_entry(entry_resp)
            filled = wait_order_filled(
                symbol,
                order_id=order_id,
                client_order_id=client_id,
                max_checks=max(1, int(ENTRY_TIMEOUT_SEC)),
                sleep_sec=1.0,
                timeout_cancel=True,
            )
            result["filled"] = filled
            if filled and str(filled.get("status", "")).upper() in ("FILLED", "PARTIALLY_FILLED"):
                return result

        except Exception as exc:
            logging.error(f"[entry] fallback error: {exc}")
            result["error"] = f"fallback_error:{exc}"
            # 다음 루프에서 재시도

    # 재시도 종료 → 미체결/취소 상태로 반환
    return result


def _append_csv(file_path: Path, data: Dict[str, Any]) -> None:
    df = pd.DataFrame([data])
    if file_path.exists():
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        cols = TRADES_HEADER if "trades" in file_path.name else BALANCE_HEADER
        df = df.reindex(columns=cols)
        df.to_csv(file_path, index=False)


def record_trade(entry: Dict[str, Any]) -> None:
    # Local file
    try:
        _append_csv(TRADE_LOG_FILE, entry)
    except Exception as exc:
        logging.error(f"Failed to record trade (local): {exc}")

    # GCS
    try:
        if gcs_enabled():
            gcs_append_csv_row("trades", TRADES_HEADER, entry)
    except Exception as exc:
        logging.error(f"Failed to record trade (GCS): {exc}")

    # JSON log
    try:
        logging.info(json.dumps({"trade": entry}))
    except Exception:
        pass


def record_balance(balance: float, timestamp: str) -> None:
    entry = {"timestamp": timestamp, "balance": balance}

    # Local file
    try:
        _append_csv(BALANCE_LOG_FILE, entry)
    except Exception as exc:
        logging.error(f"Failed to record balance (local): {exc}")

    # GCS
    try:
        if gcs_enabled():
            gcs_append_csv_row("balance", BALANCE_HEADER, entry)
    except Exception as exc:
        logging.error(f"Failed to record balance (GCS): {exc}")

    # JSON log
    try:
        logging.info(json.dumps({"balance": entry}))
    except Exception:
        pass


def _compute_risk_scalar_with_daily_alignment(
    symbol: str, base_interval: str, prob: float, direction: str
) -> float:
    try:
        df_day = fetch_data(symbol, interval="1d", ohlcv_limit=250, include_orderbook=False)["ohlcv"]
        if df_day is None or df_day.empty:
            return 1.0
        sma50 = df_day["close"].rolling(50).mean().iloc[-1]
        sma200 = df_day["close"].rolling(200).mean().iloc[-1]
        if pd.isna(sma50) or pd.isna(sma200) or sma200 == 0:
            return 1.0
        diff = float(sma50 - sma200)
        daily_trend = 1 if diff > 0 else (-1 if diff < 0 else 0)
        pred_dir = 1 if direction == "long" else (-1 if direction == "short" else 0)
        daily_conf = min(abs(diff) / float(sma200), 1.0)
        if pred_dir == 0 or daily_trend == 0:
            base = 1.0
        elif pred_dir == daily_trend:
            base = 1.0 + 0.2 * daily_conf
        else:
            base = 0.9 - 0.4 * daily_conf
        extra = max(min((prob - MIN_PROB) / max(1e-6, 1 - MIN_PROB), 1.0), 0.0) * 0.2
        scalar = max(min(base + extra, 1.2), 0.5)
        return float(scalar)
    except Exception:
        return 1.0


def generate_signal(symbol: str = "BTCUSDT", balance: float = 1000.0) -> Dict[str, Any]:
    # Fetch base TF data
    base_interval = TIMEFRAMES[0]
    data = fetch_data(symbol, interval=base_interval, include_orderbook=True)
    df = data["ohlcv"]

    if not should_predict(df):
        return {"action": "hold"}

    extra_ind = fetch_multitime_indicators(symbol, TIMEFRAMES)
    last_row = df.iloc[-1]
    extra_ind["orderbook_imbalance"] = float(last_row.get("orderbook_imbalance", 0.0))
    extra_ind["orderbook_spread"] = float(last_row.get("orderbook_spread", 0.0))

    pred = get_gemini_prediction(
        symbol=symbol,
        df_5m=df,
        extra_indicators=extra_ind,
        sentiment_score=float(last_row.get("x_sentiment", 0.0)),
        funding_rate_pct=float(data.get("funding_rate_pct", 0.0)),
        times=data.get("times"),
    )

    direction = pred.get("direction", "hold")
    prob = float(pred.get("prob", 0.5))
    price_now = float(df["close"].iloc[-1])
    if direction == "hold" or prob < MIN_PROB:
        return {"action": "hold", "prob": prob}

    # Risk sizing
    atr_key = f"ATR_{base_interval}"
    atr = float(extra_ind.get(atr_key, 0.0))

    # Position qty – 기본 예시
    risk_amount = balance * RISK_PER_TRADE
    qty = risk_amount / price_now

    # 심볼 필터 적용
    um = _get_client()
    filters = load_symbol_filters(um, symbol)
    qty = float(
        ensure_min_notional(
            normalize_qty(Decimal(str(qty)), filters["stepSize"], filters["minQty"]),
            Decimal(str(price_now)),
            filters["stepSize"],
            filters["minNotional"],
        )
    )

    # --- TP/SL 산출(코드 기반) ---
    support = float(pred.get("support", 0.0))
    resistance = float(pred.get("resistance", 0.0))
    tp, sl = _calc_tp_sl(price_now, atr, support, resistance, direction)

    action = "buy" if direction == "long" else "sell"

    return {
        "action": action,
        "qty": qty,
        "entry": price_now,
        "tp": tp,
        "sl": sl,
        "prob": prob,
        "reason": pred.get("reasoning", ""),
    }


def _extract_ids_from_entry(entry_resp: Optional[Dict[str, Any]]) -> Tuple[Optional[Union[int, str]], Optional[str]]:
    if not isinstance(entry_resp, dict):
        return None, None
    order_id = entry_resp.get("orderId") or entry_resp.get("orderID") or entry_resp.get("order_id")
    client_id = entry_resp.get("clientOrderId") or entry_resp.get("newClientOrderId")
    raw = entry_resp.get("__raw__")
    if (order_id is None or client_id is None) and isinstance(raw, dict):
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        order_id = order_id or data.get("orderId")
        client_id = client_id or data.get("clientOrderId") or data.get("newClientOrderId")
    return order_id, client_id


def manage_trade(symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
    """
    엔트리 체결 로직을 place_entry_with_fallback() 기반으로 구성.
    - 지정가(post-only 지향) 우선 시도
    - 타임아웃 시 오더 취소
    - 오더북/스프레드/잔량 조건 충족 & 슬리피지 상한(bps) 이내면
      marketable-limit(또는 market)로 제한적 재시도
    - 조건 미충족 시 Skip
    - 엔트리 타입/시도 횟수/슬리피지/스프레드 bp 등 로그 필드 확장
    """
    # --- Wallet balance ---
    bal_df, _ = get_overview()
    if bal_df is None or bal_df.empty:
        logging.warning("Balance dataframe empty – skip trade.")
        return {"action": "hold", "reason": "empty_balance_df"}

    try:
        usdt_row = bal_df[bal_df["asset"] == "USDT"] if "asset" in bal_df.columns else pd.DataFrame()
        balance = 0.0
        if not usdt_row.empty:
            for col in ("availableBalance", "walletBalance", "balance", "crossWalletBalance"):
                if col in usdt_row.columns:
                    balance = float(usdt_row[col].iloc[0])
                    break
    except Exception:
        logging.exception("Wallet balance parse error")
        return {"action": "hold", "reason": "balance_parse_exc"}

    if balance == 0:
        logging.warning("Balance is zero – skip trade.")
        return {"action": "hold", "reason": "zero_balance"}

    # --- Check and cancel existing open orders ---
    try:
        if not PAPER_TRADING:
            cancel_open_orders(symbol)
    except Exception as exc:
        logging.error(f"Failed to cancel open orders for {symbol}: {exc}")

    # --- Fetch price_now first for leverage calculation ---
    data = fetch_data(symbol, interval="1h", include_orderbook=False)
    df = data["ohlcv"]
    price_now = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0
    if price_now == 0:
        logging.warning("price_now is zero – skip trade.")
        return {"action": "hold", "reason": "zero_price_now"}

    # --- Trading params ---
    try:
        set_position_mode(dual_side=False)
        set_margin_type(symbol, margin_type=os.getenv("MARGIN_TYPE", "ISOLATED"))

        # 멀티 타임프레임 지표 기반 변동성/레버리지 산출
        extra_ind = fetch_multitime_indicators(symbol, ["1h", "4h"])
        vol_1h = extra_ind.get("volatility_1h", 0.0)
        vol_4h = extra_ind.get("volatility_4h", 0.0) / 4  # 4h → 시간당 환산
        vol = max(vol_1h, vol_4h)
        vol_ratio = vol / price_now if price_now > 0 else 0.0  # 예: 0.02 = 2%
        # 변동성 ↑ 일수록 레버리지 ↓   (vol_ratio 0 → 5×, 0.05 → 4×, 0.1 → 3×)
        leverage = max(3, min(5, int(round(5 - 20 * vol_ratio))))
        set_leverage(symbol, leverage=leverage)
    except Exception as exc:
        logging.error(f"Failed to set trading params for {symbol}: {exc}")

    # --- Generate signal ---
    signal = generate_signal(symbol, balance)
    if signal.get("action") == "hold":
        return None

    # --- Execute entry (with fallback) ---
    result: Dict[str, Any] = signal.copy()
    now_ts = pd.Timestamp.utcnow().isoformat()
    result["timestamp"] = now_ts

    # 엔트리 실행 (PAPER_TRADING인 경우 함수 내부에서 모의 체결 처리)
    exec_res = place_entry_with_fallback(
        symbol=symbol,
        action=signal["action"],      # "buy" or "sell"
        qty=signal["qty"],
        preferred_price=signal["entry"]
    )

    # 결과 병합
    result.update({
        "entry_type": exec_res.get("entry_type"),        # limit/marketable_limit/market/paper/None
        "attempts": exec_res.get("attempts"),
        "entry_order": exec_res.get("entry_resp"),
        "entry_fill": exec_res.get("filled"),
        "slippage_bp": exec_res.get("slippage_bp"),
        "book_spread_bp": exec_res.get("book_spread_bp"),
        "error": exec_res.get("error"),
    })

    filled = exec_res.get("filled")
    filled_status = str(filled.get("status", "")) if isinstance(filled, dict) else ""
    is_filled = filled_status.upper() in ("FILLED", "PARTIALLY_FILLED")

    # --- Fee estimate (간단 추정; 실제 수수료는 거래소 체결 내역 우선) ---
    # Binance Futures 일반 등급 가정: maker 2bp(0.02%), taker 4bp(0.04%)
    # ※ 수수료는 레버리지에 곱하지 않습니다 (선물은 명목가치 기준 수수료).
    fee_maker_bps = float(os.getenv("FEE_MAKER_BPS", "2.0"))
    fee_taker_bps = float(os.getenv("FEE_TAKER_BPS", "4.0"))
    fee_rate = (fee_taker_bps / 10000.0) if result.get("entry_type") in ("market", "marketable_limit") else (fee_maker_bps / 10000.0)

    try:
        fill_price = None
        if isinstance(filled, dict):
            # 다양한 응답 포맷 대비
            fill_price = (
                float(filled.get("price")) if filled.get("price") not in (None, "", 0) else
                float(filled.get("avgPrice")) if filled.get("avgPrice") not in (None, "", 0) else
                float(signal.get("entry", 0.0))
            )
        else:
            fill_price = float(signal.get("entry", 0.0))
        if not isinstance(fill_price, (int, float)) or fill_price <= 0:
            fill_price = float(signal.get("entry", 0.0))

        notional = float(signal["qty"]) * float(fill_price)
        result["fee"] = notional * fee_rate
    except Exception as exc:
        logging.error(f"Fee calc error: {exc}")
        result["fee"] = 0.0

    # --- Unfilled / Cancelled 처리 ---
    if not PAPER_TRADING and not is_filled:
        # 미체결이거나 취소/실패
        trade_entry = {
            "timestamp": now_ts,
            "symbol": symbol,
            "side": signal["action"],
            "qty": signal["qty"],
            "entry_price": signal.get("entry"),
            "exit_price": None,
            "realized_pnl": None,
            "prob": signal.get("prob"),
            "target_price": signal.get("tp"),
            "stop_loss": signal.get("sl"),
            "reason": signal.get("reason") or (result.get("error") or "unfilled"),
            "fee": result.get("fee", 0.0),
            "status": filled_status if filled_status else "UNFILLED",
            "entry_type": result.get("entry_type"),
            "attempts": result.get("attempts"),
            "slippage_bp": result.get("slippage_bp"),
            "book_spread_bp": result.get("book_spread_bp"),
        }
        try:
            record_trade(trade_entry)
        except Exception as exc:
            logging.error(f"Trade record error(unfilled): {exc}")
        return result

    # --- Place TP/SL (filled 또는 paper) ---
    if is_filled or PAPER_TRADING:
        try:
            tp, sl = signal.get("tp"), signal.get("sl")
            if tp and sl and not PAPER_TRADING:
                # 부분 체결 보호: 필요 시 실행 수량 사용
                try:
                    exec_qty = float(filled.get("executedQty")) if (isinstance(filled, dict) and filled.get("executedQty")) else float(signal["qty"])
                except Exception:
                    exec_qty = float(signal["qty"])
                bracket = place_bracket_orders(symbol, signal["action"], exec_qty, tp, sl)
                result["bracket_order"] = bracket
            elif not tp or not sl:
                logging.error("TP/SL values missing; skip bracket orders.")
        except Exception as exc:
            logging.error(f"Bracket orders error: {exc}")

    # --- Record trade intent/result ---
    try:
        trade_entry = {
            "timestamp": now_ts,
            "symbol": symbol,
            "side": signal["action"],
            "qty": signal["qty"],
            "entry_price": signal.get("entry"),
            "exit_price": None,
            "realized_pnl": None,
            "prob": signal.get("prob"),
            "target_price": signal.get("tp"),
            "stop_loss": signal.get("sl"),
            "reason": signal.get("reason"),
            "fee": result.get("fee", 0.0),
            "status": filled_status if filled_status else ("FILLED" if PAPER_TRADING else "PENDING"),
            "entry_type": result.get("entry_type"),
            "attempts": result.get("attempts"),
            "slippage_bp": result.get("slippage_bp"),
            "book_spread_bp": result.get("book_spread_bp"),
        }
        record_trade(trade_entry)
    except Exception as exc:
        logging.error(f"Trade record error: {exc}")

    return result


def get_trade_history(symbol: str, max_lines: int = 100) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []

    # Local
    try:
        if TRADE_LOG_FILE.exists():
            local_df = pd.read_csv(TRADE_LOG_FILE)
            history.extend(local_df.tail(max_lines).to_dict("records"))
    except Exception as exc:
        logging.error(f"Local history read error: {exc}")

    # GCS
    try:
        if gcs_enabled():
            gcs_file = gcs_read_recent_csvs("trades")
            if gcs_file:
                gcs_df = pd.read_csv(gcs_file)
                existing_ts = {h.get("timestamp") for h in history}
                new_records = [
                    r for r in gcs_df.to_dict("records") if r.get("timestamp") not in existing_ts
                ]
                history.extend(new_records)
    except Exception as exc:
        logging.error(f"GCS history read error: {exc}")

    return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:max_lines]


def auto_trade(event: dict, context: Any) -> None:
    symbols = ["BTCUSDT", "ETHUSDT"]
    for sym in symbols:
        try:
            manage_trade(sym)
        except Exception as exc:
            logging.error(f"Auto trade error for {sym}: {exc}")


# 간단 백테스트(모의) – 참고용
def backtest_strategy(symbol: str = "BTCUSDT", initial_balance: float = 1000.0, num_days: int = 365) -> Dict[str, float]:
    df_day = fetch_data(symbol, interval="1d", ohlcv_limit=num_days, include_orderbook=False)["ohlcv"]
    if df_day is None or df_day.empty:
        return {"cum_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    returns = []
    balance = initial_balance
    leverage = 3  # 고정 가정(참고용)
    for _ in range(len(df_day)):
        signal = generate_signal(symbol, balance)  # 실제 signal 호출
        if signal.get("action") == "hold":
            continue
        entry = signal["entry"]
        tp = signal["tp"]
        sl = signal["sl"]
        # Mock outcome (win_rate based on prob)
        if np.random.rand() < signal["prob"]:
            ret = (tp - entry) / entry * leverage * RISK_PER_TRADE
        else:
            ret = (sl - entry) / entry * leverage * RISK_PER_TRADE
        returns.append(ret)
        balance *= (1 + ret - 0.0002)  # fee deduct

    if not returns:
        return {"cum_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    df_ret = pd.DataFrame({'returns': returns})
    cum_ret = (1 + df_ret['returns']).cumprod() - 1
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0  # annualized
    drawdown = ((cum_ret.cummax() - cum_ret) / cum_ret.cummax()).max()

    logging.info(f"Backtest for {symbol}: Cum Return {cum_ret.iloc[-1]:.2%}, Sharpe {sharpe:.2f}, Max Drawdown {drawdown:.2%}")
    return {"cum_return": float(cum_ret.iloc[-1]), "sharpe": float(sharpe), "max_drawdown": float(drawdown)}
