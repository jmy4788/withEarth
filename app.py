from __future__ import annotations
"""
app.py — Flask server (GAE/로컬 공용)
- /health: 헬스체크
- /tasks/trader: 심볼별 시그널 생성/주문. debug=1이면 에러 스택을 JSON에 포함 (임시 계측)

주의: DataFrame을 불리언으로 평가하는 코드가 어딘가 남아있을 가능성이 큼.
원인 라인을 정확히 찾기 위해 본 파일은 스택트레이스를 파일 및 응답(JSON, debug=1)으로 노출.
문제 해결 후 debug 출력은 끄세요.

Env:
- SYMBOLS (기본: BTCUSDT,ETHUSDT)
- EXECUTE_TRADES ("true"면 manage_trade 실행)
- PORT (기본: 8080)
- LOG_DIR (기본: ./logs)
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request

# -----------------------------
# 로깅 설정 (파일 + 콘솔)
# -----------------------------
def setup_logging() -> logging.Logger:
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "bot.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 중복 핸들러 방지
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

    if not any(h.__class__.__name__ == "StreamHandler" for h in logger.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(console)

    logging.info("Logging initialized. log_path=%s", log_path)
    return logger

logger = setup_logging()

# -----------------------------
# Flask
# -----------------------------
server = Flask(__name__)
server.logger.handlers = []  # Flask 기본 핸들러 제거
server.logger.propagate = True  # 루트 로거로 전달

# -----------------------------
# 안전 헬퍼
# -----------------------------
def _parse_symbols(param: Optional[str]) -> List[str]:
    if not param:
        param = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    syms = [s.strip().upper() for s in str(param).split(",") if s.strip()]
    return syms or ["BTCUSDT", "ETHUSDT"]

def _bool_env(name: str, default: bool=False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","y","on")

# -----------------------------
# 프로젝트 모듈 임포트
# -----------------------------
try:
    from helpers.signals import generate_signal, manage_trade  # type: ignore
except Exception as e:
    logger.exception("Failed to import project modules")
    raise

# -----------------------------
# Routes
# -----------------------------
@server.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})

@server.route("/tasks/trader", methods=["GET", "POST"])
def tasks_trader():
    """
    debug=1 쿼리(또는 본문 {"debug":1})면, 에러 시 traceback 문자열을 results[*].trace에 포함.
    """
    try:
        payload_json: Dict[str, Any] = {}
        if request.method == "POST":
            payload_json = request.get_json(silent=True) or {}

        query_debug = request.args.get("debug")
        dbg = payload_json.get("debug", None)
        want_debug = (str(query_debug).strip() == "1") or (str(dbg).strip() == "1")

        symbols = _parse_symbols(payload_json.get("symbols") or request.args.get("symbols"))
        do_trade = _bool_env("EXECUTE_TRADES", False)

        results: List[Dict[str, Any]] = []
        for sym in symbols:
            try:
                sig = generate_signal(sym)  # dict-like

                exec_info = None
                if do_trade:
                    try:
                        exec_info = manage_trade(sym)
                    except Exception:
                        logger.exception("manage_trade failed for %s", sym)
                        if want_debug:
                            exec_info = {"error": "manage_trade failed", "trace": traceback.format_exc()}
                        else:
                            exec_info = {"error": "manage_trade failed"}

                results.append({
                    "symbol": sym,
                    "result": {
                        "action": sig.get("action", sig.get("direction","hold")),
                        "direction": sig.get("direction", "hold"),
                        "entry": sig.get("entry", 0.0),
                        "tp": sig.get("tp", 0.0),
                        "sl": sig.get("sl", 0.0),
                        "prob": sig.get("prob", 0.5),
                        "risk_ok": sig.get("risk_ok", False),
                        "rr": sig.get("rr", 0.0),
                        "reason": sig.get("reasoning", sig.get("reason","")),
                        "payload_preview": sig.get("payload_preview", {}),
                    },
                    **({"exec": exec_info} if exec_info else {}),
                })
            except Exception as e:
                # 핵심: 파일 로그 + 선택적 JSON trace
                logger.exception("tasks_trader symbol=%s failed", sym)
                item = {"symbol": sym, "error": str(e)}
                if want_debug:
                    item["trace"] = traceback.format_exc()
                results.append(item)

        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        logger.exception("/tasks/trader failed")
        out = {"status": "error", "error": str(e)}
        if request.args.get("debug") == "1":
            out["trace"] = traceback.format_exc()
        return jsonify(out), 500

# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    server.run(host="0.0.0.0", port=port, debug=True)
