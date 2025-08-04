"""Fetch and analyse sentiment for a given query using x.ai Grok 4."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests

from .utils import get_secret

__all__ = ["fetch_and_analyze_x_sentiment"]

XAI_URL = "https://api.x.ai/v1/chat/completions"
XAI_API_KEY: Optional[str] = get_secret("XAI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}" if XAI_API_KEY else "",
    "Content-Type": "application/json",
}
MODEL = "grok-4-0709"
TIMEOUT = 90
RETRIES = 3

# 간단한 메모리 캐시
CACHE: Dict[str, Dict[str, Any]] = {}


def _post(payload: Dict[str, Any]) -> dict:
    for i in range(RETRIES):
        try:
            start = time.time()
            response = requests.post(
                XAI_URL, headers=HEADERS, json=payload, timeout=TIMEOUT
            )
            if response.status_code in (429, 500, 502, 503):
                time.sleep(2**i)
                continue
            response.raise_for_status()
            logging.debug(f"[xAI] latency {time.time() - start:.2f}s")
            return response.json()
        except (requests.Timeout, requests.ConnectionError):
            logging.warning(f"[xAI] timeout {i + 1}/{RETRIES}")
        except Exception as exc:
            logging.error(f"[xAI] error: {exc}")
            break
    raise RuntimeError("xAI retries exhausted")


def fetch_and_analyze_x_sentiment(
    query: str = "BTCUSDT sentiment",
    limit: int = 8,
    cache_minutes: int = 15,
) -> float:
    if not XAI_API_KEY:
        logging.info("XAI_API_KEY not provided; returning neutral sentiment.")
        return 0.0

    cache_key = f"{query}_{limit}"
    cached = CACHE.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0) < cache_minutes * 60):
        return float(cached.get("score", 0.0))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 따옴표/중괄호 충돌 방지를 위해 문자열을 분할하고, JSON 예시는 단일따옴표 리터럴로 명시
    prompt = (
        f"Search X for the most recent posts about '{query}'. "
        f"Use up to {limit} tweets. "
        'Return ONLY JSON like {"score": 0.27}.'
    )

    payload: Dict[str, Any] = {
        "model": MODEL,
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "messages": [{"role": "user", "content": prompt}],
        "search_parameters": {
            "mode": "on",
            "sources": [{"type": "x"}],
            "max_search_results": limit,
            "from_date": today,  # 날짜만 사용
            "return_citations": False,
            "live_search_timeout_ms": 40000,
        },
    }

    try:
        resp = _post(payload)
        raw_txt = resp["choices"][0]["message"]["content"]
        raw = json.loads(raw_txt.strip())
        score = float(raw.get("score", 0.0))
    except Exception as exc:
        logging.error(f"[xSentiment] error→0.0 {exc}")
        score = 0.0

    CACHE[cache_key] = {"ts": time.time(), "score": score}
    logging.info(f"[xSentiment] score={score:+.2f}")
    return score
