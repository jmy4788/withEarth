from __future__ import annotations
"""
helpers/utils.py — refactor (2025-08-12, KST)

역할
- 환경변수/비밀 관리(get_secret) — Google Secret Manager가 있으면 우선 사용
- 로그/디렉토리 준비(LOG_DIR)
- GCS 보조 유틸: CSV 1행 스냅샷 업로드(gcs_append_csv_row)
- (선택) 파일 로거 초기화

호환성
- signals.py, predictor.py, binance_client.py가 기대하는 심볼을 그대로 제공합니다:
  - LOG_DIR: Path 또는 str
  - get_secret(name) -> Optional[str]
  - gcs_enabled() -> bool
  - gcs_append_csv_row(dataset: str, headers: list[str], row: dict) -> None
"""
import json
import csv
import logging
import os
import random
import string
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

# .env 로딩(있으면)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------------------------
# 경로/로그 디렉토리
# ---------------------------
_DEFAULT_TMP = "/tmp/trading_bot"  # GAE 등에서 쓰기 가능
_LOG_DIR_ENV = os.getenv("LOG_DIR", "./logs")


def _ensure_dir_writable(pref: str) -> Path:
    """pref 경로가 쓰기 가능하면 그대로, 아니면 /tmp로 폴백."""
    try:
        p = Path(pref)
        p.mkdir(parents=True, exist_ok=True)
        t = p / ".wtest"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)  # type: ignore
        return p
    except Exception:
        p = Path(_DEFAULT_TMP)
        p.mkdir(parents=True, exist_ok=True)
        return p


LOG_DIR: Path = _ensure_dir_writable(_LOG_DIR_ENV)


# ---------------------------
# 시간/유틸
# ---------------------------
def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _rand_token(n: int = 6) -> str:
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


# ---------------------------
# Secret 관리
# ---------------------------
_SM_READY = False
try:
    from google.cloud import secretmanager  # type: ignore
    _SM_READY = True
except Exception:
    _SM_READY = False

_SECRET_CACHE: dict[str, Optional[str]] = {}


def get_secret(name: str) -> Optional[str]:
    """
    비밀 조회(환경변수 우선 → Secret Manager).
    - 환경변수에 있으면 그대로 반환.
    - Secret Manager 사용 시:
        * 프로젝트는 ENV 'GOOGLE_CLOUD_PROJECT' 또는 'GCP_PROJECT'로 추정
        * 비밀 버전은 'latest'
    """
    if not name:
        return None
    if name in _SECRET_CACHE:
        return _SECRET_CACHE[name]

    # 1) 환경변수 우선
    val = os.getenv(name)
    if val:
        _SECRET_CACHE[name] = val
        return val

    # 2) Google Secret Manager (선택)
    if _SM_READY:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
        if project_id:
            try:
                client = secretmanager.SecretManagerServiceClient()
                secret_path = f"projects/{project_id}/secrets/{name}/versions/latest"
                resp = client.access_secret_version(name=secret_path)
                val = resp.payload.data.decode("utf-8")
                _SECRET_CACHE[name] = val
                return val
            except Exception as e:
                logging.info(f"SecretManager miss for {name}: {e}")

    _SECRET_CACHE[name] = None
    return None


# ---------------------------
# 파일 로깅(선택)
# ---------------------------
def setup_file_logger(filename: str = "bot.log", level: int = logging.INFO) -> None:
    """
    간단 파일 로거. app.py가 자체 로깅을 세팅했다면 호출할 필요 없음.
    """
    try:
        path = LOG_DIR / filename
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        fh.setFormatter(fmt)
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(fh)
    except Exception as e:
        logging.warning(f"setup_file_logger failed: {e}")


# ---------------------------
# GCS 업로드 유틸(스냅샷 방식)
# ---------------------------
_GCS_OK = False
try:
    from google.cloud import storage  # type: ignore
    _GCS_OK = True
except Exception:
    _GCS_OK = False

GCS_BUCKET = os.getenv("GCS_BUCKET")  # ex) my-bucket
GCS_PREFIX = os.getenv("GCS_PREFIX", "trading_bot")  # ex) trading_bot


def gcs_enabled() -> bool:
    """
    GCS 업로드 가능 여부.
    - 환경변수 GCS_BUCKET이 있고
    - google-cloud-storage가 임포트 가능하면 True
    (자격증명은 런타임 기본 ADC를 사용)
    """
    return bool(GCS_BUCKET) and _GCS_OK


def _gcs_upload_bytes(dst_path: str, data: bytes, content_type: str = "text/plain") -> None:
    """dst_path = 'prefix/dir/file.csv'"""
    if not gcs_enabled():
        return
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)  # type: ignore[arg-type]
        blob = bucket.blob(dst_path)
        blob.cache_control = "no-cache"
        blob.upload_from_string(data, content_type=content_type)
    except Exception as e:
        logging.info(f"GCS upload failed: {e}")


def gcs_append_csv_row(dataset: str, headers: List[str], row: dict) -> None:
    """
    CSV 1행을 '스냅샷 파일'로 업로드.
    - 'append'를 GCS에서 원자적으로 하기 어렵기 때문에,
      작은 CSV 파일을 꾸준히 쌓는 방식(리플레이/수집 파이프라인에 유리).
    - 업로드 경로: {GCS_PREFIX}/{dataset}/YYYYMMDD/{HHMMSS}_{rand}.csv
    """
    if not gcs_enabled():
        return
    try:
        date_dir = _now_utc().strftime("%Y%m%d")
        time_tag = _now_utc().strftime("%H%M%S")
        rand = _rand_token(5)
        dst = f"{GCS_PREFIX}/{dataset}/{date_dir}/{time_tag}_{rand}.csv"

        # 임시 CSV 생성
        with tempfile.NamedTemporaryFile("w", newline="", delete=False, encoding="utf-8") as tf:
            w = csv.DictWriter(tf, fieldnames=headers)
            w.writeheader()
            w.writerow(row)
            tmp_path = tf.name

        # 업로드
        data = Path(tmp_path).read_bytes()
        _gcs_upload_bytes(dst, data, content_type="text/csv")

        # 청소
        try:
            Path(tmp_path).unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass
    except Exception as e:
        logging.info(f"gcs_append_csv_row failed: {e}")

def log_event(event: str, **fields) -> None:
    """
    Cloud Logging에서 보기 편하도록 JSON 문자열로 INFO 레벨 로그를 남깁니다.
    - 표준 출력으로 흘러가므로 GAE에서 자동 수집됩니다.
    - 민감정보(API 키 등)는 포함하지 않습니다.
    """
    try:
        logging.getLogger("event").info(
            json.dumps({"event": event, **fields}, ensure_ascii=False, default=str)
        )
    except Exception as e:
        logging.getLogger(__name__).info("log_event failed: %s", e)

__all__ = [
    "LOG_DIR",
    "get_secret",
    "setup_file_logger",
    "gcs_enabled",
    "gcs_append_csv_row",
    "log_event",
]
