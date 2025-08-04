# helpers/utils.py
from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

# google.cloud.*는 배포 환경에서만 필수
try:
    from google.cloud import secretmanager, storage  # type: ignore
except Exception:  # 로컬 개발에서 미설치 가능
    secretmanager = None
    storage = None

# --- Initial Setup ---
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
IS_SERVERLESS = bool(os.getenv("GAE_ENV", "").startswith("standard") or os.getenv("K_SERVICE"))

# --- Logging Configuration ---
if IS_SERVERLESS:
    # Use /tmp for logs in serverless environments
    LOG_DIR = Path("/tmp/trading_bot")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]  # stdout
    )
    # File logging (best-effort)
    try:
        fh = logging.FileHandler(str(LOG_DIR / "bot.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception as e:
        logging.warning(f"File logging setup failed: {e}")
else:
    LOG_DIR = ROOT_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename=str(LOG_DIR / "bot.log"),
    )

# --- Global Utilities ---
TZ = os.getenv("TZ", "UTC")
GCS_BUCKET = os.getenv("GCS_BUCKET")  # ex) my-bucket
GCS_PREFIX = os.getenv("GCS_PREFIX", "trading_bot")  # ex) trading_bot

def _now_tz() -> datetime:
    """Return current time in configured timezone."""
    if TZ == "Asia/Seoul":
        return datetime.now(timezone.utc) + timedelta(hours=9)
    return datetime.now(timezone.utc)

def _detect_project_id() -> Optional[str]:
    """Detect GCP project id from env."""
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GOOGLE_PROJECT_ID")
    )

def get_secret(secret_name: str) -> Optional[str]:
    """
    1) env 우선
    2) 서버리스 환경이면 Secret Manager 사용
    """
    env_val = os.getenv(secret_name)
    if env_val:
        return env_val

    if not IS_SERVERLESS or secretmanager is None:
        logging.info(f"'{secret_name}' not in env; skipping Secret Manager in local.")
        return None

    try:
        project_id = _detect_project_id()
        if not project_id:
            raise RuntimeError("GCP Project ID not found in environment variables.")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.warning(f"Failed to retrieve secret '{secret_name}' from GCP: {e}")
        return None

# ----------------------------
#        GCS Utilities
# ----------------------------
def gcs_enabled() -> bool:
    """
    GCS 로깅 활성화 여부.
    - 필수: GCS_BUCKET 설정
    - 배포환경: 기본 True (권한 필요)
    - 로컬: GOOGLE_APPLICATION_CREDENTIALS 등이 설정되어 있으면 True
    """
    if not GCS_BUCKET:
        return False
    if IS_SERVERLESS:
        return storage is not None
    # local
    return storage is not None and bool(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT")
    )

def _gcs_client() -> Optional["storage.Client"]:
    if not gcs_enabled():
        return None
    try:
        return storage.Client()  # type: ignore
    except Exception as e:
        logging.warning(f"GCS client init failed: {e}")
        return None

def _gcs_blob_path(kind: str, ts: Optional[datetime] = None) -> str:
    """
    kind: 'trades' | 'balance'
    파일을 'prefix/kind/YYYYMM/DD/epoch.csv' 로 저장 (append 없이 신규 1행 파일 생성)
    """
    ts = ts or _now_tz()
    yyyymm = ts.strftime("%Y%m")
    dd = ts.strftime("%d")
    epoch = int(ts.timestamp() * 1_000_000)  # microsecond-level
    return f"{GCS_PREFIX}/{kind}/{yyyymm}/{dd}/{kind}-{epoch}.csv"

def gcs_append_csv_row(kind: str, headers: List[str], row: dict) -> None:
    """
    GCS에 '1행짜리 CSV'를 개별 객체로 업로드.
    - GCS는 append가 불가하므로, 일 단위/시간 단위로 작은 파일을 쌓고
      조회 시 병합해서 읽는 방식으로 처리.
    """
    client = _gcs_client()
    if not client:
        return
    try:
        bucket = client.bucket(GCS_BUCKET)  # type: ignore
        blob = bucket.blob(_gcs_blob_path(kind))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        # 필드 누락 대비: 순서/헤더 정렬
        normalized = {h: row.get(h) for h in headers}
        writer.writerow(normalized)
        blob.upload_from_string(buf.getvalue(), content_type="text/csv")
    except Exception as e:
        logging.warning(f"GCS upload failed: {e}")

def gcs_read_recent_csvs(kind: str, max_files: int = 30) -> Optional[str]:
    """
    GCS에 쌓인 '작은 CSV 파일들'을 최신순 최대 max_files 만큼 내려받아
    로컬 임시 파일 하나로 병합 후 그 경로를 반환.
    pandas가 읽을 수 있는 단일 CSV 경로를 돌려주는 것이 목표.
    """
    client = _gcs_client()
    if not client:
        return None
    try:
        bucket = client.bucket(GCS_BUCKET)  # type: ignore
        prefix = f"{GCS_PREFIX}/{kind}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            return None
        # 최신 객체 우선
        blobs.sort(key=lambda b: b.updated or _now_tz(), reverse=True)
        blobs = blobs[:max_files]

        # 병합
        tmp_path = Path(tempfile.gettempdir()) / f"{kind}_recent.csv"
        rows = []
        header = None
        for b in blobs:
            data = b.download_as_text()
            s = io.StringIO(data)
            reader = csv.DictReader(s)
            if header is None:
                header = reader.fieldnames
            for r in reader:
                rows.append(r)
        if not rows or not header:
            return None
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        return str(tmp_path)
    except Exception as e:
        logging.warning(f"GCS read/merge failed: {e}")
        return None
