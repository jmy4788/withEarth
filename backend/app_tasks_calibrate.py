# app_tasks_calibrate.py
from __future__ import annotations
from flask import Blueprint, request, jsonify
import os
from pathlib import Path

from helpers.utils import log_event, gcs_enabled  # Cloud Logging JSON 로그, GCS 여부 확인  :contentReference[oaicite:1]{index=1}
from helpers.calibration import ProbCalibrator     # 보정 곡선 저장/로드/적용                     :contentReference[oaicite:2]{index=2}
from tools.calibrate_from_trades import load_samples  # (prob,label) 샘플 생성 로직               :contentReference[oaicite:3]{index=3}

calib_bp = Blueprint("tasks_calibrate", __name__)

@calib_bp.route("/tasks/calibrate", methods=["GET", "POST"])
def tasks_calibrate():
    # App Engine Cron 보호: 내부 크론 호출만 허용
    if request.headers.get("X-Appengine-Cron", "").lower() != "true":
        return jsonify({"error": "forbidden"}), 403

    symbols_env = os.getenv("SYMBOLS", "")  # 예: "BTCUSDT,ETHUSDT"
    only_symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    horizon_min = int(os.getenv("HORIZON_MIN", "30"))        # 신호의 기본 호라이즌과 동일      :contentReference[oaicite:4]{index=4}
    min_samples = int(os.getenv("CALIB_MIN_SAMPLES", "150")) # 최소 샘플수                      :contentReference[oaicite:5]{index=5}
    bins = int(os.getenv("CALIB_BINS", "10"))                # 비닝 수                           :contentReference[oaicite:6]{index=6}

    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    trades_csv = log_dir / "trades.csv"

    # 1) trades.csv에서 (prob,label) 수집
    samples = load_samples(trades_csv, horizon_min, only_symbols=only_symbols)
    n = len(samples)
    log_event("calibrate.collect", symbols=only_symbols or "ALL", horizon_min=horizon_min, samples=n)

    if n < min_samples:
        return jsonify({"status": "skipped", "reason": "insufficient_samples", "n": n, "min_samples": min_samples}), 200

    probs = [s.prob for s in samples]
    labels = [s.label for s in samples]

    # 2) 보정 곡선 적합 및 저장
    calib = ProbCalibrator(bins=bins, min_samples=min_samples)
    ok = calib.fit_from_arrays(probs, labels)  # 내부에서 단조 보정 + save() 수행                 :contentReference[oaicite:7]{index=7}
    if not ok:
        return jsonify({"status": "failed", "reason": "fit_failed"}), 200

    calib.save()
    log_event("calibrate.saved", path=calib.path, bins=bins, n=n)

    # 3) (선택) 최신 보정 파일을 GCS에도 업로드 — 다중 인스턴스/롤링 재시작 대비
    if gcs_enabled():
        try:
            from google.cloud import storage  # 배포 이미지에 포함되어 있어야 함
            client = storage.Client()
            bucket = client.bucket(os.getenv("GCS_BUCKET"))
            prefix = os.getenv("GCS_PREFIX", "trading_bot")
            dst = f"{prefix}/calibration/latest/calibration.json"
            blob = bucket.blob(dst)
            blob.cache_control = "no-cache"
            blob.upload_from_filename(calib.path, content_type="application/json")
            log_event("calibrate.gcs_uploaded", gcs_path=dst)
        except Exception as e:
            log_event("calibrate.gcs_upload_failed", error=str(e))

    return jsonify({"status": "ok", "saved_to": calib.path, "samples_used": n, "bins": bins}), 200
