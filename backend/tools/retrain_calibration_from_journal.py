import os
import csv
from helpers.calibration import ProbCalibrator
from helpers.utils import LOG_DIR

TRADES = os.path.join(str(LOG_DIR), "trades.csv")
OUT_PATH = os.environ.get(
    "PROB_CALIBRATION_PATH",
    os.path.join(str(LOG_DIR), "calibration_tsfm2.json"),
)


def load_labels_probs(path: str):
    probs, labels = [], []
    if not os.path.exists(path):
        print("no trades.csv:", path)
        return probs, labels
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status", "") or row.get("close_reason", "")).lower()
            label = 1 if "closed_tp" in status else 0
            try:
                prob = float(row.get("prob_raw") or row.get("prob") or 0.5)
            except Exception:
                prob = 0.5
            probs.append(prob)
            labels.append(label)
    return probs, labels


def main():
    probs, labels = load_labels_probs(TRADES)
    calibrator = ProbCalibrator(
        path=OUT_PATH,
        bins=int(os.environ.get("CALIB_BINS", "10")),
        min_samples=int(os.environ.get("CALIB_MIN_SAMPLES", "150")),
    )
    if calibrator.fit_from_arrays(probs, labels):
        calibrator.save()
        print("saved:", OUT_PATH, "n=", len(probs))
    else:
        print("not enough samples:", len(probs))


if __name__ == "__main__":
    main()
