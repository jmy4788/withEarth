# analyze_run.py
# 사용법: python analyze_run.py --base https://<PROJECT_ID>.appspot.com --limit 5000
import argparse, json, sys, time
from collections import Counter, defaultdict
import requests
import pandas as pd

def get_json(base, path, params=None):
    r = requests.get(base + path, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_all(base, limit):
    trades = get_json(base, "/api/trades", {"limit": limit})
    logs = get_json(base, "/api/logs", {"lines": 10000})
    signals = get_json(base, "/api/signals")
    return trades, logs, signals

def parse_event_lines(lines):
    events = []
    for ln in lines.get("lines", []):
        # ln 예: '2025-08-20 23:49:52,058 INFO [event] {"event": "signal.gate", ...}'
        try:
            p = ln.split("] ", 1)[1]
            obj = json.loads(p)
            if isinstance(obj, dict) and obj.get("event"):
                events.append(obj)
        except Exception:
            continue
    return events

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    trades, logs, signals = fetch_all(args.base, args.limit)
    events = parse_event_lines(logs)
    ev_by = defaultdict(list)
    for e in events:
        ev_by[e.get("event")].append(e)

    # 1) 게이팅 사유 분해
    gate = ev_by.get("signal.gate", [])
    reasons = Counter()
    rr_vals, probs, spreads = [], [], []
    for g in gate:
        rs = str(g.get("reasons","")).split(";") if g.get("reasons") else []
        for r in rs:
            if not r or r=="ok": continue
            # rr_net_below_min(x<y) → 키만 추출
            reasons[r.split("(")[0]] += 1
        if "rr" in g: rr_vals.append(float(g["rr"]))
        if "prob" in g: probs.append(float(g["prob"]))
        if "spread_bps" in g: spreads.append(float(g["spread_bps"]))

    # 2) 의사결정/예측 분포
    dec = ev_by.get("signal.decision", [])
    dec_ok = sum(1 for d in dec if str(d.get("risk_ok","")).lower()=="true")
    dir_cnt = Counter(str(d.get("direction","")) for d in dec)

    # 3) 트레이드 레코드 요약
    tj = trades.get("summary", {})
    rows = trades.get("rows", [])
    # 엔트리-슬 거리/엔트리 비율, 엔트리-TP 거리/엔트리 비율 분포
    import math
    k_sl, k_tp = [], []
    for r in rows:
        e, sl, tp = float(r.get("entry",0)), float(r.get("sl",0)), float(r.get("tp",0))
        if e>0 and sl>0 and tp>0:
            if e>sl: k_sl.append((e-sl)/e)  # long 기준 비율
            else:    k_sl.append((sl-e)/e)  # short
            if tp>e: k_tp.append((tp-e)/e)
            else:    k_tp.append((e-tp)/e)

    # 출력
    print("=== Gate Reasons (top 10) ===")
    for k,v in reasons.most_common(10):
        print(f"{k:24s} : {v}")
    print("\nCounts: gate_events={}, decisions={}, decisions(risk_ok)={}".format(len(gate), len(dec), dec_ok))
    print("Direction mix:", dict(dir_cnt))
    print("RR (median, p10~p90):", (pd.Series(rr_vals).median() if rr_vals else None,
                                     pd.Series(rr_vals).quantile(0.1) if rr_vals else None,
                                     pd.Series(rr_vals).quantile(0.9) if rr_vals else None))
    print("Prob (median):", (pd.Series(probs).median() if probs else None))
    print("Spread bps (median):", (pd.Series(spreads).median() if spreads else None))
    print("Trade rows:", len(rows), "Summary:", tj)

    # CSV로 저장
    pd.DataFrame(gate).to_csv("gate_events.csv", index=False)
    pd.DataFrame(dec).to_csv("decisions.csv", index=False)
    pd.DataFrame(rows).to_csv("trades_rows.csv", index=False)

    # 참고: payload/decision 파일은 /api/signals로 간접 확인
    items = signals.get("items", [])
    print("\nSignals items:", len(items))
    if items[:3]:
        print("Sample decisions:", json.dumps(items[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
