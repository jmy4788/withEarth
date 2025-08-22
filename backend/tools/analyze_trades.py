# tools/analyze_trades.py
from __future__ import annotations
import argparse, csv, json, math, os
from datetime import datetime, timezone, timedelta
from pathlib import Path

def _parse_bool(s: str, default=False)->bool:
    if s is None: return default
    return str(s).lower() in ("1","true","yes","y","on")

def _safe_float(x, d=0.0):
    try: return float(x)
    except: return d

def _coerce_label(status: str, pnl: float)->int:
    st = str(status or "").strip().lower()
    if "closed_tp" in st or ("tp" in st and "stop" not in st): return 1
    if "closed_sl" in st or "stop" in st or "time_exit" in st: return 0
    return 1 if float(pnl or 0.0) > 0 else 0

def _read(path: Path, limit:int)->list[dict]:
    if not path.exists(): return []
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows[-limit:]

def _brier(arr): 
    return sum((p-y)**2 for p,y in arr)/len(arr) if arr else None

def _logloss(arr):
    if not arr: return None
    eps=1e-12
    return -sum(y*math.log(max(eps,min(1-eps,p))) + (1-y)*math.log(max(eps,min(1-eps,1-p))) for p,y in arr)/len(arr)

def _auc(arr):
    if not arr: return None
    n1=sum(1 for _,y in arr if y==1); n0=sum(1 for _,y in arr if y==0)
    if n1==0 or n0==0: return None
    arr=sorted(arr,key=lambda x:x[0])
    i=0; n=len(arr); rank_sum_pos=0.0
    while i<n:
        j=i; v=arr[i][0]
        while j<n and arr[j][0]==v: j+=1
        avg=(i+1 + j)/2.0
        rank_sum_pos += avg * sum(1 for k in range(i,j) if arr[k][1]==1)
        i=j
    U = rank_sum_pos - n1*(n1+1)/2.0
    return U/(n1*n0)

def _ece(arr, bins:int=10):
    if not arr: return None, []
    edges=[i/bins for i in range(bins+1)]
    pts=[]; total=len(arr); ece=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        seg=[(p,y) for (p,y) in arr if (p>=lo and (p<hi if i<bins-1 else p<=hi))]
        if len(seg)<3: continue
        mp=sum(p for p,_ in seg)/len(seg)
        my=sum(y for _,y in seg)/len(seg)
        ece+= (len(seg)/total)*abs(mp-my)
        pts.append({"x":mp,"y":my,"n":len(seg)})
    return ece, pts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.getenv("LOG_DIR","./logs"))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--symbol", default="")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args=ap.parse_args()

    path=Path(args.log_dir)/"trades.csv"
    rows=_read(path,args.limit)
    if args.symbol: rows=[r for r in rows if str(r.get("symbol","")).upper()==args.symbol.upper()]
    if args.days>0:
        cutoff=datetime.now(tz=timezone.utc)-timedelta(days=args.days)
        filt=[]
        for r in rows:
            ts=str(r.get("timestamp",""))
            try:
                dt=datetime.fromisoformat(ts.replace("Z","+00:00"))
                if dt>=cutoff: filt.append(r)
            except: filt.append(r)
        rows=filt
    rows=[r for r in rows if str(r.get("status","")).lower()!="open"]

    if not rows:
        print("No trades"); return

    pnl=[_safe_float(r.get("pnl",0.0)) for r in rows]
    total=sum(pnl); wins=len([x for x in pnl if x>0]); n=len(rows)
    pf=(sum([x for x in pnl if x>0])/max(1e-12,-sum([x for x in pnl if x<=0]))) if wins<n else None
    labels=[_coerce_label(r.get("status",""), _safe_float(r.get("pnl",0.0))) for r in rows]
    probs=[_safe_float(r.get("prob",0.0)) for r in rows]
    arr_cal=list(zip(probs,labels))
    ece_cal, rel_cal = _ece(arr_cal, bins=args.bins)
    # raw if present
    pr=[]
    lab_raw=[]
    for i,r in enumerate(rows):
        try:
            prx=float(r.get("prob_raw"))
            if prx==prx:  # not NaN
                pr.append(prx); lab_raw.append(labels[i])
        except: pass
    arr_raw=list(zip(pr,lab_raw))

    out={
        "summary":{"n":n,"wins":wins,"win_rate":wins/max(1,n),"total_pnl":total,"profit_factor":pf},
        "prediction":{
            "brier_cal":_brier(arr_cal),"auc_cal":_auc(arr_cal),"logloss_cal":_logloss(arr_cal),
            **({"brier_raw":_brier(arr_raw),"auc_raw":_auc(arr_raw),"logloss_raw":_logloss(arr_raw)} if arr_raw else {})
        },
        "reliability":{"calibrated":{"ece":ece_cal,"points":rel_cal}},
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"N={n}, WinRate={out['summary']['win_rate']:.2%}, PnL={total:.4f}, PF={pf if pf is not None else 'NA'}")
        print(f"Brier(cal)={out['prediction']['brier_cal']:.6f}, AUC(cal)={out['prediction']['auc_cal']:.4f}, LogLoss(cal)={out['prediction']['logloss_cal']:.6f}")
        if arr_raw:
            print(f"Brier(raw)={out['prediction']['brier_raw']:.6f}, AUC(raw)={out['prediction']['auc_raw']:.4f}, LogLoss(raw)={out['prediction']['logloss_raw']:.6f}")
        print("Reliability (x=mean p, y=empirical):")
        for p in rel_cal: print(f"  x={p['x']:.3f} -> y={p['y']:.3f} (n={p['n']})")

if __name__=="__main__":
    main()
