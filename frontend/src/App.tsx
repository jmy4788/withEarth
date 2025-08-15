import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import { createChart, CrosshairMode, IChartApi, CandlestickData, ColorType, UTCTimestamp } from "lightweight-charts";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Coins,
  Gauge,
  LineChart as LineChartIcon,
  ShieldCheck,
} from "lucide-react";

/**
 * Liquid Glass Crypto Dashboard (React SPA) — MVP 1~5
 *
 * 구현 범위
 * 1) Main Glass Dashboard 레이아웃/테마/탑스탯
 * 2) 5m 차트 + 체결 마커 + TP/SL 밴드
 * 3) Orderbook Mini-Depth + Spread 라인
 * 4) PNL/Drawdown 차트 + Trades Table Pro(요약)
 * 5) Reasoning Card + What-if 슬라이더(로컬 시뮬)
 *
 * 백엔드 기대 API (Flask)
 * - GET /api/overview → { balances:[{asset,balance,unrealizedPnL}], positions:[...] }
 * - GET /api/trades?limit=200 → [{timestamp,symbol,side,qty,entry,tp,sl,exit,pnl,status,id}, ...]
 * - GET /api/candles?symbol=BTCUSDT&tf=5m&limit=500 → [{t, o, h, l, c, v}, ...]
 * - GET /api/signals/latest?symbol=BTCUSDT → { direction, prob, entry, tp, sl, rr, reasoning, payload_preview }
 * - (선택) GET /api/orderbook?symbol=BTCUSDT&limit=10 → { bids:[[p,q],...], asks:[[p,q],...] }
 *
 * 주의: 해당 엔드포인트가 아직 없으면 UI는 자동으로 skeleton/empty 상태로 폴백.
 */

// ------------------------------
// 공용 타입
// ------------------------------
/** @typedef {{ t:number|string, o:number, h:number, l:number, c:number, v:number }} Candle */
/** @typedef {{ timestamp:string, symbol:string, side:"long"|"short", qty:number, entry:number, tp:number, sl:number, exit:number, pnl:number, status:string, id:string }} TradeRow */

// ------------------------------
// 유틸
// ------------------------------
const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const fmt4 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });
const toPercent = (v:number) => `${(v*100).toFixed(2)}%`;

const glass =
  "backdrop-blur-xl backdrop-saturate-150 bg-white/10 dark:bg-slate-900/40 border border-white/15 shadow-[inset_0_1px_0_rgba(255,255,255,.2)] rounded-2xl";
const neonBTC = "shadow-[0_0_35px_rgba(255,165,0,.35)]";
const neonETH = "shadow-[0_0_35px_rgba(80,140,255,.35)]";

// ------------------------------
// 데이터 훅
// ------------------------------
function useApi<T = any>(url: string | null, deps: any[] = [], { intervalMs = 0 }: { intervalMs?: number } = {}) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(!!url);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!url) return;
    let timer: any;
    let aborted = false;

    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = await res.json();
        if (!aborted) setData(json);
      } catch (e:any) {
        if (!aborted) setError(e?.message || "fetch error");
      } finally {
        if (!aborted) setLoading(false);
      }
    };

    load();
    if (intervalMs > 0) {
      timer = setInterval(load, intervalMs);
    }
    return () => {
      aborted = true;
      if (timer) clearInterval(timer);
    };
  }, [url, intervalMs, ...deps]);

  return { data, loading, error } as const;
}

// ------------------------------
// 상단 통계 (Top Stats Dock)
// ------------------------------
function TopStats({ overview, pnlSummary }: { overview: any, pnlSummary: { pnl:number, win:number, total:number, today:number } }) {
  const balUSDT = useMemo(() => {
    const usdt = (overview?.balances || []).find((b:any) => (b.asset || "").toUpperCase() === "USDT");
    return usdt ? Number(usdt.balance || 0) : 0;
  }, [overview]);

  const unreal = useMemo(() => {
    const usdt = (overview?.balances || []).find((b:any) => (b.asset || "").toUpperCase() === "USDT");
    return usdt ? Number(usdt.unrealizedPnL || 0) : 0;
  }, [overview]);

  const winRate = pnlSummary.total > 0 ? (pnlSummary.win / pnlSummary.total) : 0;

  const statCard = (title:string, value:string, icon:React.ReactNode, accent:boolean=false) => (
    <Card className={`${glass} ${accent ? "ring-1 ring-emerald-400/30" : ""}`}>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs text-slate-300">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex items-center justify-between">
        <div className="text-2xl font-semibold text-white/90">{value}</div>
        <div className="text-slate-300">{icon}</div>
      </CardContent>
    </Card>
  );

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {statCard("USDT Balance", `$${fmt.format(balUSDT)}`, <Coins className="w-5 h-5" />)}
      {statCard("Unrealized PnL", `${fmt.format(unreal)}`, <Activity className="w-5 h-5" />)}
      {statCard("Win Rate", toPercent(winRate), <Gauge className="w-5 h-5" />)}
      {statCard("Today PnL", `$${fmt.format(pnlSummary.today)}`, <LineChartIcon className="w-5 h-5" />, true)}
    </div>
  );
}

// ------------------------------
// 5m 캔들 + 트레이드 마커 + TP/SL 밴드
// ------------------------------
function CandlePanel({ symbol, candles, trades, latestSignal }: { symbol:string, candles:Candle[]|null, trades:TradeRow[]|null, latestSignal:any }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<any>(null);

  // 캔들 데이터 가공
  const cdata: CandlestickData[] = useMemo(() => {
    if (!candles) return [];
    return candles.map((k) => ({
      time: typeof k.t === "string" ? (Date.parse(k.t)/1000 as UTCTimestamp) : ((k.t/1000) as UTCTimestamp),
      open: Number(k.o), high: Number(k.h), low: Number(k.l), close: Number(k.c),
    }));
  }, [candles]);

  useEffect(() => {
    if (!ref.current) return;
    if (chartRef.current) return; // 1회 생성
    const el = ref.current;
    const chart = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#d1d5db" },
      grid: { vertLines: { color: "rgba(255,255,255,.06)" }, horzLines: { color: "rgba(255,255,255,.06)" } },
      rightPriceScale: { borderColor: "rgba(255,255,255,.12)" },
      timeScale: { borderColor: "rgba(255,255,255,.12)", timeVisible: true, secondsVisible: false },
      crosshair: { mode: CrosshairMode.Normal },
      autoSize: true,
    });
    const series = chart.addCandlestickSeries({ upColor: "#22c55e", downColor: "#ef4444", borderVisible: false, wickUpColor: "#22c55e", wickDownColor: "#ef4444" });
    chartRef.current = chart;
    seriesRef.current = series;
    const onResize = () => chart.applyOptions({});
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.remove(); };
  }, []);

  useEffect(() => {
    if (!seriesRef.current) return;
    seriesRef.current.setData(cdata);

    // TP/SL 밴드 (최근 신호 기준)
    if (latestSignal && latestSignal.entry && latestSignal.tp && latestSignal.sl) {
      const { entry, tp, sl } = latestSignal;
      const addLine = (price:number, color:string, title:string) => seriesRef.current.createPriceLine({ price, color, lineWidth: 1, title, lineStyle: 0, axisLabelVisible: true });
      addLine(Number(entry), "#a78bfa", "ENTRY");
      addLine(Number(tp), "#22c55e", "TP");
      addLine(Number(sl), "#ef4444", "SL");
    }

    // 체결 마커 (최근 200개 제한)
    const markers: any[] = [];
    (trades || []).slice(-200).forEach((t) => {
      const ts = Date.parse(t.timestamp)/1000;
      const isLong = t.side === "long";
      markers.push({
        time: ts as UTCTimestamp,
        position: isLong ? "belowBar" : "aboveBar",
        color: isLong ? "#22c55e" : "#ef4444",
        shape: isLong ? "arrowUp" : "arrowDown",
        text: `${t.symbol} ${isLong?"LONG":"SHORT"} ${fmt4.format(t.qty)}`,
      });
    });
    seriesRef.current.setMarkers(markers);
  }, [cdata, trades, latestSignal]);

  return (
    <Card className={`${glass} p-3 min-h-[420px] ${symbol.startsWith("BTC") ? neonBTC : neonETH}`}>
      <div className="flex items-center justify-between px-2 pb-2">
        <div className="flex items-center gap-2">
          <Badge variant={symbol.startsWith("BTC")?"default":"secondary"}>{symbol}</Badge>
          <span className="text-sm text-slate-300">5m Candles · Markers · TP/SL</span>
        </div>
      </div>
      <div ref={ref} className="w-full h-[380px]" />
    </Card>
  );
}

// ------------------------------
// Orderbook Mini-Depth + Spread 라인
// ------------------------------
function OrderbookPanel({ orderbook }: { orderbook: any }) {
  const top = useMemo(() => {
    if (!orderbook || !orderbook.bids || !orderbook.asks) return { bids:[], asks:[], spread: 0 };
    const bids = (orderbook.bids || []).slice(0, 10).map((x:any, i:number) => ({ level: `B${i+1}`, price: Number(x[0]), qty: Number(x[1]) }));
    const asks = (orderbook.asks || []).slice(0, 10).map((x:any, i:number) => ({ level: `A${i+1}`, price: Number(x[0]), qty: Number(x[1]) }));
    const bb = bids[0]?.price || 0; const ba = asks[0]?.price || 0; const mid = bb&&ba? (bb+ba)/2 : 0; const spread = mid? ((ba-bb)/mid*10000):0;
    return { bids, asks, spread };
  }, [orderbook]);

  return (
    <Card className={`${glass} p-3`}> 
      <div className="flex items-center justify-between pb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-300">Orderbook Top10 · Spread</span>
          <Badge variant="outline">{top.spread? `${top.spread.toFixed(2)} bps` : "—"}</Badge>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top.bids} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.08)" />
              <XAxis dataKey="level" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: "rgba(15,23,42,.9)", border: "1px solid rgba(255,255,255,.15)", borderRadius: 12 }} />
              <Bar dataKey="qty" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top.asks} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.08)" />
              <XAxis dataKey="level" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: "rgba(15,23,42,.9)", border: "1px solid rgba(255,255,255,.15)", borderRadius: 12 }} />
              <Bar dataKey="qty" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  );
}

// ------------------------------
// PNL/Drawdown + Trades Table
// ------------------------------
function pnlFromTrades(trades: TradeRow[]|null) {
  const rows = trades || [];
  let cum = 0; const series:any[] = []; let peak = 0; let ddSeries:any[] = [];
  rows.forEach((r) => {
    cum += Number(r.pnl || 0);
    peak = Math.max(peak, cum);
    const dd = peak ? (cum - peak) : 0; // 음수
    series.push({ t: r.timestamp, pnl: cum });
    ddSeries.push({ t: r.timestamp, dd });
  });
  return { series, ddSeries, total: cum };
}

function TradesPanel({ trades }: { trades: TradeRow[]|null }) {
  const rows = trades || [];
  return (
    <Card className={`${glass} p-3`}> 
      <div className="flex items-center justify-between pb-2">
        <span className="text-sm text-slate-300">Recent Trades</span>
        <Badge variant="secondary">{rows.length}</Badge>
      </div>
      <div className="overflow-auto max-h-64">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-300">
              <th className="text-left p-2">Time</th>
              <th className="text-left p-2">Symbol</th>
              <th className="text-left p-2">Side</th>
              <th className="text-right p-2">Qty</th>
              <th className="text-right p-2">Entry</th>
              <th className="text-right p-2">TP</th>
              <th className="text-right p-2">SL</th>
              <th className="text-right p-2">PnL</th>
              <th className="text-left p-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-200).reverse().map((r, idx) => (
              <tr key={idx} className="hover:bg-white/5">
                <td className="p-2 text-slate-300">{new Date(r.timestamp).toLocaleString()}</td>
                <td className="p-2">{r.symbol}</td>
                <td className="p-2">{r.side}</td>
                <td className="p-2 text-right">{fmt4.format(r.qty)}</td>
                <td className="p-2 text-right">{fmt.format(r.entry)}</td>
                <td className="p-2 text-right">{fmt.format(r.tp)}</td>
                <td className="p-2 text-right">{fmt.format(r.sl)}</td>
                <td className={`p-2 text-right ${r.pnl>=0?"text-emerald-400":"text-rose-400"}`}>{fmt.format(r.pnl)}</td>
                <td className="p-2 text-slate-300">{r.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function PnlDrawdownPanel({ trades }: { trades: TradeRow[]|null }) {
  const { series, ddSeries } = useMemo(() => pnlFromTrades(trades), [trades]);
  return (
    <Card className={`${glass} p-3`}>
      <div className="text-sm text-slate-300 pb-2">PNL & Drawdown</div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.08)" />
            <XAxis dataKey="t" stroke="#94a3b8" tickFormatter={(v)=> new Date(v).toLocaleTimeString() }/>
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: "rgba(15,23,42,.9)", border: "1px solid rgba(255,255,255,.15)", borderRadius: 12 }} />
            <Area type="monotone" dataKey="pnl" stroke="#22c55e" fillOpacity={1} fill="url(#pnlGrad)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <Separator className="my-3 bg-white/10"/>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={ddSeries} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.08)" />
            <XAxis dataKey="t" stroke="#94a3b8" tickFormatter={(v)=> new Date(v).toLocaleTimeString() }/>
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: "rgba(15,23,42,.9)", border: "1px solid rgba(255,255,255,.15)", borderRadius: 12 }} />
            <Area type="monotone" dataKey="dd" stroke="#ef4444" fillOpacity={1} fill="url(#ddGrad)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

// ------------------------------
// Reasoning + What-if 슬라이더
// ------------------------------
function ReasoningPanel({ latestSignal }: { latestSignal: any }) {
  const [minProb, setMinProb] = useState<number>(Number(process.env.MIN_PROB || 0.6));
  const [rrMin, setRrMin] = useState<number>(Number(process.env.RR_MIN || 1.2));

  const hypot = useMemo(() => {
    if (!latestSignal) return { decision: "hold", rr: 0 };
    const rr = Number(latestSignal.rr || 0);
    const prob = Number(latestSignal.prob || 0.5);
    const decision = (prob >= minProb && rr >= rrMin && ["long","short"].includes((latestSignal.direction||"").toLowerCase())) ? "enter" : "hold";
    return { decision, rr, prob };
  }, [latestSignal, minProb, rrMin]);

  return (
    <Card className={`${glass} p-3`}>
      <div className="flex items-center justify-between pb-2">
        <span className="text-sm text-slate-300">Reasoning · What-if</span>
        <Badge variant={hypot.decision === "enter" ? "default" : "secondary"}>{hypot.decision.toUpperCase()}</Badge>
      </div>
      <div className="space-y-2 text-slate-200">
        <div className="text-sm whitespace-pre-wrap leading-relaxed">
          {latestSignal?.reasoning || "—"}
        </div>
        <Separator className="my-2 bg-white/10" />
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-slate-300 mb-2">MIN_PROB: {minProb.toFixed(2)}</div>
            <Slider defaultValue={[minProb]} min={0} max={1} step={0.01} onValueChange={(v)=> setMinProb(v[0])} />
          </div>
          <div>
            <div className="text-xs text-slate-300 mb-2">RR_MIN: {rrMin.toFixed(2)}</div>
            <Slider defaultValue={[rrMin]} min={0.5} max={3} step={0.05} onValueChange={(v)=> setRrMin(v[0])} />
          </div>
        </div>
      </div>
    </Card>
  );
}

// ------------------------------
// 메인 앱
// ------------------------------
export default function DashboardApp() {
  const [symbol, setSymbol] = useState<"BTCUSDT"|"ETHUSDT">("BTCUSDT");
  const [tf] = useState<"5m">("5m");

  // 데이터 로딩 (폴링)
  const { data: overview } = useApi<any>(`/api/overview`, [symbol], { intervalMs: 10_000 });
  const { data: trades } = useApi<TradeRow[]>(`/api/trades?limit=200`, [symbol], { intervalMs: 10_000 });
  const { data: candles } = useApi<Candle[]>(`/api/candles?symbol=${symbol}&tf=${tf}&limit=500`, [symbol, tf], { intervalMs: 15_000 });
  const { data: latestSignal } = useApi<any>(`/api/signals/latest?symbol=${symbol}`, [symbol], { intervalMs: 15_000 });
  const { data: orderbook } = useApi<any>(`/api/orderbook?symbol=${symbol}&limit=10`, [symbol], { intervalMs: 8_000 });

  // PnL 요약 계산
  const pnlSummary = useMemo(() => {
    const rows = (trades || []).filter((r) => r.symbol === symbol);
    const total = rows.length;
    const win = rows.filter((r) => Number(r.pnl || 0) > 0).length;
    const today = rows.filter((r)=> new Date(r.timestamp).toDateString() === new Date().toDateString()).reduce((s,r)=> s+Number(r.pnl||0), 0);
    const pnl = rows.reduce((s,r)=> s+Number(r.pnl||0), 0);
    return { total, win, today, pnl };
  }, [trades, symbol]);

  // 레이아웃
  return (
    <div className="min-h-screen w-full bg-[radial-gradient(1200px_600px_at_10%_-10%,rgba(255,165,0,.06),transparent),radial-gradient(1000px_500px_at_90%_10%,rgba(80,140,255,.06),transparent)] text-white">
      <div className="max-w-7xl mx-auto p-4 md:p-6 space-y-6">
        {/* 헤더 */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ type: "spring", stiffness: 120 }} className={`p-2 px-3 ${glass} ${symbol.startsWith("BTC")?neonBTC:neonETH}`}>
              <div className="flex items-center gap-2">
                {symbol.startsWith("BTC") ? <ArrowUpRight className="w-4 h-4 text-amber-400"/> : <ArrowDownRight className="w-4 h-4 text-blue-400"/>}
                <span className="font-semibold tracking-wide">withEarth — Auto Trader</span>
              </div>
            </motion.div>
            <Badge variant="outline" className="bg-white/10">Liquid Glass</Badge>
          </div>
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-emerald-400"/>
            <span className="text-slate-300 text-sm">Paper/Testnet/Real 토글은 추후(아이템 6)</span>
          </div>
        </div>

        {/* 심볼 탭 + Top Stats */}
        <Tabs value={symbol} onValueChange={(v)=> setSymbol(v as any)}>
          <TabsList className={`${glass} p-1`}> 
            <TabsTrigger value="BTCUSDT">BTCUSDT</TabsTrigger>
            <TabsTrigger value="ETHUSDT">ETHUSDT</TabsTrigger>
          </TabsList>
          <TabsContent value={symbol} className="mt-4">
            <TopStats overview={overview} pnlSummary={pnlSummary} />
          </TabsContent>
        </Tabs>

        {/* 메인 그리드: 좌 차트 / 우 사이드패널 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            <CandlePanel symbol={symbol} candles={candles||null} trades={trades||null} latestSignal={latestSignal||null} />
            <PnlDrawdownPanel trades={trades||null} />
          </div>
          <div className="space-y-4">
            <OrderbookPanel orderbook={orderbook||null} />
            <ReasoningPanel latestSignal={latestSignal||null} />
            <TradesPanel trades={trades||null} />
          </div>
        </div>

        {/* 푸터 노트 */}
        <div className="text-center text-xs text-slate-400 pt-2">UI v0 · Items 1~5 · 다음 단계: 6) Execution Toggle, 7) Toast/Error, 8) SSE Stream</div>
      </div>
    </div>
  );
}
