param(
  [string]$Project = "vaulted-scholar-466013-r5",
  [string]$Target  = "prod",    # prod | local
  [string]$Sym1    = "BTCUSDT",
  [string]$Sym2    = "ETHUSDT",
  [int]$Limit      = 100,      # metrics/trades/diagnostics rows
  [int]$LogLines   = 100,      # /api/logs lines
  [string]$ReadApiKey = "",     # for /api/open_orders (if protected)
  [string]$Since   = "",        # ISO8601 override for orders_history window
  [string]$GcsDate = "",        # YYYYMMDD for /api/gcs/snapshots (optional)
  [string]$OutDir  = "",        # default: script directory
  [switch]$Cron                 # send X-Appengine-Cron:true to /tasks/*
)

# ---- Base URL ----
$BaseProd  = "https://$Project.appspot.com"
$BaseLocal = "http://localhost:8080"
$Base = if ($Target -eq "local") { $BaseLocal } else { $BaseProd }
Write-Host "Base = $Base" -ForegroundColor Cyan

# ---- Output dirs & files ----
$ScriptRoot = Split-Path -Parent $PSCommandPath
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = $ScriptRoot }
$OutJsonDir = Join-Path $OutDir 'json'
$OutLogDir  = Join-Path $OutDir 'log'
New-Item -ItemType Directory -Force -ErrorAction SilentlyContinue -Path $OutJsonDir | Out-Null
New-Item -ItemType Directory -Force -ErrorAction SilentlyContinue -Path $OutLogDir  | Out-Null
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$JsonFile = Join-Path $OutJsonDir ("analysis_{0}_{1}_{2}.json" -f $Project, $Target, $ts)
$LogFile  = Join-Path $OutLogDir  ("analysis_{0}_{1}_{2}.log"  -f $Project, $Target, $ts)

# ---- Utils ----
function Write-Log($text) { $text | Out-File -FilePath $LogFile -Append -Encoding utf8 }
function Write-Section($title) { $line = "=== $title ==="; Write-Host $line -ForegroundColor Yellow; Write-Log $line }

$DefaultHeaders = @{}
if ($Cron) { $DefaultHeaders["X-Appengine-Cron"] = "true" }  # for /tasks/* only when testing

function Get-JsonObj([string]$url, [hashtable]$headers=$null) {
  try {
    if ($null -eq $headers) { $headers = $DefaultHeaders }
    return Invoke-RestMethod -Method GET -Uri $url -Headers $headers -TimeoutSec 30
  } catch {
    Write-Log ("GET failed: {0} => {1}" -f $url, $_.Exception.Message)
    return $null
  }
}

function Save-Json([object]$obj, [string]$path) {
  try { $obj | ConvertTo-Json -Depth 60 | Out-File -FilePath $path -Encoding utf8 }
  catch { Write-Log ("Save-Json failed: {0}" -f $_.Exception.Message) }
}

# ---- CSV helpers ----
function Parse-CsvText([string]$csvText) {
  if ([string]::IsNullOrWhiteSpace($csvText)) { return @() }
  try { return ($csvText | ConvertFrom-Csv) } catch { return @() }
}

function Last-OpenTradeRow([object[]]$rows, [string]$symbolPrefer="") {
  if ($rows.Count -eq 0) { return $null }
  $candidates = $rows | Where-Object { $_.status -and $_.status.ToString().ToLower() -eq "open" }
  if ($symbolPrefer) {
    $p = $candidates | Where-Object { $_.symbol -and $_.symbol.ToString().ToUpper() -eq $symbolPrefer.ToUpper() }
    if ($p.Count -gt 0) { $candidates = $p }
  }
  if ($candidates.Count -eq 0) { return $null }
  # rows are not guaranteed sorted; sort by timestamp
  return ($candidates | Sort-Object { [datetime]::Parse($_.timestamp) })[-1]
}

# ---- Log parsing: timeline around entry ±60s ----
$EventRegex = '"event"\s*:\s*"([^"]+)"'
$TsRegex    = '^(?<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})'
$Keywords   = @(
  'brackets\.reset\.cancelled',
  'brackets\.reset\.placed',
  'brackets\.reset\.skip',
  'brackets\.reset\.fallback_qty',
  'placing brackets failed',
  'entry\.aborted_not_filled'
)

function Build-Timeline([string[]]$lines, [datetime]$entryTs, [int]$windowSec=60) {
  if ($null -eq $lines -or $lines.Count -eq 0 -or $null -eq $entryTs) { return @() }
  $start = $entryTs.AddSeconds(-$windowSec)
  $end   = $entryTs.AddSeconds( $windowSec)
  $out = @()
  foreach ($ln in $lines) {
    $mTs = [regex]::Match($ln, $TsRegex)
    if (-not $mTs.Success) { continue }
    $ts = (Get-Date $mTs.Groups['ts'].Value)
    if ($ts -lt $start -or $ts -gt $end) { continue }
    $hit = $false
    foreach ($k in $Keywords) { if ($ln -match $k) { $hit = $true; break } }
    if (-not $hit) { continue }
    $ev = "text"
    $mEv = [regex]::Match($ln, $EventRegex)
    if ($mEv.Success) { $ev = $mEv.Groups[1].Value }
    $out += [PSCustomObject]@{ ts = $ts.ToString("o"); event = $ev; line = $ln }
  }
  return $out | Sort-Object ts
}

# ---- Step 1: Basic health/env/knobs ----
Write-Section "Health & Env"
$envObj   = Get-JsonObj "$Base/api/debug/env"
$knobs    = Get-JsonObj "$Base/api/debug/knobs"
$health   = Get-JsonObj "$Base/health"
Write-Log ("env: " + ($envObj | ConvertTo-Json -Depth 6))
Write-Log ("knobs: " + ($knobs  | ConvertTo-Json -Depth 6))

# ---- Step 2: Metrics snapshot (KPI / diagnostics / curve / calibration) ----
Write-Section "Metrics snapshot"
$metrics       = Get-JsonObj "$Base/api/metrics?limit=$Limit&include_open=false&bins=10"
$diagnostics   = Get-JsonObj "$Base/api/metrics/diagnostics?limit=$Limit"
$curve         = Get-JsonObj "$Base/api/metrics/curve?limit=$Limit"
$calibration   = Get-JsonObj "$Base/api/metrics/calibration?limit=$Limit&bins=10"

# ---- Step 3: Trades + full journal CSV + signals ----
Write-Section "Trades & Signals"
$trades        = Get-JsonObj "$Base/api/trades?limit=$Limit"
$tradesFull    = Get-JsonObj "$Base/api/trades_full?limit=$Limit"   # extra: reprices/used_market_fallback etc.
$journalCsvTxt = ""
try { $journalCsvTxt = Invoke-RestMethod -Method GET -Uri "$Base/api/journal/local.csv" -TimeoutSec 30 -ErrorAction Stop } catch { $journalCsvTxt = "" }
$journalRows   = Parse-CsvText $journalCsvTxt

$signals = [ordered]@{}
foreach($s in @($Sym1,$Sym2) | Select-Object -Unique) {
  if ([string]::IsNullOrWhiteSpace($s)) { continue }
  $signals[$s] = Get-JsonObj "$Base/api/signals/latest?symbol=$s"
}

# ---- Step 4: logs + open_orders + orders_history ----
Write-Section "Logs & Orders"
$logsObj = Get-JsonObj "$Base/api/logs?lines=$LogLines"
$lines   = @()
if ($logsObj -and $logsObj.lines) { $lines = @($logsObj.lines) }  # array of strings

# Determine entry timestamp (prefer: last OPEN row for Sym1; else override; else most recent closed)
$entryIso = $null
if (-not [string]::IsNullOrWhiteSpace($Since)) {
  $entryIso = $Since
} else {
  $lastOpen = $null
  if ($journalRows.Count -gt 0) { $lastOpen = Last-OpenTradeRow -rows $journalRows -symbolPrefer $Sym1 }
  if ($lastOpen) { $entryIso = $lastOpen.timestamp }
}
# Fallback: if still empty, try last trade row for Sym1 from trades_full
if (-not $entryIso) {
  if ($tradesFull -and $tradesFull.rows) {
    $rowsSym1 = $tradesFull.rows | Where-Object { $_.symbol -and $_.symbol.ToString().ToUpper() -eq $Sym1.ToUpper() }
    if ($rowsSym1) {
      $entryIso = ($rowsSym1 | Sort-Object { [datetime]::Parse($_.timestamp) } | Select-Object -Last 1).timestamp
    }
  }
}

# Build timeline around entry ±60s
$timeline = @()
$entryDt = $null
if ($entryIso) {
  try { $entryDt = Get-Date $entryIso } catch { $entryDt = $null }
  if ($entryDt) { $timeline = Build-Timeline -lines $lines -entryTs $entryDt -windowSec 60 }
}

# open_orders (key protected optional)
function OpenOrders-Url([string]$sym) {
  if ([string]::IsNullOrWhiteSpace($ReadApiKey)) { return "$Base/api/open_orders?symbol=$sym" }
  else { return "$Base/api/open_orders?symbol=$sym&key=$ReadApiKey" }
}
$openOrders = [ordered]@{}
foreach($s in @($Sym1,$Sym2) | Select-Object -Unique) {
  if ([string]::IsNullOrWhiteSpace($s)) { continue }
  $openOrders[$s] = Get-JsonObj (OpenOrders-Url $s)
}

# Count TP/SL orders (bracket sanity)
function Count-Brackets([object]$ordersObj) {
  $tp = 0; $sl = 0
  try {
    foreach($o in ($ordersObj.orders)) {
      $t = ($o.type ?? $o.Type ?? "")
      $t = $t.ToString().ToUpper()
      if ($t -like "TAKE_PROFIT*") { $tp++ }
      if ($t -eq "STOP" -or $t -eq "STOP_MARKET") { $sl++ }
    }
  } catch {}
  return [PSCustomObject]@{ tp=$tp; sl=$sl; ok = (($tp -eq 1) -and ($sl -eq 1)) }
}

$bracketCheck = [ordered]@{}
foreach($s in @($Sym1,$Sym2) | Select-Object -Unique) {
  if ($openOrders[$s]) { $bracketCheck[$s] = Count-Brackets $openOrders[$s] }
}

# orders_history since entry
$ordersHistory = $null
if ($entryIso) { $ordersHistory = Get-JsonObj "$Base/api/orders/history?symbol=$Sym1&limit=100&since=$entryIso" }

# ---- Step 5: trade_labels (label vs exchange vs pnl sign) ----
$tradeLabels = Get-JsonObj "$Base/api/diagnostics/trade_labels?limit=500"

# ---- Step 6: optional GCS snapshots ----
$gcsSnaps = $null
if (-not [string]::IsNullOrWhiteSpace($GcsDate)) {
  $gcsSnaps = Get-JsonObj "$Base/api/gcs/snapshots?dataset=trades&date=$GcsDate&limit=200"
}

# ---- Step 7: optional /tasks/* (for controlled test only) ----
$tasks = $null
if ($Cron) {
  $tasks = @{
    trader    = Get-JsonObj "$Base/tasks/trader"
    maintain  = Get-JsonObj "$Base/tasks/maintain"
    journal   = Get-JsonObj "$Base/tasks/journal_sync?mode=backup"
  }
}

# ---- Step 8: Compose final bundle (WHAT YOU MUST RECEIVE) ----
$bundle = [ordered]@{}
$bundle.url           = $Base
$bundle.env           = $envObj
$bundle.knobs         = $knobs
$bundle.logs          = $logsObj
$bundle.metrics       = $metrics
$bundle.diagnostics   = $diagnostics
$bundle.signals       = @{}
$bundle.signals[$Sym1] = $signals[$Sym1]
$bundle.signals[$Sym2] = $signals[$Sym2]
$bundle.open_orders   = @{}
$bundle.open_orders[$Sym1] = $openOrders[$Sym1]
$bundle.open_orders[$Sym2] = $openOrders[$Sym2]
$bundle.orders_history = $ordersHistory
$bundle.trades        = $trades
$bundle.curve         = $curve
$bundle.calibration   = $calibration
$bundle.trade_labels  = $tradeLabels
if ($gcsSnaps) { $bundle.gcs_snapshots = $gcsSnaps }  # optional

# ---- Extra evidence fields for today's issue ----
$extra = [ordered]@{}
$extra.entry_ts_iso   = $entryIso
$extra.timeline_around_entry = $timeline
$extra.bracket_check  = $bracketCheck
$extra.trades_full    = $tradesFull  # includes reprices / used_market_fallback / mode
$extra.journal_csv_included = ($journalRows.Count -gt 0)
$bundle.extra         = $extra

# ---- Save ----
Save-Json $bundle $JsonFile
Write-Host "Saved JSON: $JsonFile" -ForegroundColor Cyan
Write-Log  "Saved JSON: $JsonFile"
Write-Host "Saved LOG : $LogFile" -ForegroundColor Cyan
Write-Log  "Saved LOG : $LogFile"
Write-Host "=== Done ===" -ForegroundColor Green
