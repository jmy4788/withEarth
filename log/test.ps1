# ===== 0) 환경 =====
$env:BASE   = "https://vaulted-scholar-466013-r5.appspot.com"   # GAE 베이스 URL
$env:REGION = "asia-northeast3"
# 배포된 Cloud Run URL 자동 가져오기(이미 설정되어 있으면 생략 가능)
try { $env:TSFM_URL = (gcloud run services describe tsfm-svc --region $env:REGION --format "value(status.url)").Trim() } catch {}

# ===== 1) 공통 유틸 =====
function Get-Json($url, $Headers=@{}) {
  try { Invoke-RestMethod -UseBasicParsing -Uri $url -Method GET -Headers $Headers -TimeoutSec 30 }
  catch { @{ error = $_.Exception.Message; url = $url } }
}
function Post-Json($url, $obj, $Headers=@{}) {
  try {
    $json = $obj | ConvertTo-Json -Depth 10
    Invoke-RestMethod -UseBasicParsing -Uri $url -Method POST -Headers ($Headers + @{"Content-Type"="application/json"}) -Body $json -TimeoutSec 60
  } catch { @{ error = $_.Exception.Message; url = $url } }
}

# ===== 2) GAE TRIAGE 수집 =====
function Get-WithEarthTriage {
  param(
    [string]$Base = $env:BASE,
    [string]$Out  = ".\triage_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"),
    [int]$LogLines = 400
  )
  $cron = @{ "X-Appengine-Cron"="true" }  # /tasks/* 전용

  $triage = [ordered]@{
    meta = @{
      project = "vaulted-scholar-466013-r5"
      base    = $Base
      ts      = (Get-Date).ToString("o")
      sym1    = "BTCUSDT"; sym2="ETHUSDT"; limit=2000
    }
    env         = Get-Json "$Base/api/debug/env"
    knobs       = Get-Json "$Base/api/debug/knobs"
    logs        = Get-Json "$Base/api/logs?lines=$LogLines"
    sig_btc     = Get-Json "$Base/api/signals/latest?symbol=BTCUSDT"
    sig_eth     = Get-Json "$Base/api/signals/latest?symbol=ETHUSDT"
    metrics     = Get-Json "$Base/api/metrics"
    diagnostics = Get-Json "$Base/api/metrics/diagnostics"
    calib       = Get-Json "$Base/api/metrics/calibration"
    # 옵션: 저널 동기화(읽기/백업). Cron 헤더 필수
    task_journal_sync = Post-Json "$Base/tasks/journal_sync?mode=backup" $null $cron
  }

  $triage | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $Out
  Write-Host "Saved -> $Out"
  return $triage
}

# ===== 3) TSFM Cloud Run Quick Test =====
function Test-TsfmService {
  param(
    [string]$Url = $env:TSFM_URL
  )
  if (-not $Url) { Write-Warning "TSFM_URL 비어있음"; return }
  Write-Host "Cloud Run URL =" $Url

  # 3-1) 헬스/루트 확인 (라우트 없으면 404 가능)
  $root   = Get-Json "$Url/"
  $health = Get-Json "$Url/health"
  $hz     = Get-Json "$Url/healthz"

  # 3-2) /predict 호출 (TSFM 클라이언트가 기대하는 스키마)
  $payload = @{
    inputs     = @(@{ target = @(1,2,3,4,5,6,7,8,9,10) })
    parameters = @{ prediction_length = 12; quantiles = @(0.05,0.5,0.95) }
  }
  $pred = Post-Json "$Url/predict" $payload

  [ordered]@{
    url = $Url
    root = $root
    health = $health
    healthz = $hz
    predict = $pred
  } | ConvertTo-Json -Depth 10
}

# ===== 4) 실행 예시 =====
# triage 수집 → 결과 JSON을 새 대화창에 붙여넣어 주세요.
# $tri = Get-WithEarthTriage

# Cloud Run TSFM 헬스/예측 확인
Test-TsfmService
