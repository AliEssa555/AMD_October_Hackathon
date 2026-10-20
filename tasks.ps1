<#
.SYNOPSIS
  Windows stand-in for the Makefile (no `make` on this machine).
  Same target names:  .\tasks.ps1 lint   .\tasks.ps1 test   .\tasks.ps1 data
#>
param(
    [Parameter(Position = 0)][string]$Target = 'help',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Prefer the project venv so targets work without manual activation.
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$py = if (Test-Path $venvPy) { $venvPy } else { 'python' }
$venvBin = Join-Path $PSScriptRoot '.venv\Scripts'
if (Test-Path $venvBin) { $env:PATH = "$venvBin;$env:PATH" }

# Load .env into the process (same vars the Makefile exports).
$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            Set-Item -Path "env:$($Matches[1])" -Value $Matches[2].Trim('"')
        }
    }
}

function Invoke-Step($cmd, $cmdArgs) {
    Write-Host "> $cmd $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    & $cmd @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "$cmd failed with exit code $LASTEXITCODE" }
}

switch ($Target) {
    'setup' {
        Invoke-Step 'py' @('-3.11', '-m', 'venv', '.venv')
        Invoke-Step $venvPy @('-m', 'pip', 'install', '-U', 'pip')
        Invoke-Step $venvPy @('-m', 'pip', 'install', '-e', '.[dev,train]')
        Invoke-Step $venvPy @('-m', 'pre_commit', 'install')
    }
    'lint' {
        Invoke-Step 'ruff' @('check', '.')
        Invoke-Step 'ruff' @('format', '--check', '.')
    }
    'fmt' {
        Invoke-Step 'ruff' @('check', '--fix', '.')
        Invoke-Step 'ruff' @('format', '.')
    }
    'test'     { Invoke-Step $py @('-m', 'pytest', '-m', 'not slow') }
    'test-all' { Invoke-Step $py @('-m', 'pytest') }
    'data'     { Invoke-Step $py @('-m', 'dvc', 'repro') }
    'smoke' {
        Invoke-Step $py @('-m', 'yolo_der.train', '--config', 'configs/baseline.yaml',
            '--seed', '0', '--fraction', '0.02', '--epochs-override', '1')
    }
    'tunnel' {
        Invoke-Step 'ssh' @('-N', '-L', "5000:$($env:HUB_PRIVATE_IP):5000", "root@$($env:HUB_PUBLIC_IP)")
    }
    'export'  { Invoke-Step $py (@('-m', 'yolo_der.export') + $Rest) }
    'api-dev' { Push-Location api; try { Invoke-Step 'npm' @('run', 'dev') } finally { Pop-Location } }

    # --- browsing the runs pulled off the Droplet -----------------------------
    # The Droplet is gone; these read the copy scripts/collect_results.py made.
    'runs'     { Invoke-Step $py (@('-m', 'scripts.view_runs') + $Rest) }
    'runs-sum' { Invoke-Step $py @('-m', 'scripts.view_runs', '--summary') }
    'curve'    { Invoke-Step $py (@('-m', 'scripts.view_runs', '--curves') + $Rest) }
    'report'   {
        Invoke-Step $py @('-m', 'yolo_der.evaluate', '--aggregate', '--reports', 'reports/eval', '--out', 'reports')
        Invoke-Step $py @('-m', 'yolo_der.analysis.plots', '--reports', 'reports', '--out', 'reports/figures')
    }

    # --- cloud -----------------------------------------------------------------
    # launch_gpu.sh and reaper.sh were deleted, not stubbed: both need a DO API
    # token, which this account cannot issue (CLAUDE.md, approved deviation from
    # rule 0.2). Launching is a console action documented in docs/MANUAL_LAUNCH.md.
    'send'    { Invoke-Step 'bash' (@('infra/send_repo.sh') + $Rest) }
    'fetch'   { Invoke-Step 'bash' (@('infra/fetch_results.sh') + $Rest) }
    'status'  { Invoke-Step 'bash' @('infra/status.sh') }
    default {
        Write-Host 'Targets:'
        Write-Host '  setup lint fmt test test-all data smoke'
        Write-Host '  runs            MLflow UI on the pulled store'
        Write-Host '  runs-sum        one-screen summary of every run'
        Write-Host '  curve <run>     per-epoch mAP50 for one run, e.g. .\tasks.ps1 curve der_enh-s0'
        Write-Host '  report          rebuild reports/ablation.md and the figures'
        Write-Host '  send <host>     copy the repo to a Droplet    fetch <host>  pull results back'
        Write-Host '  export api-dev status tunnel'
    }
}
