$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error 'Project virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"'
    exit 1
}

Push-Location -LiteralPath $projectRoot
try {
    & $python -m employee_monitor.cli @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
