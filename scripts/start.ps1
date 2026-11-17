# Windows start script. Run from the project folder in PowerShell:  .\scripts\start.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = "python"
if (Test-Path ".\venv\Scripts\python.exe") { $py = ".\venv\Scripts\python.exe" }

try {
    Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/ | Out-Null
} catch {
    Write-Host "Ollama is not running. Start it from the Start menu, then run this again."
    exit 1
}

& $py -m app.server
