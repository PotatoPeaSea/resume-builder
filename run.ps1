# run.ps1 — Launch the Resume Builder app (Tauri dev)
#
# Sets up the portable toolchain PATH + native-build env vars, then runs
# `npm run tauri dev`. All tools were installed without admin rights.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1          # dev (default)
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 -Build   # production bundle

param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

# --- Toolchain locations (portable installs, see handoff.md) ---
$NodeDir  = "$env:LOCALAPPDATA\nodejs\node-v20.18.3-win-x64"
$CargoBin = "$env:USERPROFILE\.cargo\bin"
$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

# --- Sanity checks ---
foreach ($t in @(
    @{ Name = "Node";  Path = "$NodeDir\node.exe" },
    @{ Name = "Cargo"; Path = "$CargoBin\cargo.exe" }
)) {
    if (-not (Test-Path $t.Path)) {
        Write-Error "$($t.Name) not found at $($t.Path). See handoff.md for setup."
    }
}

# --- Environment ---
# NOTE: Local LLM generation now goes through Ollama over HTTP, so cmake/libclang
# are no longer needed to build (the llama-cpp dependency was removed).
$env:PATH = "$CargoBin;$NodeDir;$env:PATH"

# Ensure the MSVC Rust toolchain is the default (GNU/MSVC were both installed).
$active = (& "$CargoBin\rustup.exe" show active-toolchain 2>$null)
if ($active -notmatch "msvc") {
    Write-Host "Switching rustup default to MSVC toolchain..." -ForegroundColor Yellow
    & "$CargoBin\rustup.exe" default stable-x86_64-pc-windows-msvc | Out-Null
}

# Ensure the Ollama server is running (default LLM mode is "local" → Ollama).
if (Test-Path $OllamaExe) {
    $ollamaUp = $false
    try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 | Out-Null; $ollamaUp = $true } catch {}
    if (-not $ollamaUp) {
        Write-Host "Starting Ollama server..." -ForegroundColor Yellow
        Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Ollama not found ($OllamaExe). Local LLM mode needs Ollama installed + 'ollama pull qwen3.5:9b'." -ForegroundColor DarkYellow
}

Set-Location $PSScriptRoot

if ($Build) {
    Write-Host "Building production bundle (npm run tauri build)..." -ForegroundColor Cyan
    & "$NodeDir\npm.cmd" run tauri build
} else {
    Write-Host "Starting Resume Builder (npm run tauri dev)..." -ForegroundColor Cyan
    Write-Host "First build after a clean checkout takes a few minutes; subsequent runs are fast." -ForegroundColor DarkGray
    & "$NodeDir\npm.cmd" run tauri dev
}
