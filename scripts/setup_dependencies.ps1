# PowerShell script to set up dependencies for Heartbeat on Windows
# Run as Administrator for system-wide installation

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Heartbeat Dependency Setup (Windows)" -ForegroundColor Cyan  
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check for winget
$hasWinget = Get-Command winget -ErrorAction SilentlyContinue

# 1. Install espeak-ng
Write-Host "[1/4] Installing espeak-ng..." -ForegroundColor Yellow

$espeakPath = "C:\Program Files\eSpeak NG"
if (Test-Path $espeakPath)
{
    Write-Host "  [OK] espeak-ng already installed at $espeakPath" -ForegroundColor Green
}
else
{
    if ($hasWinget)
    {
        Write-Host "  Installing via winget..." -ForegroundColor Gray
        winget install -e --id eSpeak-NG.eSpeak-NG --accept-package-agreements --accept-source-agreements
    }
    else
    {
        Write-Host "  [!] winget not available. Please download manually from:" -ForegroundColor Red
        Write-Host "    https://github.com/espeak-ng/espeak-ng/releases" -ForegroundColor Cyan
        Write-Host "    Download: espeak-ng-X64.msi" -ForegroundColor Gray
    }
}

# 2. Install Python dependencies
Write-Host ""
Write-Host "[2/4] Installing Python dependencies..." -ForegroundColor Yellow

$pythonDeps = @("torch", "numpy", "gguf", "huggingface_hub")

foreach ($dep in $pythonDeps)
{
    Write-Host "  Installing $dep..." -ForegroundColor Gray
    python -m pip install $dep --quiet --upgrade
}
Write-Host "  [OK] Python dependencies installed" -ForegroundColor Green

# 3. Clone GGML submodule
Write-Host ""
Write-Host "[3/4] Setting up GGML submodule..." -ForegroundColor Yellow

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ggmlPath = Join-Path $projectRoot "extern\ggml"

if (Test-Path (Join-Path $ggmlPath "CMakeLists.txt"))
{
    Write-Host "  [OK] GGML already cloned" -ForegroundColor Green
}
else
{
    Write-Host "  Cloning ggml..." -ForegroundColor Gray
    $externPath = Join-Path $projectRoot "extern"
    if (-not (Test-Path $externPath))
    {
        New-Item -ItemType Directory -Force -Path $externPath | Out-Null
    }
    git clone --depth 1 https://github.com/ggerganov/ggml.git $ggmlPath
    Write-Host "  [OK] GGML cloned successfully" -ForegroundColor Green
}

# 4. Set up KissFFT
Write-Host ""
Write-Host "[4/4] Setting up KissFFT..." -ForegroundColor Yellow

$kissfftPath = Join-Path $projectRoot "extern\kissfft"

if (Test-Path (Join-Path $kissfftPath "kiss_fft.h"))
{
    Write-Host "  [OK] KissFFT already present" -ForegroundColor Green
}
else
{
    Write-Host "  Downloading KissFFT..." -ForegroundColor Gray
    if (-not (Test-Path $kissfftPath))
    {
        New-Item -ItemType Directory -Force -Path $kissfftPath | Out-Null
    }
    
    $kissfftUrl = "https://raw.githubusercontent.com/mborgerding/kissfft/master"
    Invoke-WebRequest -Uri "$kissfftUrl/kiss_fft.h" -OutFile (Join-Path $kissfftPath "kiss_fft.h")
    Invoke-WebRequest -Uri "$kissfftUrl/kiss_fft.c" -OutFile (Join-Path $kissfftPath "kiss_fft.c")
    Invoke-WebRequest -Uri "$kissfftUrl/_kiss_fft_guts.h" -OutFile (Join-Path $kissfftPath "_kiss_fft_guts.h")
    
    Write-Host "  [OK] KissFFT downloaded" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download model:  python scripts/download_model.py" -ForegroundColor White
Write-Host "  2. Export to GGUF:  python scripts/export_kokoro.py" -ForegroundColor White
Write-Host "  3. Build project:   mkdir build; cd build; cmake ..; cmake --build ." -ForegroundColor White
Write-Host ""

# Set environment variable for espeak-ng
$espeakDataPath = "C:\Program Files\eSpeak NG\espeak-ng-data"
if (Test-Path $espeakDataPath)
{
    [Environment]::SetEnvironmentVariable("ESPEAK_DATA_PATH", $espeakDataPath, "User")
    Write-Host "Set ESPEAK_DATA_PATH environment variable" -ForegroundColor Gray
}
