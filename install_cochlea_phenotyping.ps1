# Installs the Cochlea Phenotyping napari plugin and its measurement pipeline (Windows, no administrator rights needed).
#
# Run it from the folder that holds this file (the downloaded napari-cochlea-phenotyping repository, with the folders
# napari-cochlea-phenotyping, stereocilia and models):
#   powershell -ExecutionPolicy Bypass -File .\install_cochlea_phenotyping.ps1
#   powershell -ExecutionPolicy Bypass -File .\install_cochlea_phenotyping.ps1 -Rebuild     (start the environment over)
#   powershell -ExecutionPolicy Bypass -File .\install_cochlea_phenotyping.ps1 -CpuOnly     (skip the graphics card build)
#
# What it does
#   1. Uses an existing Miniforge or Anaconda install, or installs Miniforge for this user only.
#   2. Creates the environment "cochlea-phenotyping" (Python 3.12) and installs napari, the plugin and everything the
#      pipeline imports with pip. scikit-learn is held at the version the trained models were saved with.
#      If an NVIDIA graphics card is found, PyTorch is replaced by its graphics card build (CUDA 12.8).
#   3. Tells the plugin where the pipeline scripts and trained models are (COCHLEA_PIPELINE_SCRIPTS and
#      COCHLEA_PIPELINE_MODELS, set inside the Desktop shortcuts only, so other napari installs are not affected), downloads the Cellpose-SAM model (about 1.2 GB),
#      loads every trained model once and runs the plugin's own tests.
#   4. Puts two shortcuts on the Desktop: "Cochlea Phenotyping" opens napari with the panel, and
#      "Cochlea Phenotyping prompt" opens a command window with the environment ready.
# Everything is written to install_log.txt next to this file. Running it again is safe: finished parts are kept.

param([switch]$Rebuild, [switch]$CpuOnly)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Transcript -Path (Join-Path $Here "install_log.txt") -Append | Out-Null
$EnvName = "cochlea-phenotyping"
$SklearnVersion = "1.9.1"          # the version the models in .\models were saved with

function Step($msg) { Write-Host ""; Write-Host ("== " + $msg + "  (" + (Get-Date -Format "yyyy-MM-dd h:mm tt") + ")") -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw ("Command failed (exit code $LASTEXITCODE): $exe " + ($argv -join " ")) }
}

# ---------------------------------------------------------------- 0. the downloaded folder
# The repository root holds napari-cochlea-phenotyping, stereocilia and models; this file may sit in the root or in
# napari-cochlea-phenotyping.
$RepoRoot = if (Test-Path (Join-Path $Here "napari-cochlea-phenotyping\pyproject.toml")) { $Here } else { Split-Path -Parent $Here }
$PluginDir = Join-Path $RepoRoot "napari-cochlea-phenotyping"
$ScriptsDir = Join-Path $RepoRoot "stereocilia\scripts"
$ModelsDir = Join-Path $RepoRoot "models"
if (-not (Test-Path (Join-Path $ModelsDir "hair_cell_classifier_v2.joblib")) -and $env:COCHLEA_PIPELINE_MODELS) { $ModelsDir = $env:COCHLEA_PIPELINE_MODELS }
foreach ($p in @((Join-Path $PluginDir "pyproject.toml"), (Join-Path $ScriptsDir "run_pipeline.py"), (Join-Path $ModelsDir "hair_cell_classifier_v2.joblib"))) {
    if (-not (Test-Path $p)) { throw "Not found: $p. Run this file from the downloaded repository folder (it holds napari-cochlea-phenotyping, stereocilia and models)." }
}

# ---------------------------------------------------------------- 1. conda
Step "Looking for conda"
$candidates = @("$env:USERPROFILE\miniforge3", "$env:LOCALAPPDATA\miniforge3", "$env:USERPROFILE\anaconda3",
                "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\anaconda3", "$env:ProgramData\miniforge3", "$env:ProgramData\anaconda3")
$candidates += @("$env:USERPROFILE\AppData\Local\miniforge3", "$env:USERPROFILE\AppData\Local\anaconda3")
if ($env:CONDA_EXE) { $candidates = @((Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE))) + $candidates }
$onPath = Get-Command conda.exe -ErrorAction SilentlyContinue
if ($onPath) { $candidates = @((Split-Path -Parent (Split-Path -Parent $onPath.Source))) + $candidates }
$Root = $candidates | Where-Object { $_ -and (Test-Path (Join-Path $_ "Scripts\conda.exe")) } | Select-Object -First 1
if (-not $Root) {
    Step "Installing Miniforge for this user"
    $Root = "$env:USERPROFILE\miniforge3"
    $inst = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $inst
    Start-Process -FilePath $inst -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$Root" -Wait
    if (-not (Test-Path (Join-Path $Root "Scripts\conda.exe"))) { throw "Miniforge did not install to $Root" }
}
$Conda = Join-Path $Root "Scripts\conda.exe"
$EnvDir = Join-Path $Root "envs\$EnvName"
$Py = Join-Path $EnvDir "python.exe"
Write-Host "conda: $Conda"

function UseEnvPath {
    $env:PATH = (@($EnvDir, "$EnvDir\Library\mingw-w64\bin", "$EnvDir\Library\usr\bin", "$EnvDir\Library\bin",
                   "$EnvDir\Scripts", "$EnvDir\bin") -join ";") + ";" + $env:PATH
    $env:CONDA_PREFIX = $EnvDir
    $env:PYTHONPATH = ""
}
function Pip([string[]]$argv) { Run $Py (@("-m", "pip") + $argv) }
function PyCode([string]$code) {
    $f = Join-Path $env:TEMP ("cochlea_" + [guid]::NewGuid().ToString("N") + ".py")
    Set-Content -Path $f -Value $code -Encoding UTF8
    try { Run $Py @($f) } finally { Remove-Item $f -ErrorAction SilentlyContinue }
}

$nvidia = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" })
if ($CpuOnly) { $nvidia = @() }
if ($nvidia.Count -gt 0) { Write-Host ("NVIDIA graphics card found: " + $nvidia[0].Name) }
else { Write-Host "No NVIDIA graphics card used: the pipeline will run on the processor (slower, same results within rounding)." -ForegroundColor Yellow }

# ---------------------------------------------------------------- 2. environment
Step "Environment $EnvName (Python 3.12, packages from pip)"
$exists = Test-Path $Py
if ($exists -and $Rebuild) {
    Write-Host "Removing the earlier $EnvName environment so it can be built cleanly"
    Run $Conda @("env", "remove", "-y", "-n", $EnvName); $exists = $false
}
if (-not $exists) { Run $Conda @("create", "-y", "-n", $EnvName, "python=3.12", "pip", "-c", "conda-forge") }
UseEnvPath
Pip @("install", "--upgrade", "pip")
Pip @("install", "napari[all]", "scikit-learn==$SklearnVersion", "pytest")
Pip @("install", "-e", ($PluginDir + "[pipeline]"))
if ($nvidia.Count -gt 0) {
    # PyPI's PyTorch is processor-only on Windows; the graphics card build replaces it at the same version.
    # CUDA 12.8 builds also cover the newest (Blackwell) NVIDIA cards.
    $tv = (& $Py -c "import importlib.metadata as m; print(m.version('torch').split('+')[0] + ' ' + m.version('torchvision').split('+')[0])").Trim().Split(" ")
    Pip @("install", "--force-reinstall", "--no-deps", "torch==$($tv[0])", "torchvision==$($tv[1])", "--index-url", "https://download.pytorch.org/whl/cu128")
}

# ---------------------------------------------------------------- 3. folders, models, checks
Step "Telling the plugin where the scripts and models are"
$env:COCHLEA_PIPELINE_SCRIPTS = $ScriptsDir; $env:COCHLEA_PIPELINE_MODELS = $ModelsDir
Write-Host "scripts: $ScriptsDir"; Write-Host "models:  $ModelsDir"

Step "Checking that everything loads, and downloading the Cellpose-SAM model (about 1.2 GB, first time only)"
PyCode @"
import os, glob, warnings
import joblib, torch, napari, nd2, skimage, sklearn, statsmodels, matplotlib
from importlib.metadata import version
for p in ['napari', 'napari-cochlea-phenotyping', 'cellpose', 'torch', 'nd2', 'scikit-image', 'scikit-learn', 'statsmodels', 'numpy', 'pandas']:
    try: print(f'{p:28s} {version(p)}')
    except Exception: print(f'{p:28s} MISSING')
print('graphics card used by PyTorch:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
bad = []
for f in sorted(glob.glob(os.path.join(os.environ['COCHLEA_PIPELINE_MODELS'], '*.joblib'))):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try: joblib.load(f)
        except Exception as e: bad.append(f'{os.path.basename(f)}: {e}')
        if any('InconsistentVersion' in type(x.message).__name__ for x in w): bad.append(f'{os.path.basename(f)}: saved with another scikit-learn version')
print('trained models loaded:', len(glob.glob(os.path.join(os.environ['COCHLEA_PIPELINE_MODELS'], '*.joblib'))) - len(bad), 'of', len(glob.glob(os.path.join(os.environ['COCHLEA_PIPELINE_MODELS'], '*.joblib'))))
if bad: raise SystemExit('Problem with the trained models: ' + '; '.join(bad))
from cellpose import models
models.CellposeModel(gpu=False)
print('Cellpose-SAM model ready')
from napari_cochlea_phenotyping._run import find_scripts_folder
print('pipeline scripts found:', find_scripts_folder())
"@

Step "Running the plugin's own tests"
$env:QT_QPA_PLATFORM = "offscreen"
Push-Location $PluginDir
& $Py -m pytest -q -p no:cacheprovider tests
$testExit = $LASTEXITCODE
Pop-Location
Remove-Item Env:\QT_QPA_PLATFORM
if ($testExit -ne 0) { Write-Host "Some plugin tests failed; see install_log.txt. The plugin may still open." -ForegroundColor Yellow }
else { Write-Host "All plugin tests passed." -ForegroundColor Green }

# ---------------------------------------------------------------- 4. Desktop shortcuts
Step "Desktop shortcuts"
$Activate = Join-Path $Root "Scripts\activate.bat"
$Desktop = [Environment]::GetFolderPath("Desktop")
$napariBat = Join-Path $Desktop "Cochlea Phenotyping.bat"
$promptBat = Join-Path $Desktop "Cochlea Phenotyping prompt.bat"
$setVars = @("set COCHLEA_PIPELINE_SCRIPTS=$ScriptsDir", "set COCHLEA_PIPELINE_MODELS=$ModelsDir", "set PYTHONPATH=")
Set-Content -Path $napariBat -Encoding ASCII -Value (@("@echo off") + $setVars + @("call `"$Activate`" $EnvName", "napari -w napari-cochlea-phenotyping"))
Set-Content -Path $promptBat -Encoding ASCII -Value (@("@echo off") + $setVars + @("cmd /k `"call `"$Activate`" $EnvName`""))
Write-Host "Double-click 'Cochlea Phenotyping' on the Desktop to open napari with the panel."
Step "Done"
Stop-Transcript | Out-Null
