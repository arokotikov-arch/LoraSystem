$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $root
if (!(Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python 3.10/3.11 не найден в PATH.' }
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
# CUDA-compatible PyTorch wheel. NVIDIA driver must be up-to-date.
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path 'vendor\diffusers\examples\text_to_image\train_text_to_image_lora.py')) {
  New-Item -ItemType Directory -Force vendor | Out-Null
  git clone --depth 1 https://github.com/huggingface/diffusers.git vendor\diffusers
}
& .\.venv\Scripts\accelerate.exe config default
Write-Host "`nГотово. Запустите scripts\start_ui.bat" -ForegroundColor Green
