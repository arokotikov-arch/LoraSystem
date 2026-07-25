$root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $root
& .\.venv\Scripts\python.exe app\ui.py
