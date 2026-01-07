# build_win.ps1
# Rebuilds the Windows executable using PyInstaller

$ErrorActionPreference = "Stop"

# Activate venv
.\venv\Scripts\Activate.ps1

# Ensure build tools are up to date
python -m pip install --upgrade pip pyinstaller

# Build app (onedir)
python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name "txt2svg" `
  --add-data "language_presets.json;." `
  --collect-all vpype `
  --collect-all vpype_cli `
  --collect-all shapely `
  --collect-all pnoise `
  --copy-metadata pnoise `
  --hidden-import "3c22db458360489351e4__mypyc" `
  .\text_to_centerline_svg.py
