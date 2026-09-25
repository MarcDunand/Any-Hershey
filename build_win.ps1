# build_win.ps1
# Rebuilds the Windows executable using PyInstaller, then zips it for release
# as dist\AnyHershey_windows.zip

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
  --name "AnyHershey" `
  --add-data "language_presets.json;." `
  --collect-all vpype `
  --collect-all vpype_cli `
  --collect-all shapely `
  --collect-all pnoise `
  --copy-metadata pnoise `
  --hidden-import "3c22db458360489351e4__mypyc" `
  .\anyhershey.py

# End-user instructions ship next to the exe
Copy-Item .\README_windows.txt .\dist\AnyHershey\README.txt

# Zip for the GitHub release
Compress-Archive -Path .\dist\AnyHershey -DestinationPath .\dist\AnyHershey_windows.zip -Force
