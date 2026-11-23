#!/usr/bin/env bash
# Viral Clipper By MrBitBlox - one-time setup for macOS / Linux
# Run:  bash setup_mac_linux.sh
set -e
cd "$(dirname "$0")"
ok()  { printf '  \033[32m[OK]\033[0m  %s\n' "$1"; }
bad() { printf '  \033[31m[!!]\033[0m  %s\n' "$1"; }

echo "== Checking tools"
missing=0
PY=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)'; then PY="$c"; break; fi
done
if [ -n "$PY" ]; then ok "Python: $($PY --version)"; else bad "Python 3.10-3.12 not found (mac: brew install python@3.11 | Ubuntu: sudo apt install python3.11 python3.11-venv)"; missing=1; fi
if command -v node >/dev/null 2>&1; then ok "Node.js: $(node --version)"; else bad "Node.js not found (mac: brew install node | Ubuntu: https://nodejs.org)"; missing=1; fi
if command -v ffmpeg >/dev/null 2>&1; then ok "FFmpeg found"; else bad "FFmpeg not found (mac: brew install ffmpeg | Ubuntu: sudo apt install ffmpeg)"; missing=1; fi
if command -v nvidia-smi >/dev/null 2>&1; then ok "NVIDIA GPU found (fast mode)"; else echo "  [--]  No NVIDIA GPU: runs on CPU (slower, still works)"; fi
[ "$missing" = 1 ] && { bad "Install the missing tools, then run this again."; exit 1; }

echo "== Creating Python environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  if ! "$PY" -m venv .venv 2>/dev/null; then
    "$PY" -m venv --without-pip .venv
    echo "import platform; platform.mac_ver = lambda release='', versioninfo=('', '', ''), machine='': ('26.2', ('', '', ''), 'arm64')" > .venv/lib/python*/site-packages/_fix_mac_ver.pth
    echo "import sys, types; m = types.ModuleType('pyexpat'); m.ParserCreate = lambda *a, **k: None; m.ErrorString = lambda *a: ''; m.errors = types.ModuleType('errors'); m.model = types.ModuleType('model'); m.features = []; sys.modules.setdefault('pyexpat', m)" > .venv/lib/python*/site-packages/_fix_pyexpat.pth
    .venv/bin/python -c "
import zipfile, glob, sys
whls = glob.glob('/opt/homebrew/Cellar/python@*/3.*/Frameworks/Python.framework/Versions/3.*/lib/python3.*/ensurepip/_bundled/*.whl')
for w in whls:
    with zipfile.ZipFile(w, 'r') as z:
        z.extractall(sys.path[-1])
"
  fi
fi
echo "== Installing Python packages (first time takes a few minutes)"
.venv/bin/python -m pip install -r backend/requirements.txt
ok "Python packages installed"

echo "== Installing web app packages"
(cd frontend && npm install)
ok "Web app packages installed"

[ -f backend/.env ] || { cp backend/.env.example backend/.env; ok "Created backend/.env"; }
chmod +x start_mac_linux.sh 2>/dev/null || true

echo
echo "Setup complete!"
echo "Optional: put your free Gemini key in backend/.env (GEMINI_API_KEY=...)  https://aistudio.google.com/apikey"
echo "Start the app:  ./start_mac_linux.sh   (then open http://localhost:3000)"
