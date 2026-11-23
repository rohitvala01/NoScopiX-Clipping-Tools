#!/usr/bin/env bash
# Viral Clipper By MrBitBlox - start the app (macOS / Linux)
cd "$(dirname "$0")"
if [ ! -f .venv/bin/activate ]; then echo "Run setup first:  bash setup_mac_linux.sh"; exit 1; fi
. .venv/bin/activate
# Linux + NVIDIA: let faster-whisper find the pip-installed CUDA libraries
NV_LIBS=$(python - <<'EOF' 2>/dev/null
import os, glob
try:
    import nvidia
    print(":".join(d for b in nvidia.__path__ for d in glob.glob(os.path.join(b, "*", "lib"))))
except Exception:
    pass
EOF
)
[ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH}"
echo "==================================================="
echo "  Viral Clipper By MrBitBlox"
echo "  Open http://localhost:3000  (first load can take ~20 s)"
echo "  Press Ctrl+C to stop."
echo "==================================================="
mkdir -p backend/data
python backend/server.py > backend/data/server.log 2>&1 &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT INT TERM
cd frontend && npm run dev
