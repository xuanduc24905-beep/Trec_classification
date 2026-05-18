#!/usr/bin/env bash
# One-shot setup script — tạo venv, cài deps, sẵn sàng chạy `python start_server.py`.
# Usage:
#   bash setup.sh        # CPU-only torch
#   bash setup.sh gpu    # GPU torch (CUDA 11.6)

set -e

PY=python3.9
if ! command -v $PY >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info[:2]==(3,9)' 2>/dev/null; then
        PY=python3
    else
        echo "❌ Cần Python 3.9 (allennlp 2.10 không hỗ trợ ≥3.10)"
        echo "   Cài: sudo apt install python3.9 python3.9-venv  (Ubuntu)"
        echo "        brew install python@3.9  (macOS)"
        exit 1
    fi
fi

echo "✓ Python: $($PY --version)"

if [ ! -d .venv ]; then
    echo "→ Tạo venv ở .venv/"
    $PY -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip --quiet

if [ "$1" = "gpu" ]; then
    echo "→ Cài torch GPU (CUDA 11.6)..."
    pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "→ Cài torch CPU-only..."
    pip install torch==1.12.1
fi

echo "→ Cài phần còn lại từ requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Setup xong. Chạy server:"
echo "   source .venv/bin/activate"
echo "   python start_server.py"
