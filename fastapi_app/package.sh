#!/usr/bin/env bash
# Đóng gói project thành 1 zip để gửi cho người khác.
#
# Usage:
#   bash package.sh           # full bundle (kèm ELMo + checkpoints, ~430MB)
#   bash package.sh slim      # slim (không kèm ELMo — người nhận tự download, ~70MB)

set -e

MODE="${1:-full}"
OUT="trec-qc-fastapi-${MODE}.tar.gz"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT/fastapi_app"

cd "$ROOT"

# File list
FILES=(
    "fastapi_app/app.py"
    "fastapi_app/start_server.py"
    "fastapi_app/setup.sh"
    "fastapi_app/index.html"
    "fastapi_app/Dockerfile"
    "fastapi_app/.dockerignore"
    "fastapi_app/requirements.txt"
    "fastapi_app/README.md"
    "fastapi_app/models/__init__.py"
    "fastapi_app/models/classifier.py"
    "fastapi_app/models/translator.py"
    "checkpoints/model1_best.pt"
    "checkpoints/model2_best.pt"
)

if [ "$MODE" = "full" ]; then
    FILES+=(
        "data/elmo/elmo_options.json"
        "data/elmo/elmo_weights.hdf5"
    )
fi

# Check files exist
MISSING=()
for f in "${FILES[@]}"; do
    [ ! -f "$f" ] && MISSING+=("$f")
done
if [ ${#MISSING[@]} -ne 0 ]; then
    echo "❌ Thiếu file:"
    for f in "${MISSING[@]}"; do echo "   - $f"; done
    exit 1
fi

echo "→ Đóng gói $MODE bundle vào $OUT ..."
rm -f "$OUT"
tar --exclude='*/__pycache__' -czf "$OUT" "${FILES[@]}"

SIZE=$(du -h "$OUT" | cut -f1)
echo ""
echo "✅ Xong: $OUT ($SIZE)"
echo ""
echo "Người nhận chạy:"
echo "  tar -xzf $OUT && cd fastapi_app"
echo "  bash setup.sh         # (hoặc 'bash setup.sh gpu' nếu có GPU)"
echo "  source .venv/bin/activate"
echo "  python start_server.py"
