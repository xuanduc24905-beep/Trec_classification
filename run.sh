#!/usr/bin/env bash
# =====================================================================
#  TREC Question Classification — One-click launcher
#
#  Sử dụng:
#     bash run.sh              # mặc định FastAPI (web demo)
#     bash run.sh fastapi      # FastAPI version (http://localhost:8000)
#     bash run.sh streamlit    # Streamlit version (http://localhost:8501)
#     bash run.sh stop         # stop server đang chạy
#     bash run.sh health       # check health endpoint
# =====================================================================

set -e

MODE="${1:-fastapi}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/xuand/miniconda3/envs/NLP/bin/python"
STREAMLIT="/home/xuand/miniconda3/envs/NLP/bin/streamlit"
FASTAPI_PORT=8000
STREAMLIT_PORT=8765

# Màu cho output
G='\033[0;32m'   # green
Y='\033[1;33m'   # yellow
R='\033[0;31m'   # red
NC='\033[0m'

log()  { echo -e "${G}→${NC} $*"; }
warn() { echo -e "${Y}!${NC} $*"; }
err()  { echo -e "${R}✗${NC} $*" >&2; }

# ---------------------------------------------------------------------
check_env() {
    if [ ! -x "$PY" ]; then
        err "Không tìm thấy Python NLP env tại: $PY"
        err "Bạn cần conda env 'NLP' với torch 1.12 + allennlp 2.10."
        exit 1
    fi

    # Kiểm tra checkpoints
    if [ ! -f "$ROOT/checkpoints/model1_best.pt" ] && [ ! -f "$ROOT/checkpoints/model1_last.pt" ]; then
        warn "Không tìm thấy checkpoints/model1_*.pt — Model 1 (CNN) sẽ KHÔNG load."
    fi
    if [ ! -f "$ROOT/checkpoints/model2_best.pt" ] && [ ! -f "$ROOT/checkpoints/model2_last.pt" ]; then
        err "Không tìm thấy checkpoints/model2_*.pt"
        exit 1
    fi

    # Kiểm tra ELMo
    if [ ! -f "$ROOT/data/elmo/elmo_weights.hdf5" ]; then
        warn "Thiếu data/elmo/elmo_weights.hdf5 — start_server.py sẽ tự download (~360MB)."
    fi
}

# ---------------------------------------------------------------------
stop_servers() {
    log "Đang dừng các server cũ..."
    pkill -f "uvicorn.*app:app" 2>/dev/null && log "Đã stop FastAPI" || true
    pkill -f "streamlit run app.py" 2>/dev/null && log "Đã stop Streamlit" || true
    pkill -f "start_server.py" 2>/dev/null || true
    sleep 1
    log "OK."
}

# ---------------------------------------------------------------------
check_health() {
    if curl -sf "http://localhost:${FASTAPI_PORT}/health" >/dev/null 2>&1; then
        echo -e "${G}FastAPI${NC} đang chạy @ http://localhost:${FASTAPI_PORT}/"
        curl -s "http://localhost:${FASTAPI_PORT}/health" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:${FASTAPI_PORT}/health"
    else
        echo -e "${Y}FastAPI${NC} không chạy."
    fi
    if curl -sf "http://localhost:${STREAMLIT_PORT}/" >/dev/null 2>&1; then
        echo -e "${G}Streamlit${NC} đang chạy @ http://localhost:${STREAMLIT_PORT}/"
    else
        echo -e "${Y}Streamlit${NC} không chạy."
    fi
}

# ---------------------------------------------------------------------
run_fastapi() {
    check_env
    log "Khởi động FastAPI server @ http://localhost:${FASTAPI_PORT}/ ..."
    echo ""
    echo -e "   ${G}Demo UI :${NC} http://localhost:${FASTAPI_PORT}/"
    echo -e "   ${G}API docs:${NC} http://localhost:${FASTAPI_PORT}/docs"
    echo -e "   ${G}Health  :${NC} http://localhost:${FASTAPI_PORT}/health"
    echo ""
    log "Lần đầu chạy mất ~1-2 phút (load ELMo + vinai translator)."
    log "Nhấn Ctrl+C để dừng."
    echo ""
    cd "$ROOT/fastapi_app"
    exec "$PY" start_server.py
}

# ---------------------------------------------------------------------
run_streamlit() {
    check_env
    log "Khởi động Streamlit @ http://localhost:${STREAMLIT_PORT}/ ..."
    echo ""
    cd "$ROOT"
    exec "$STREAMLIT" run app.py \
        --server.headless true \
        --server.port "$STREAMLIT_PORT" \
        --browser.gatherUsageStats false
}

# ---------------------------------------------------------------------
case "$MODE" in
    fastapi|api|web)
        run_fastapi
        ;;
    streamlit|st)
        run_streamlit
        ;;
    stop|kill)
        stop_servers
        ;;
    health|status|check)
        check_health
        ;;
    *)
        echo "Sử dụng: bash run.sh [fastapi|streamlit|stop|health]"
        echo ""
        echo "  fastapi   - FastAPI server  (port ${FASTAPI_PORT})  [mặc định]"
        echo "  streamlit - Streamlit demo  (port ${STREAMLIT_PORT})"
        echo "  stop      - Stop server đang chạy"
        echo "  health    - Check trạng thái server"
        exit 1
        ;;
esac
