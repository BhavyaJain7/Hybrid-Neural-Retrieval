#!/usr/bin/env bash
# =============================================================================
# Neural Search — Master Control Script
#
# Commands:
#   ./run.sh setup              Create venv, install deps, download NLTK data
#   ./run.sh ingest             Ingest documents from data/documents/
#   ./run.sh ingest --reset     Wipe indexes and reingest
#   ./run.sh api                Start FastAPI server (foreground)
#   ./run.sh ui                 Start Streamlit dashboard (foreground)
#   ./run.sh start              Start API (background) + Streamlit (foreground)
#   ./run.sh stop               Stop background API process
#   ./run.sh eval               Run retrieval evaluation (P@K, MRR, nDCG)
#   ./run.sh eval --k 5         Evaluate at k=5
#   ./run.sh verify             Check BM25 and Qdrant index sync
#   ./run.sh test               Run all tests
#   ./run.sh test unit          Run unit tests only
#   ./run.sh test integration   Run integration tests only
#   ./run.sh test coverage      Run all tests with coverage report
#   ./run.sh clean              Wipe all indexes, snapshots, uploaded documents, and collection metadata
#   ./run.sh logs               Tail live API logs
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
SRC_DIR="$PROJECT_ROOT/src"
ENV_FILE="$PROJECT_ROOT/.env"
PID_FILE="$PROJECT_ROOT/.api.pid"
API_HOST="127.0.0.1"
API_PORT="8000"
API_LOG="$PROJECT_ROOT/logs/api.log"
DOCUMENTS_DIR="$PROJECT_ROOT/data/documents"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()     { echo -e "${CYAN}[neural-search]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*"; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
check_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        warn ".env not found — copying from .env.example"
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        warn "Fill in GROQ_API_KEY in .env before running ingest or api"
    fi
}

activate_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found — run './run.sh setup' first"
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
}

ensure_dirs() {
    mkdir -p \
        "$PROJECT_ROOT/logs" \
        "$PROJECT_ROOT/data/documents" \
        "$PROJECT_ROOT/data/qdrant" \
        "$PROJECT_ROOT/data/bm25_index" \
        "$PROJECT_ROOT/data/collections" \
        "$PROJECT_ROOT/data/snapshots"
}

# Locate the ui directory — supports ui/ at project root or src/ui/
find_ui_dir() {
    if [[ -f "$PROJECT_ROOT/ui/app.py" ]]; then
        echo "$PROJECT_ROOT/ui"
    elif [[ -f "$PROJECT_ROOT/src/ui/app.py" ]]; then
        echo "$PROJECT_ROOT/src/ui"
    else
        error "Cannot find ui/app.py — expected at $PROJECT_ROOT/ui/ or $PROJECT_ROOT/src/ui/"
    fi
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_setup() {
    log "Setting up Neural Search environment..."
    ensure_dirs
    check_env

    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    log "Installing dependencies..."
    if command -v uv &>/dev/null; then
        uv pip install -e "$PROJECT_ROOT[dev]"
    else
        pip install --upgrade pip -q
        pip install -e "$PROJECT_ROOT[dev]" -q
    fi

    log "Installing test dependencies..."
    pip install pytest-cov -q

    log "Downloading NLTK data..."
    python3 -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
"

    success "Setup complete"
    echo -e "\n${BOLD}Next steps:${RESET}"
    echo "  1. Add your GROQ_API_KEY to .env"
    echo "  2. Drop PDF/DOCX files into data/documents/"
    echo "  3. ./run.sh ingest"
    echo "  4. ./run.sh start"
}

cmd_ingest() {
    activate_venv
    check_env
    log "Starting document ingestion from: $DOCUMENTS_DIR"
    python3 "$PROJECT_ROOT/scripts/ingest_documents.py" \
        --input-dir "$DOCUMENTS_DIR" "$@"
    success "Ingestion complete"
}

cmd_api() {
    activate_venv
    check_env
    log "Starting FastAPI server at http://$API_HOST:$API_PORT"
    log "Swagger docs:  http://$API_HOST:$API_PORT/docs"
    log "Health check:  http://$API_HOST:$API_PORT/health"
    uvicorn neural_search.api.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        --log-level info
}

cmd_ui() {
    activate_venv
    local ui_dir
    ui_dir="$(find_ui_dir)"
    log "Starting Streamlit dashboard at http://localhost:8501"
    log "UI directory: $ui_dir"
    cd "$ui_dir"
    PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" streamlit run app.py \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false
}

cmd_start() {
    activate_venv
    check_env
    ensure_dirs

    # Start API in background if not already running
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "API already running (PID $(cat "$PID_FILE")) — skipping"
    else
        log "Starting FastAPI in background..."
        PYTHONPATH="$SRC_DIR" uvicorn neural_search.api.main:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level info \
            >> "$API_LOG" 2>&1 &
        echo $! > "$PID_FILE"
        success "API started (PID $(cat "$PID_FILE")) — logs: logs/api.log"

        # Wait up to 15s for API to be ready
        log "Waiting for API to be ready..."
        for i in {1..15}; do
            if curl -sf "http://$API_HOST:$API_PORT/health" &>/dev/null; then
                success "API is up at http://$API_HOST:$API_PORT"
                break
            fi
            if [[ $i -eq 15 ]]; then
                warn "API did not respond in time — check logs/api.log"
            fi
            sleep 1
        done
    fi

    # Start Streamlit in foreground — Ctrl+C triggers cleanup via trap
    local ui_dir
    ui_dir="$(find_ui_dir)"
    log "Starting Streamlit dashboard (Ctrl+C to stop both)..."
    trap cmd_stop EXIT
    cd "$ui_dir"
    PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" streamlit run app.py \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            success "API stopped (PID $PID)"
        else
            warn "API process not running"
        fi
        rm -f "$PID_FILE"
    else
        warn "No API PID file found — nothing to stop"
    fi
}

cmd_eval() {
    activate_venv
    check_env
    log "Running retrieval evaluation..."
    python3 "$PROJECT_ROOT/scripts/run_eval.py" "$@"
}

cmd_verify() {
    activate_venv
    log "Verifying index sync..."
    python3 "$PROJECT_ROOT/scripts/verify_index.py"
}

cmd_test() {
    activate_venv
    local scope="${1:-all}"
    shift 2>/dev/null || true

    case "$scope" in
        unit)
            log "Running unit tests..."
            PYTHONPATH="$SRC_DIR" pytest tests/unit/ -v --tb=short "$@"
            ;;
        integration)
            log "Running integration tests..."
            PYTHONPATH="$SRC_DIR" pytest tests/integration/ -v --tb=short "$@"
            ;;
        coverage)
            log "Running all tests with coverage report..."
            PYTHONPATH="$SRC_DIR" pytest tests/ \
                -v --tb=short \
                --cov=neural_search \
                --cov-report=term-missing \
                --cov-report=html:logs/coverage \
                "$@"
            success "HTML coverage report: logs/coverage/index.html"
            ;;
        all)
            log "Running all tests..."
            PYTHONPATH="$SRC_DIR" pytest tests/ -v --tb=short "$@"
            ;;
        *)
            log "Running: pytest $scope $*"
            PYTHONPATH="$SRC_DIR" pytest "$scope" -v --tb=short "$@"
            ;;
    esac
}

cmd_logs() {
    if [[ ! -f "$API_LOG" ]]; then
        warn "No log file at $API_LOG — has the API been started?"
        exit 1
    fi
    log "Tailing API logs (Ctrl+C to stop)..."
    tail -f "$API_LOG"
}

cmd_clean() {
    echo -e "${YELLOW}This will wipe ALL indexes, collection metadata, snapshots, and uploaded documents.${RESET}"
    echo -e "${YELLOW}This cannot be undone.${RESET}"
    read -rp "Continue? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        # Indexes and metadata
        rm -rf \
            "$PROJECT_ROOT/data/qdrant/"* \
            "$PROJECT_ROOT/data/bm25_index/"* \
            "$PROJECT_ROOT/data/collections/"* \
            "$PROJECT_ROOT/data/snapshots/"* \
            "$PROJECT_ROOT/data/learned_fusion/"* \
            "$PROJECT_ROOT/data/training_pairs.jsonl"

        # Uploaded documents — per-collection subdirectories
        if [[ -d "$PROJECT_ROOT/data/documents" ]]; then
            find "$PROJECT_ROOT/data/documents" -mindepth 1 -delete
        fi

        success "Wiped: qdrant, bm25_index, collections, snapshots, learned_fusion, training_pairs, documents"

        # Recreate empty directory structure so the app starts cleanly
        ensure_dirs
        success "Empty data directories recreated — ready for fresh ingest"
    else
        log "Aborted"
    fi
}

cmd_help() {
    echo -e "\n${BOLD}Neural Search — run.sh${RESET}"
    echo -e "${CYAN}Usage: ./run.sh <command> [options]${RESET}\n"
    echo -e "${BOLD}Setup${RESET}"
    echo "  setup                   Create venv, install deps, download NLTK data"
    echo ""
    echo -e "${BOLD}Data${RESET}"
    echo "  ingest                  Ingest documents from data/documents/"
    echo "  ingest --reset          Wipe indexes and reingest from scratch"
    echo "  verify                  Check BM25 and Qdrant index are in sync"
    echo "  clean                   Wipe all indexes, collections, snapshots, and uploaded documents"
    echo ""
    echo -e "${BOLD}Server${RESET}"
    echo "  api                     Start FastAPI server in foreground"
    echo "  ui                      Start Streamlit dashboard in foreground"
    echo "  start                   Start API (background) + UI (foreground)"
    echo "  stop                    Stop background API process"
    echo "  logs                    Tail live API logs"
    echo ""
    echo -e "${BOLD}Evaluation${RESET}"
    echo "  eval                    Run retrieval evaluation (P@K, MRR, nDCG)"
    echo "  eval --k 5              Evaluate at k=5"
    echo ""
    echo -e "${BOLD}Testing${RESET}"
    echo "  test                    Run all tests"
    echo "  test unit               Run unit tests only (fast, no I/O)"
    echo "  test integration        Run integration tests only"
    echo "  test coverage           Run all tests with HTML coverage report"
    echo "  test <path>             Run a specific test file or directory"
    echo "  test unit -k metrics    Run tests matching 'metrics' in unit suite"
    echo ""
}

# ── Entrypoint ────────────────────────────────────────────────────────────────
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    setup)       cmd_setup ;;
    ingest)      cmd_ingest "$@" ;;
    api)         cmd_api ;;
    ui)          cmd_ui ;;
    start)       cmd_start ;;
    stop)        cmd_stop ;;
    eval)        cmd_eval "$@" ;;
    verify)      cmd_verify ;;
    test)        cmd_test "${@:-all}" ;;
    logs)        cmd_logs ;;
    clean)       cmd_clean ;;
    help|--help) cmd_help ;;
    *)
        error "Unknown command: '$COMMAND' — run './run.sh help' for usage"
        ;;
esac
