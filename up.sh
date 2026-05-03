#!/usr/bin/env bash
# up.sh — bring up the whisper-asr-webservice stack locally (no Docker).
#
# Checks prerequisites (python 3.11+, uv, ffmpeg, ffprobe), runs `uv sync` to
# install deps, seeds a `.env` from `example.env` if you haven't yet, and then
# starts the backend (port 9000) and frontend (port 9001) as background
# processes whose output is tee'd into `./output/logs/{backend,frontend}.log`.
# Hit Ctrl+C to stop both cleanly.
#
# Usage:
#   ./up.sh               # start everything
#   ./up.sh --no-sync     # skip `uv sync` (faster restarts)
#   BACKEND_PORT=9100 FRONTEND_PORT=9101 ./up.sh
#
# Env overrides (with defaults):
#   BACKEND_PORT   (9000)
#   FRONTEND_PORT  (9001)
#   HOST           (127.0.0.1)
#   LOG_DIR        (./output/logs)

set -euo pipefail

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'
  DIM=$'\033[2m'
  RED=$'\033[31m'
  GREEN=$'\033[32m'
  YELLOW=$'\033[33m'
  BLUE=$'\033[34m'
  RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

info()  { printf "%s›%s %s\n" "$BLUE" "$RESET" "$*"; }
ok()    { printf "%s✓%s %s\n" "$GREEN" "$RESET" "$*"; }
warn()  { printf "%s!%s %s\n" "$YELLOW" "$RESET" "$*"; }
err()   { printf "%s✗%s %s\n" "$RED" "$RESET" "$*" >&2; }
die()   { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Args + env defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-9000}"
FRONTEND_PORT="${FRONTEND_PORT:-9001}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/output/logs}"
SKIP_SYNC=0

while (( $# )); do
  case "$1" in
    --no-sync) SKIP_SYNC=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

missing=()

check_bin() {
  local name="$1" install_hint="$2"
  if command -v "$name" >/dev/null 2>&1; then
    ok "$name found at $(command -v "$name")"
  else
    err "$name is not installed or not on PATH"
    printf "  %s└%s install with: %s\n" "$DIM" "$RESET" "$install_hint"
    missing+=("$name")
  fi
}

info "Checking prerequisites…"
check_bin uv      "curl -LsSf https://astral.sh/uv/install.sh | sh"
check_bin ffmpeg  "brew install ffmpeg   # or apt-get install ffmpeg"
check_bin ffprobe "installed alongside ffmpeg"

# Python: uv manages its own interpreter via `uv python`, so a system python3
# isn't strictly required. But we still want to warn if the user doesn't have
# a working python3 anywhere — catches "accidentally on a minimal container"
# type cases.
if command -v python3 >/dev/null 2>&1; then
  py_version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")
  ok "python3 found: $(command -v python3) ($py_version)"
  case "$py_version" in
    3.11|3.12)
      :
      ;;
    ?)
      warn "could not determine python3 version"
      ;;
    *)
      warn "system python3 is $py_version — project requires >=3.11,<3.13. uv will install the right interpreter automatically."
      ;;
  esac
else
  warn "python3 is not on PATH. uv will install a managed interpreter if needed."
fi

if (( ${#missing[@]} )); then
  err "Missing required binaries: ${missing[*]}"
  err "Install the missing tools above and re-run ./up.sh"
  exit 1
fi

# ---------------------------------------------------------------------------
# Port availability
# ---------------------------------------------------------------------------

port_busy() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    # Fallback: try to bind with /dev/tcp (bash builtin). If this succeeds the
    # port is free; we treat failure as "busy or unknown".
    (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1 && return 0 || return 1
  fi
}

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if port_busy "$port"; then
    die "Port $port is already in use. Stop the existing process or set BACKEND_PORT / FRONTEND_PORT and retry."
  fi
done
ok "Ports $BACKEND_PORT (backend) and $FRONTEND_PORT (frontend) are free"

# ---------------------------------------------------------------------------
# .env seeding
# ---------------------------------------------------------------------------

if [[ ! -f .env && -f example.env ]]; then
  cp example.env .env
  warn "Created .env from example.env. Edit it to set HF_TOKEN if you need diarization."
fi

# ---------------------------------------------------------------------------
# Install deps
# ---------------------------------------------------------------------------

if (( SKIP_SYNC )); then
  info "Skipping uv sync (--no-sync)"
else
  info "Running uv sync…"
  uv sync
  ok "Dependencies installed"
fi

# ---------------------------------------------------------------------------
# Start services
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
: > "$BACKEND_LOG"
: > "$FRONTEND_LOG"

cleanup() {
  local code=$?
  echo
  info "Shutting down…"
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  # Give them a moment to exit gracefully before escalating.
  for _ in 1 2 3 4 5; do
    if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then sleep 0.2; else break; fi
  done
  [[ -n "${BACKEND_PID:-}" ]] && kill -9 "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill -9 "$FRONTEND_PID" 2>/dev/null || true
  ok "Stopped. Logs remain at $LOG_DIR/"
  exit "$code"
}
trap cleanup INT TERM EXIT

info "Starting backend on http://$HOST:$BACKEND_PORT …"
uv run uvicorn app.webservice:app \
  --host "$HOST" --port "$BACKEND_PORT" \
  --log-level info \
  >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

info "Starting frontend on http://$HOST:$FRONTEND_PORT …"
uv run uvicorn frontend.app:app \
  --host "$HOST" --port "$FRONTEND_PORT" \
  --log-level warning \
  >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

# Wait for the backend to open its port before declaring victory. Backend can
# take a moment because it eagerly loads the default WhisperX model.
info "Waiting for backend to become ready (this warms the default model)…"
for i in $(seq 1 120); do
  if port_busy "$BACKEND_PORT"; then
    ok "Backend is listening"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    err "Backend exited early. Tail of $BACKEND_LOG:"
    tail -n 40 "$BACKEND_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

for i in $(seq 1 30); do
  if port_busy "$FRONTEND_PORT"; then
    ok "Frontend is listening"
    break
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    err "Frontend exited early. Tail of $FRONTEND_LOG:"
    tail -n 40 "$FRONTEND_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

echo
printf "%sReady:%s\n" "$BOLD" "$RESET"
printf "  %sFrontend UI %s → http://%s:%s\n" "$GREEN" "$RESET" "$HOST" "$FRONTEND_PORT"
printf "  %sAPI docs    %s → http://%s:%s/docs\n" "$GREEN" "$RESET" "$HOST" "$BACKEND_PORT"
printf "  %sLogs        %s → %s\n" "$GREEN" "$RESET" "$LOG_DIR"
echo
info "Streaming combined logs below (Ctrl+C to stop)…"
echo

# Stream both logs until the user hits Ctrl+C. `tail -F` handles log rotation
# gracefully and survives if the files momentarily disappear.
tail -n 0 -F "$BACKEND_LOG" "$FRONTEND_LOG" &
TAIL_PID=$!

# Block the script until either child dies or we get SIGINT. `wait -n`
# returns when the first background job exits; re-check and bail out.
while :; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    err "Backend process died. Recent output:"
    tail -n 20 "$BACKEND_LOG" >&2 || true
    kill "$TAIL_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    err "Frontend process died. Recent output:"
    tail -n 20 "$FRONTEND_LOG" >&2 || true
    kill "$TAIL_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
