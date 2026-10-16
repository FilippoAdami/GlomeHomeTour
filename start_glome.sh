#!/usr/bin/env bash
# start_glome.sh — Launch both Glome Backend API and Frontend Web App together

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "=========================================="
echo " Starting Glome Home Tour Local Services "
echo "=========================================="

# 1. Start Python FastAPI Backend
echo "-> Starting Backend API (port 8000)..."
cd "$DIR/backend"
if [ ! -d ".venv" ]; then
    echo "Error: backend/.venv not found."
    exit 1
fi

.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started (PID $BACKEND_PID)"

# 2. Start React Vite Frontend
echo "-> Starting Frontend Web App (port 5173)..."
cd "$DIR/web_app"
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!
echo "Frontend started (PID $FRONTEND_PID)"

echo "------------------------------------------"
echo "  Web App:       http://localhost:5173"
echo "  Backend API:   http://localhost:8000"
echo "  Interactive Docs: http://localhost:8000/docs"
echo "------------------------------------------"
echo "Press Ctrl+C to terminate both services."

cleanup() {
    echo ""
    echo "Stopping Glome services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

wait
