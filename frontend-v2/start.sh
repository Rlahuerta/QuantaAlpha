#!/bin/bash


echo "🚀 Starting QuantaAlpha AI V2..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# Check Node.js
# =============================================================================
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js not found"
    echo "Please install Node.js first: https://nodejs.org/"
    exit 1
fi
echo "✅ Node.js: $(node --version)"

# =============================================================================
# Activate conda env (same as main experiment: quantaalpha)
# =============================================================================
eval "$(conda shell.bash hook)" 2>/dev/null
CONDA_ENV="${CONDA_ENV_NAME:-quantaalpha}"
conda activate "${CONDA_ENV}" 2>/dev/null

if [ $? -ne 0 ]; then
    source activate "${CONDA_ENV}" 2>/dev/null
fi

if ! python -c "import quantaalpha" 2>/dev/null; then
    echo "❌ Error: quantaalpha package is not installed"
    echo "Run first: conda activate ${CONDA_ENV} && cd ${PROJECT_ROOT} && pip install -e ."
    exit 1
fi
echo "✅ Python: $(python --version) (conda env: ${CONDA_ENV})"

# =============================================================================
# Load .env configuration
# =============================================================================
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
    echo "✅ Loaded .env configuration"
else
    echo "⚠️  .env file not found, backend will use defaults"
fi

# =============================================================================
# Install frontend dependencies
# =============================================================================
cd "${SCRIPT_DIR}"
if [ ! -d "node_modules" ]; then
    echo ""
    echo "📦 Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install frontend dependencies"
        exit 1
    fi
    echo "✅ Frontend dependencies installed"
fi

# =============================================================================
# Install backend dependencies (inside conda env)
# =============================================================================
echo "📦 Checking/installing backend Python dependencies..."
pip install -q fastapi uvicorn websockets python-multipart python-dotenv pyyaml 2>/dev/null || true
echo "✅ Backend dependencies ready"

# =============================================================================
# Get local IP (for LAN access hint)
# =============================================================================
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP="localhost"
fi

# =============================================================================
# Detect and start backend (reuse existing service or restart)
# =============================================================================
BACKEND_PID=""
BACKEND_REUSED=false

echo ""
echo "🔍 Checking backend service (port 8000)..."
if curl -s --connect-timeout 2 http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend service already running (port 8000), reusing existing process"
    BACKEND_REUSED=true
    BACKEND_PID=$(lsof -ti:8000 2>/dev/null | head -1)
else
    # Clean up stale process occupying the port but not serving properly
    OLD_PID=$(lsof -ti:8000 2>/dev/null)
    if [ -n "$OLD_PID" ]; then
        echo "⚠️  Port 8000 is occupied but unhealthy, cleaning stale process (PID: $OLD_PID)..."
        kill $OLD_PID 2>/dev/null
        sleep 1
        kill -9 $OLD_PID 2>/dev/null 2>&1
    fi

    echo "🔧 Starting backend service (port 8000)..."
    cd "${SCRIPT_DIR}"
    python backend/app.py &
    BACKEND_PID=$!

    # Wait for backend startup
    sleep 3

    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend service started (PID: $BACKEND_PID)"
    else
        echo "❌ Backend startup failed, check logs"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
fi

# =============================================================================
# Detect and start frontend (reuse existing service or restart)
# =============================================================================
FRONTEND_PID=""
FRONTEND_REUSED=false

echo ""
echo "🔍 Checking frontend service (port 3000)..."
if curl -s --connect-timeout 2 http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend service already running (port 3000), reusing existing process"
    FRONTEND_REUSED=true
    FRONTEND_PID=$(lsof -ti:3000 2>/dev/null | head -1)
else
    # Clean up stale process occupying the port but not serving properly
    OLD_PID=$(lsof -ti:3000 2>/dev/null)
    if [ -n "$OLD_PID" ]; then
        echo "⚠️  Port 3000 is occupied but unhealthy, cleaning stale process (PID: $OLD_PID)..."
        kill $OLD_PID 2>/dev/null
        sleep 1
        kill -9 $OLD_PID 2>/dev/null 2>&1
    fi

    echo "🎨 Starting frontend service (port 3000)..."
    cd "${SCRIPT_DIR}"
    npm run dev &
    FRONTEND_PID=$!
    sleep 3
fi

echo ""
echo "============================================"
echo "✅ All services started!"
echo ""
echo "📍 Access URLs:"
echo "   Local:     http://localhost:3000"
if [ "$HOST_IP" != "localhost" ]; then
echo "   LAN:       http://${HOST_IP}:3000"
fi
echo "   Backend API: http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
if [ "$BACKEND_REUSED" = true ] || [ "$FRONTEND_REUSED" = true ]; then
echo "ℹ️  Some services reuse existing processes (multi-user shared mode)"
echo "   Ctrl+C only stops services started by this script; shared services keep running"
fi
echo ""
echo "Press Ctrl+C to stop services"
echo "============================================"
echo ""

# Handle exit signal — only stop processes started by this script
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    if [ "$BACKEND_REUSED" = false ] && [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "  Stopped backend (PID: $BACKEND_PID)"
    else
        echo "  Backend is shared; keeping it running"
    fi
    if [ "$FRONTEND_REUSED" = false ] && [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "  Stopped frontend (PID: $FRONTEND_PID)"
    else
        echo "  Frontend is shared; keeping it running"
    fi
    echo "✅ Done"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for child processes
wait
