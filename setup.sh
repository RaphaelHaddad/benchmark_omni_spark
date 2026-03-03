#!/usr/bin/env bash
# Setup script for Qwen3-Omni benchmark framework

set -euo pipefail

echo "=========================================="
echo "Qwen3-Omni Benchmark Framework Setup"
echo "=========================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check if venv exists
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import sys
sys.path.insert(0, '.')
from benchmark_framework.config.settings import BenchmarkConfig
print('✓ Benchmark framework installed successfully')
"

# Check FFmpeg
echo ""
echo "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is installed"
    ffmpeg -version | head -n 1
else
    echo "⚠ FFmpeg not found. Please install FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run a benchmark:"
echo "  python benchmarks/run_benchmark.py --video ./videos/test.mp4"
echo ""
