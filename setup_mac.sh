#!/bin/bash
# ============================================================================
# setup_mac.sh
# ============================================================================
# Sets up the Audio Analyzer on macOS with Apple Silicon (M1/M2/M3/M4).
# Run once from the project root:
#   bash setup_mac.sh
#
# Creates a .venv virtual environment in the project root.
# Activate before running the analyzer:
#   source .venv/bin/activate
#
# Requirements:
#   - macOS 12.3+ (Monterey or later, required for MPS support)
#   - Homebrew (https://brew.sh)
#   - Python 3.10+ (via Homebrew or pyenv recommended)
# ============================================================================

set -e
echo "=========================================="
echo " Audio Analyzer — macOS Silicon Setup"
echo "=========================================="

# --- Check for Homebrew ----------------------------------------------------
if ! command -v brew &>/dev/null; then
    echo "✗ Homebrew not found. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
echo "✓ Homebrew found"

# --- Check for Python -------------------------------------------------------
# Use the Python that is currently active in the shell (respects pyenv, venv, etc.)
# Prefer `python` over `python3` so pyenv shims resolve correctly.
PYTHON=$(command -v python || command -v python3)
if [ -z "$PYTHON" ]; then
    echo "✗ Python not found. Install with: brew install python"
    exit 1
fi
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "✓ $PYTHON_VERSION ($(which $PYTHON))"

# --- Create virtual environment --------------------------------------------
echo ""
echo "→ Setting up virtual environment..."
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "  .venv already exists — reusing it"
else
    $PYTHON -m venv "$VENV_DIR"
    echo "  ✓ Created .venv"
fi

# All subsequent commands use the venv
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
echo "  Active: $($PYTHON --version) at $PYTHON"

# --- System dependencies (ffmpeg for MP3 support) --------------------------
echo ""
echo "→ Installing system packages via Homebrew..."
brew install ffmpeg libsndfile 2>/dev/null || brew upgrade ffmpeg libsndfile 2>/dev/null || true
echo "  ✓ ffmpeg + libsndfile installed"

# --- PyTorch (standard wheels — no CUDA index needed on macOS) -------------
echo ""
echo "→ Installing PyTorch with MPS support..."
# Standard pip install gets the macOS/ARM wheels automatically.
# MPS support is included in PyTorch >= 1.12 for Apple Silicon.
# torch >= 2.6.0 is required for CVE-2025-32434 (torch.load safety fix).
$PIP install --upgrade "torch>=2.6.0" torchaudio
echo "  ✓ PyTorch installed"

# --- Verify MPS availability -----------------------------------------------
echo ""
echo "→ Checking MPS (Metal Performance Shaders) availability..."
$PYTHON -c "
import torch
cuda = torch.cuda.is_available()
mps  = torch.backends.mps.is_available() and torch.backends.mps.is_built()
print(f'  torch version : {torch.__version__}')
print(f'  CUDA available: {cuda}')
print(f'  MPS available : {mps}')
if mps:
    print('  ✓ MPS ready — model will run on Apple GPU')
else:
    print('  ⚠ MPS not available — will fall back to CPU')
    print('    Requires macOS 12.3+ and Apple Silicon (M1/M2/M3/M4)')
"

# --- Python dependencies ---------------------------------------------------
echo ""
echo "→ Installing Python packages..."
$PIP install -r requirements.txt
echo "  ✓ Python packages installed"

# --- Pre-download CLAP model -----------------------------------------------
echo ""
echo "→ Pre-downloading CLAP model from HuggingFace Hub..."
echo "  (This is ~1.2 GB and may take a few minutes)"
$PYTHON -c "
from transformers import ClapModel, ClapProcessor
print('  Downloading laion/larger_clap_general...')
ClapProcessor.from_pretrained('laion/larger_clap_general')
ClapModel.from_pretrained('laion/larger_clap_general')
print('  ✓ Model cached.')
"

# --- Create output directories ------------------------------------------------
echo ""
echo "→ Creating output directories..."
mkdir -p outputs/json outputs/csv outputs/markdown
echo "  ✓ Output directories ready"

# --- Final verification --------------------------------------------------------
echo ""
echo "→ Verifying setup..."
$PYTHON -c "
import torch, librosa, transformers, soundfile, rich
print(f'  torch        : {torch.__version__}')
print(f'  librosa      : {librosa.__version__}')
print(f'  transformers : {transformers.__version__}')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  active device: {device}')
print('  ✓ All imports successful')
"

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " Activate the virtual environment first:"
echo "   source .venv/bin/activate"
echo ""
echo " Then run the analyzer:"
echo "   python timbre.py analyze samples/0_sample.wav"
echo "   python timbre.py batch ./samples/"
echo ""
echo " Or run without activating:"
echo "   .venv/bin/python timbre.py analyze samples/0_sample.wav"
echo "=========================================="
