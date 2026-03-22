#!/bin/bash
# ============================================================================
# setup_runpod.sh
# ============================================================================
# Sets up the Audio Analyzer on a RunPod GPU instance.
# Run this script once after starting the pod:
#   bash setup_runpod.sh
#
# Tested on: RunPod PyTorch 2.1 template (CUDA 12.1, Ubuntu 22.04)
# ============================================================================

set -e
echo "=========================================="
echo " Audio Analyzer — RunPod Setup"
echo "=========================================="

# --- System dependencies (ffmpeg for MP3 support) --------------------------
echo ""
echo "→ Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 git

echo "  ✓ ffmpeg installed"

# --- Python dependencies ---------------------------------------------------
echo ""
echo "→ Installing Python packages..."

# Upgrade pip
pip install --upgrade pip -q

# Install requirements
pip install -r requirements.txt -q

echo "  ✓ Python packages installed"

# --- Pre-download CLAP model (optional but avoids first-run latency) -------
echo ""
echo "→ Pre-downloading CLAP model from HuggingFace Hub..."
python -c "
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

# --- Verify setup --------------------------------------------------------------
echo ""
echo "→ Verifying setup..."
python -c "
import torch, librosa, transformers, soundfile, rich
print(f'  torch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')
print(f'  librosa: {librosa.__version__}')
print(f'  transformers: {transformers.__version__}')
print('  ✓ All imports successful')
"

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " To analyze a single file:"
echo "   python analyze.py samples/your_file.wav"
echo ""
echo " To run batch analysis:"
echo "   python batch_process.py ./samples/"
echo "=========================================="
