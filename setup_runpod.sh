#!/bin/bash
# ============================================================================
# setup_runpod.sh
# ============================================================================
# Sets up the Audio Analyzer on a RunPod GPU instance.
# Run once after starting the pod:
#   bash setup_runpod.sh
#
# Tested on: RunPod PyTorch 2.4 template (CUDA 12.4, Ubuntu 22.04)
#
# Key notes:
#   - torch >= 2.6.0 is required (CVE-2025-32434 torch.load safety fix)
#   - torch, torchaudio, torchvision must be upgraded together or
#     torchvision will break transformers' image_utils import
#   - ffmpeg is required for MP3 support via librosa/audioread
# ============================================================================

set -e
echo "=========================================="
echo " Audio Analyzer — RunPod Setup"
echo "=========================================="

# --- Detect CUDA version ---------------------------------------------------
echo ""
echo "→ Detecting CUDA version..."
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "  Detected CUDA: ${CUDA_VERSION}"

# Map CUDA version to PyTorch wheel index
if [[ "$CUDA_VERSION" == 12.4* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "$CUDA_VERSION" == 12.1* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_VERSION" == 11.8* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    echo "  Unknown CUDA version, defaulting to cu124 wheel"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi
echo "  Using wheel index: ${TORCH_INDEX}"

# --- Upgrade PyTorch stack (torch + torchaudio + torchvision together) -----
# Must be done as a unit — mismatched versions break torchvision's
# _meta_registrations which transformers imports indirectly via image_utils.
echo ""
echo "→ Upgrading PyTorch stack to >= 2.6.0..."
pip install --force-reinstall --ignore-installed \
    "torch>=2.6.0" \
    "torchaudio>=2.6.0" \
    "torchvision>=0.21.0" \
    --index-url "${TORCH_INDEX}" \
    --quiet
echo "  ✓ PyTorch stack upgraded"

# --- Verify torch version --------------------------------------------------
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "  torch version: ${TORCH_VER}"
python -c "
import torch
from packaging import version
if version.parse(torch.__version__) < version.parse('2.6.0'):
    print('  ERROR: torch < 2.6.0 still active. Check pip install location.')
    exit(1)
else:
    print('  ✓ torch >= 2.6.0 confirmed')
" 2>/dev/null || python -c "
import torch
v = tuple(int(x) for x in torch.__version__.split('.')[:2])
if v < (2, 6):
    print('  WARNING: torch may be < 2.6.0, version:', torch.__version__)
else:
    print('  ✓ torch >= 2.6.0 confirmed:', torch.__version__)
"

# --- System dependencies ---------------------------------------------------
echo ""
echo "→ Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1
echo "  ✓ ffmpeg + libsndfile1 installed"

# --- Python dependencies ---------------------------------------------------
echo ""
echo "→ Installing Python packages..."
pip install -r requirements.txt --quiet
echo "  ✓ Python packages installed"

# --- Pre-download CLAP model -----------------------------------------------
echo ""
echo "→ Pre-downloading CLAP model from HuggingFace Hub..."
python -c "
from transformers import ClapModel, ClapProcessor
print('  Downloading laion/larger_clap_general...')
ClapProcessor.from_pretrained('laion/larger_clap_general')
ClapModel.from_pretrained('laion/larger_clap_general')
print('  ✓ Model cached.')
"

# --- Create output directories ---------------------------------------------
echo ""
echo "→ Creating output directories..."
mkdir -p outputs/json outputs/csv outputs/markdown
echo "  ✓ Output directories ready"

# --- Final verification ----------------------------------------------------
echo ""
echo "→ Verifying setup..."
python -c "
import torch, librosa, transformers, soundfile, rich
print(f'  torch      : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  librosa    : {librosa.__version__}')
print(f'  transformers: {transformers.__version__}')
print('  ✓ All imports successful')
"

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " Analyze a single file:"
echo "   python analyze.py samples/your_file.wav"
echo ""
echo " Batch analyze a folder:"
echo "   python batch_process.py ./samples/"
echo "=========================================="
