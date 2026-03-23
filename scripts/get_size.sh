#!/usr/bin/env bash

# torch + torchaudio (biggest packages)
du -sh $(python -c "import torch, os; print(os.path.dirname(torch.__file__))") 2>/dev/null
du -sh $(python -c "import torchaudio, os; print(os.path.dirname(torchaudio.__file__))") 2>/dev/null

# transformers + all ML packages
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

# HuggingFace model cache (CLAP ~1.2 GB once downloaded)
du -sh ~/.cache/huggingface 2>/dev/null
