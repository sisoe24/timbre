# Audio Analyzer — Phase 1

> A practical audio intelligence system for generating accurate, human-readable descriptions of audio clips — suitable for building searchable audio catalogs.

---

## What This Does

This system takes audio files as input and produces:

- A **concise one-sentence description** of what is heard
- A **detailed paragraph** covering temporal structure and acoustic character
- A **structured tag list** (controlled vocabulary)
- An **ordered list of sound events** (temporal breakdown)
- A **confidence score** for the overall description

Everything is designed for **cataloging accuracy** — no emotion analysis, no cinematic interpretation, no hallucinated context.

**Example output:**

```json
{
  "file_name": "metal_impact_01.wav",
  "short_description": "A sharp metallic impact followed by a brief echo.",
  "detailed_description": "The clip contains a metallic impact. The temporal sequence is: metallic impact → short echo tail. The sound has a sharp, transient attack.",
  "tags": ["impact", "metallic impact", "percussive", "sharp transient", "reverb"],
  "sound_events": ["metallic impact", "short echo tail"],
  "confidence": 0.84
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Audio Analyzer — Phase 1             │
│                                                         │
│   Audio File (.wav / .mp3 / .flac / .ogg / .m4a)       │
│         │                                               │
│         ▼                                               │
│   ┌──────────────┐                                      │
│   │ AudioLoader  │  librosa load + normalize to 48kHz   │
│   └──────┬───────┘                                      │
│          │                                              │
│          ├──────────────────────────┐                   │
│          ▼                          ▼                   │
│   ┌──────────────┐         ┌─────────────────┐         │
│   │FeatureExtract│         │   CLAPTagger    │         │
│   │   (librosa)  │         │(laion/larger_   │         │
│   │              │         │ clap_general)   │         │
│   │ - RMS energy │         │                 │         │
│   │ - Spectral   │         │ Full-clip       │         │
│   │ - Transients │         │ zero-shot       │         │
│   │ - Band energy│         │ classification  │         │
│   │ - Silence    │         │                 │         │
│   └──────┬───────┘         │ Sliding window  │         │
│          │                 │ event detection │         │
│          │                 └────────┬────────┘         │
│          │                          │                   │
│          └──────────┬───────────────┘                   │
│                     ▼                                   │
│             ┌──────────────┐                            │
│             │ Description  │  Template engine:          │
│             │ Synthesizer  │  tags + events + features  │
│             │              │  → natural language        │
│             └──────┬───────┘                            │
│                    │                                    │
│                    ▼                                    │
│             ┌──────────────┐                            │
│             │  Serializer  │  JSON / Markdown / CSV     │
│             └──────┬───────┘                            │
│                    ▼                                    │
│             AudioAnalysisRecord                         │
│             (Pydantic validated)                        │
└─────────────────────────────────────────────────────────┘
```

### Why CLAP + Template Engine (not an Audio LLM)?

| Property | CLAP + Templates | Audio LLM (e.g. Qwen-Audio) |
|---|---|---|
| Hallucination risk | **None** (labels are fixed) | Present |
| Consistency | **Deterministic** per run | Variable |
| Speed | **Fast** (< 2s/clip on GPU) | Slow (5–20s/clip) |
| GPU memory | **~4 GB** | 14–40 GB |
| Catalog vocabulary | **Controlled** | Free-form |
| Phase 1 suitability | **✓ Ideal** | Phase 2 enhancement |

---

## Project Structure

```
audio_analyzer/
├── timbre.py                   # Root CLI entrypoint
├── analyze.py                  # Compatibility wrapper for single-file CLI
├── batch_process.py            # Compatibility wrapper for batch CLI
├── pyproject.toml              # Poetry dependency metadata
├── requirements.txt
├── setup_mac.sh                # macOS Silicon setup (M1/M2/M3/M4)
├── setup_runpod.sh             # RunPod GPU environment setup
│
├── config/
│   ├── config.yaml             # Model, analysis, output settings
│   └── vocabulary.yaml         # Controlled vocabulary (13 categories, ~194 labels)
│
├── src/
│   ├── cli/
│   │   ├── main.py             # Top-level Click CLI with subcommands
│   │   ├── analyze.py          # Single-file analysis command
│   │   ├── batch.py            # Batch analysis command
│   │   └── cache.py            # Label-cache builder command
│   └── timbre/
│       ├── config_loader.py    # YAML config loader + logging setup
│       ├── pipeline.py         # Main orchestrator (AudioAnalysisPipeline)
│       ├── ingestion/
│       │   └── audio_loader.py # Load + validate + normalize audio files
│       ├── models/
│       │   └── clap_tagger.py  # CLAP zero-shot classification wrapper
│       ├── analysis/
│       │   ├── feature_extractor.py        # Acoustic features (librosa)
│       │   ├── event_detector.py           # Sliding-window event detection
│       │   └── description_synthesizer.py  # Natural language description builder
│       └── output/
│           ├── schema.py       # Pydantic AudioAnalysisRecord model
│           ├── serializer.py   # JSON / Markdown / CSV per-file output
│           └── catalog_builder.py  # Multi-file catalog aggregation
│
└── outputs/                    # Default output location
    ├── json/                   # Per-file JSON
    ├── markdown/               # Per-file Markdown review reports
    ├── catalog.md              # Full catalog grouped by category
    ├── catalog.csv             # Flat CSV catalog
    └── batch_results.json      # All records in one JSON array
```

---

## Running Locally — macOS Silicon (M1/M2/M3/M4)

### Requirements

- macOS 12.3 (Monterey) or later — required for MPS support
- Apple Silicon Mac (M1 or newer)
- [Homebrew](https://brew.sh)
- Python 3.10+

### Device behaviour

On Apple Silicon, PyTorch uses **MPS (Metal Performance Shaders)** — the GPU
backend for Apple's unified memory architecture. The system detects it
automatically in order of priority:

```
CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
```

fp16 is automatically disabled on MPS — CLAP runs in fp32, which is correct
and stable. Expect ~3–8x slower than a dedicated NVIDIA GPU but much faster
than CPU-only.

### Setup

```bash
cd audio_analyzer
bash setup_mac.sh
```

This creates a `.venv` in the project root, installs ffmpeg via Homebrew,
PyTorch with MPS support, all Python dependencies, and pre-downloads the
CLAP model (~1.2 GB).

### Optional: Poetry workflow

Poetry metadata is included via `pyproject.toml`. Poetry manages the Python
dependencies listed in `requirements.txt`, but **PyTorch is still intentionally
left out of Poetry** because installation is platform-specific:

- macOS: install torch separately before `poetry install`
- RunPod/CUDA: install torch, torchaudio, and torchvision together with the
  matching CUDA wheel index before `poetry install`

Example:

```bash
cd audio_analyzer
poetry install
poetry run timbre analyze samples/0_sample.wav
```

The Makefile also supports Poetry directly:

```bash
make install
make run USE_POETRY=1 FILE=samples/0_sample.wav
make batch USE_POETRY=1 DIR=./samples
```

Poetry console scripts are also defined:

```bash
poetry run timbre analyze samples/0_sample.wav
poetry run timbre batch ./samples
poetry run timbre cache --force
```

### Run

Activate the virtual environment, then run:

```bash
source .venv/bin/activate
python timbre.py analyze samples/0_sample.wav
python timbre.py batch ./samples/
```

Or skip activation and use the venv Python directly:

```bash
.venv/bin/python timbre.py analyze samples/0_sample.wav
```

To confirm MPS is active, look for this line in the output:

```
[INFO] timbre.models.clap_tagger: Loading CLAP model: laion/larger_clap_general on mps
```

---

## Running With Docker

The project now supports a simple **CPU-only Linux** Docker image for distribution.
The container exposes the existing `timbre` CLI directly, uses the bundled
`config/config.yaml` and `config/vocabulary.yaml` by default, and downloads the
CLAP model from Hugging Face on first run.

### Build the image

```bash
docker build -t timbre .
```

### Analyze one file

Mount input audio read-only and an output directory read-write:

```bash
docker run --rm \
  -v "$PWD/samples:/data/in:ro" \
  -v "$PWD/out:/data/out" \
  timbre analyze /data/in/example.wav --output-dir /data/out
```

### Batch analyze a directory

```bash
docker run --rm \
  -v "$PWD/samples:/data/in:ro" \
  -v "$PWD/out:/data/out" \
  timbre batch /data/in --output-dir /data/out
```

### Reuse the Hugging Face cache

To avoid downloading the CLAP model on every fresh container run, mount a
persistent cache directory:

```bash
docker run --rm \
  -v "$PWD/samples:/data/in:ro" \
  -v "$PWD/out:/data/out" \
  -v "$PWD/.hf-cache:/root/.cache/huggingface" \
  timbre analyze /data/in/example.wav --output-dir /data/out
```

### Use a custom config or vocabulary

Mount your custom files and pass them through the existing CLI options:

```bash
docker run --rm \
  -v "$PWD/samples:/data/in:ro" \
  -v "$PWD/out:/data/out" \
  -v "$PWD/config:/data/config:ro" \
  timbre analyze /data/in/example.wav \
    --output-dir /data/out \
    --config /data/config/config.yaml \
    --vocab /data/config/vocabulary.yaml
```

### Notes

- This first Docker workflow is CPU-only; no CUDA or GPU container support is included yet.
- The first run may take longer because model weights are downloaded at runtime.
- The recommended contract is to mount inputs read-only, outputs writable, and optionally persist `/root/.cache/huggingface`.

---

## Deploying to RunPod

### Requirements

- RunPod pod with at least one GPU (A10G, RTX 3090, A100, etc.)
- Recommended template: **RunPod PyTorch 2.4** (CUDA 12.4, Ubuntu 22.04)
- Your local machine needs: `ssh`, `scp` (both standard on macOS/Linux)

> **Note:** `torch >= 2.6.0` is required due to CVE-2025-32434. The setup
> script handles this upgrade automatically, including upgrading `torchvision`
> and `torchaudio` together to avoid version conflicts.

---

### Step 1 — Start a pod and get SSH details

In the RunPod UI, start a pod and click **Connect → SSH**. You'll get
connection details that look like:

```
ssh root@194.68.245.147 -p 22017 -i ~/.ssh/id_ed25519
```

---

### Step 2 — Upload the project with scp

From your local machine, in the directory **containing** `audio_analyzer/`:

```bash
scp -P 22017 -i ~/.ssh/id_ed25519 -r ./audio_analyzer root@194.68.245.147:~/
```

> The `-P` flag (uppercase) sets the port for scp — note this differs from
> ssh which uses lowercase `-p`.

To also upload audio samples:

```bash
scp -P 22017 -i ~/.ssh/id_ed25519 -r ./my_samples root@194.68.245.147:~/audio_analyzer/samples/
```

---

### Step 3 — SSH into the pod and run setup

```bash
ssh root@194.68.245.147 -p 22017 -i ~/.ssh/id_ed25519
cd ~/audio_analyzer
bash setup_runpod.sh
```

The setup script will:
1. Auto-detect your CUDA version and select the right PyTorch wheel
2. Create a `.venv` virtual environment in the project root
3. Install torch + torchaudio + torchvision together into the venv
4. Install ffmpeg and libsndfile1
5. Install all Python dependencies from `requirements.txt`
6. Pre-download and cache the CLAP model (~1.2 GB)
7. Verify everything works

---

### Step 4 — Run the analyzer

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

**Single file:**
```bash
python timbre.py analyze samples/0_sample.wav
```

**Single file with all outputs:**
```bash
python timbre.py analyze samples/0_sample.wav --output-dir ./outputs --markdown --full
```

**Batch — entire folder:**
```bash
python timbre.py batch ./samples/ --output-dir ./outputs
```

---

### Re-syncing after local edits

If you update the code locally and want to push changes to the pod, use scp
again. Because scp always overwrites, it's safe to re-run:

```bash
# Re-upload only the src/ folder (faster than uploading everything)
scp -P 22017 -i ~/.ssh/id_ed25519 -r ./audio_analyzer/src root@194.68.245.147:~/audio_analyzer/

# Or re-upload specific files
scp -P 22017 -i ~/.ssh/id_ed25519 \
  ./audio_analyzer/src/timbre/models/clap_tagger.py \
  root@194.68.245.147:~/audio_analyzer/src/timbre/models/
```

---

### Retrieving outputs from the pod

Copy the outputs folder back to your local machine:

```bash
scp -P 22017 -i ~/.ssh/id_ed25519 -r \
  root@194.68.245.147:~/audio_analyzer/outputs \
  ./outputs_from_pod
```

Or just the catalog files:

```bash
scp -P 22017 -i ~/.ssh/id_ed25519 \
  root@194.68.245.147:~/audio_analyzer/outputs/catalog.md \
  root@194.68.245.147:~/audio_analyzer/outputs/catalog.csv \
  root@194.68.245.147:~/audio_analyzer/outputs/batch_results.json \
  ./outputs_from_pod/
```

---

### Tip — save the pod's SSH config locally

Add an entry to `~/.ssh/config` so you don't have to type the full connection
string every time:

```
Host runpod-audio
    HostName 194.68.245.147
    Port 22017
    User root
    IdentityFile ~/.ssh/id_ed25519
```

Then you can use shorthand for everything:

```bash
ssh runpod-audio
scp -r ./audio_analyzer runpod-audio:~/
scp -r runpod-audio:~/audio_analyzer/outputs ./outputs_from_pod
```

---

## Usage

### Single file

```bash
python timbre.py analyze path/to/file.wav
```

Options:

| Flag | Description |
|---|---|
| `--output-dir` / `-o` | Directory to save output files |
| `--markdown` | Also save a per-file Markdown review report |
| `--full` | Save full JSON (includes metadata + acoustics) |
| `--no-windowed` | Disable sliding-window event detection (faster) |
| `--quiet` / `-q` | Suppress console output |

### Batch folder

```bash
python timbre.py batch ./samples/
```

Options:

| Flag | Description |
|---|---|
| `--output-dir` / `-o` | Root output directory |
| `--catalog` | Generate `catalog.md` (default: on) |
| `--csv` | Generate `catalog.csv` (default: on) |
| `--markdown` | Save per-file Markdown reports |
| `--full` | Full JSON output per file |
| `--limit N` | Only process first N files (useful for testing) |
| `--no-windowed` | Disable sliding-window event detection |

---

## Output Formats

### JSON (per file — spec format)

```json
{
  "file_name": "footsteps_gravel.wav",
  "short_description": "Footsteps on gravel with a consistent rhythmic pace.",
  "detailed_description": "The clip contains footsteps on gravel. Secondary sounds include outdoor ambience. The sound has noticeable transient elements.",
  "tags": ["movement", "footsteps on gravel", "outdoor ambience", "percussive", "rhythmic"],
  "sound_events": ["footsteps on gravel", "outdoor ambience"],
  "confidence": 0.78
}
```

### Markdown Catalog (excerpt)

```markdown
## Impact

### `metal_hit_01.wav`

**A sharp metallic impact followed by a brief reverberant tail.**

| | |
|---|---|
| Duration   | 1.23s          |
| Label      | metallic impact |
| Confidence | ████░ 0.84     |
| Events     | metallic impact → short echo tail |
| Tags       | `impact`, `metallic impact`, `percussive`, `sharp transient` |
```

### CSV Catalog

```
file_name,duration_seconds,primary_category,primary_label,confidence,short_description,...
metal_hit_01.wav,1.23,impact,metallic impact,0.84,A sharp metallic impact...,...
footsteps_01.wav,4.50,movement,footsteps on gravel,0.78,Footsteps on gravel...,...
```

---

## Configuration

### `config/config.yaml`

```yaml
model:
  model_id: "laion/larger_clap_general"
  device: null        # auto-detect: cuda → mps → cpu
  fp16: true          # float16 on GPU — reduces VRAM usage

analysis:
  use_windowed_analysis: true
  window_seconds: 2.0
  hop_seconds: 0.5
  min_confidence: 0.25

output:
  output_dir: "./outputs"
  save_per_file_markdown: true
  full_json: false
```

### `config/vocabulary.yaml`

Defines all labels CLAP classifies against. 13 categories, ~194 labels:

| Category | Example Labels |
|---|---|
| `impact` | metallic impact, glass shatter, gunshot, drum hit |
| `movement` | footsteps on gravel, door slam, paper rustling |
| `ambience` | outdoor ambience, crowd murmur, ocean waves |
| `weather` | heavy rain, thunder, wind howl |
| `machinery` | engine idle, electrical buzzing, drill |
| `vehicles` | car passing, motorcycle, aircraft flyover |
| `voices` | speech, laughter, crowd cheer |
| `water` | water dripping, waterfall, water splash |
| `animals` | bird chirping, dog bark, crickets |
| `textures` | low rumble, white noise, vinyl crackle |
| `music` | piano notes, guitar strum, rhythmic beat |
| `alerts` | alarm beep, siren, phone ringing |
| `background` | background noise, silence |

To add new labels: edit `vocabulary.yaml` and re-run. No retraining needed.

---

## Example Outputs

| File | Description | Tags | Conf |
|---|---|---|---|
| `metal_clang.wav` | A sharp metallic impact followed by a short echo. | impact, metallic impact, sharp transient, reverb | 0.87 |
| `rain_heavy.wav` | Heavy rain on a hard surface, continuous and broadband. | weather, heavy rain, broadband noise, continuous | 0.91 |
| `footsteps_wood.wav` | Fast footsteps on a wooden floor, ending with a door slam. | movement, footsteps on wood, door slam, rhythmic | 0.82 |
| `engine_idle.wav` | A low-frequency mechanical engine hum, steady and continuous. | machinery, engine idle, mechanical hum, low frequency | 0.88 |
| `forest_wind.wav` | Soft wind ambience with distant birds and gentle rustling. | ambience, wind ambience, bird chirping, continuous | 0.79 |

---

## Technical Notes

| | macOS Silicon (MPS) | RunPod (CUDA) | CPU fallback |
|---|---|---|---|
| **Setup** | `bash setup_mac.sh` | `bash setup_runpod.sh` | `pip install -r requirements.txt` |
| **Device** | MPS (auto-detected) | CUDA (auto-detected) | CPU |
| **fp16** | No (fp32 only) | Yes | No |
| **VRAM / RAM** | ~4 GB unified memory | ~4 GB VRAM | system RAM |
| **Speed (10s clip)** | ~5–15s | ~2–3s | ~30–60s |
| **torch wheels** | Standard pip | CUDA-specific index | Standard pip |

**Other notes:**
- CLAP model size: ~1.2 GB (downloaded from HuggingFace Hub on first run, then cached)
- `torch >= 2.6.0` required (CVE-2025-32434 — torch.load safety fix)
- On RunPod: `torchvision` must be upgraded together with `torch` in a single pip command to avoid internal import conflicts in transformers

---

## Phase 2 Roadmap

| Phase | Focus | Key Addition |
|---|---|---|
| Phase 1 | Cataloging | CLAP + templates (this system) |
| Phase 2 | Richer descriptions | Audio LLM (Qwen-Audio, SALMONN) |
| Phase 3 | Search | CLAP embeddings → vector database |
| Phase 4 | Similarity | Nearest-neighbor audio retrieval |
| Phase 5 | Streaming | Real-time pipeline (WebSocket) |
