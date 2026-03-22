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

**Example outputs:**

```json
{
  "file_name": "metal_impact_01.wav",
  "short_description": "A sharp metallic impact with a brief reverberant decay.",
  "detailed_description": "The clip contains a metallic impact. The sound has a sharp, transient attack; strong harmonic content suggests a tonal source; energy is concentrated in the low-frequency range.",
  "tags": ["impact", "metallic impact", "percussive", "sharp transient", "reverb", "low frequency"],
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
│   │ AudioLoader  │ librosa load + normalize to 48kHz    │
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
│             │ Description  │ Template engine:           │
│             │ Synthesizer  │ tags + events + features   │
│             │              │ → natural language          │
│             └──────┬───────┘                            │
│                    │                                    │
│                    ▼                                    │
│             ┌──────────────┐                            │
│             │  Serializer  │ JSON / Markdown / CSV      │
│             └──────┬───────┘                            │
│                    │                                    │
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
| Temporal analysis | Via sliding window | Native |
| Phase 1 suitability | **✓ Ideal** | Phase 2 enhancement |

CLAP zero-shot classification never invents labels — it scores audio against a fixed controlled vocabulary. The template engine builds descriptions from confirmed detections only.

---

## Project Structure

```
audio_analyzer/
├── analyze.py                 # CLI: single file analysis
├── batch_process.py           # CLI: batch folder analysis
├── requirements.txt
├── setup_runpod.sh            # RunPod GPU environment setup
│
├── config/
│   ├── config.yaml            # Model, analysis, output settings
│   └── vocabulary.yaml        # Controlled vocabulary (14 categories, ~200 labels)
│
├── src/
│   ├── config_loader.py       # YAML config loader + logging setup
│   ├── pipeline.py            # Main orchestrator (AudioAnalysisPipeline)
│   │
│   ├── ingestion/
│   │   └── audio_loader.py    # Load + validate + normalize audio files
│   │
│   ├── models/
│   │   └── clap_tagger.py     # CLAP zero-shot classification wrapper
│   │
│   ├── analysis/
│   │   ├── feature_extractor.py       # Acoustic feature extraction (librosa)
│   │   ├── event_detector.py          # Sliding-window temporal event detection
│   │   └── description_synthesizer.py # Natural language description builder
│   │
│   └── output/
│       ├── schema.py          # Pydantic AudioAnalysisRecord model
│       ├── serializer.py      # JSON / Markdown / CSV per-file output
│       └── catalog_builder.py # Multi-file catalog aggregation
│
└── outputs/                   # Default output location (git-ignored)
    ├── json/                  # Per-file JSON
    ├── markdown/              # Per-file Markdown review reports
    ├── catalog.md             # Full catalog (all files, grouped by category)
    ├── catalog.csv            # Flat CSV catalog
    └── batch_results.json     # All records in one JSON array
```

---

## Getting Started on RunPod

### 1. Start a Pod

Recommended template: **RunPod PyTorch 2.1** with at least one GPU (A100 40GB or similar).
The system runs on smaller GPUs too (RTX 3090, A10G) with `fp16: true`.

### 2. Clone and set up

```bash
git clone <your-repo-url> audio_analyzer
cd audio_analyzer
bash setup_runpod.sh
```

This installs dependencies, ffmpeg, and pre-downloads the CLAP model (~1.2 GB).

### 3. Analyze a single file

```bash
python analyze.py samples/my_sound.wav
```

With options:

```bash
python analyze.py samples/my_sound.wav \
  --output-dir ./outputs \
  --markdown \
  --full
```

### 4. Batch analyze a folder

```bash
python batch_process.py ./samples/ \
  --output-dir ./outputs \
  --catalog \
  --csv
```

---

## Configuration

### `config/config.yaml`

```yaml
model:
  model_id: "laion/larger_clap_general"
  device: null          # auto-detect (cuda / cpu)
  fp16: true            # float16 on GPU

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

Defines the **controlled vocabulary** — all labels CLAP classifies against. Categories:

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

To add new labels: edit `vocabulary.yaml` and re-run. **No retraining needed.**

---

## Output Formats

### JSON (per file, spec format)

```json
{
  "file_name": "footsteps_gravel.wav",
  "short_description": "Footsteps on gravel with a consistent rhythmic pace.",
  "detailed_description": "The clip contains footsteps on gravel. Secondary sounds include outdoor ambience. The sound has noticeable transient elements; high spectral flatness indicates a broadband or noise-like source.",
  "tags": ["movement", "footsteps on gravel", "outdoor ambience", "percussive", "rhythmic"],
  "sound_events": ["footsteps on gravel", "outdoor ambience"],
  "confidence": 0.78
}
```

### Markdown Catalog (excerpt)

```markdown
# Audio Catalog

Generated: 2026-03-21 14:30 UTC  |  Total files: 42

---

## Impact

### `metal_hit_01.wav`

**A sharp metallic impact followed by a brief reverberant tail.**

The clip contains a metallic impact. The temporal sequence is:
metallic impact → short echo tail. The sound has a sharp, transient attack.

| | |
|---|---|
| Duration | 1.23s |
| Label | metallic impact |
| Confidence | ████░ 0.84 |
| Events | metallic impact → short echo tail |
| Tags | `impact`, `metallic impact`, `percussive`, `sharp transient` |
```

### CSV Catalog

```csv
file_name,duration_seconds,primary_category,primary_label,confidence,short_description,tags,...
metal_hit_01.wav,1.23,impact,metallic impact,0.84,"A sharp metallic impact...","impact; metallic impact; percussive",...
footsteps_01.wav,4.50,movement,footsteps on gravel,0.78,"Footsteps on gravel...",...
```

---

## Example Outputs

### Impact sounds

| File | Description | Tags | Conf |
|---|---|---|---|
| `metal_clang.wav` | A sharp metallic impact followed by a short reverberant echo. | impact, metallic impact, sharp transient, reverb | 0.87 |
| `glass_shatter.wav` | Glass shattering with bright, dispersed high-frequency content. | impact, glass shatter, high frequency content, transient | 0.81 |
| `wooden_knock.wav` | A hollow wooden knock with low mid-frequency resonance. | impact, wooden impact, low mid, tonal | 0.74 |

### Ambience and environment

| File | Description | Tags | Conf |
|---|---|---|---|
| `rain_heavy.wav` | Heavy rain on a hard surface, continuous and broadband. | weather, heavy rain, broadband noise, continuous | 0.91 |
| `forest_wind.wav` | Soft wind ambience with distant birds and gentle rustling. | ambience, wind ambience, bird chirping, continuous | 0.79 |
| `ocean_waves.wav` | Rhythmic ocean waves with low rumble and white noise texture. | ambience, ocean waves, low rumble, rhythmic | 0.83 |

### Machinery and mechanical

| File | Description | Tags | Conf |
|---|---|---|---|
| `engine_idle.wav` | A low-frequency mechanical engine hum, steady and continuous. | machinery, engine idle, mechanical hum, low frequency, continuous | 0.88 |
| `drill_burst.wav` | An electric drill with rhythmic bursts and high-frequency noise. | machinery, drill, electrical buzzing, rhythmic, intermittent | 0.76 |

### Movement

| File | Description | Tags | Conf |
|---|---|---|---|
| `footsteps_wood.wav` | Fast footsteps on a wooden floor, ending with a door slam. | movement, footsteps on wood, door slam, percussive, rhythmic | 0.82 |
| `door_creak.wav` | A slow door creak followed by a soft closing click. | movement, door creak, door closing, tonal | 0.71 |

---

## Controlled Vocabulary Guidelines

When adding or editing vocabulary labels, follow these rules:

**Labels must:**
- Describe what is audible, not what caused it
- Use a noun phrase (thing + qualifier): "metallic impact", "heavy rain", "engine idle"
- Be specific enough to be useful for search
- Avoid emotional or cinematic language

**Good labels:** `metal scrape`, `glass shatter`, `deep bass rumble`, `electrical buzzing`, `crowd murmur`

**Bad labels:** `scary sound`, `cinematic tension`, `maybe metal`, `interesting noise`

---

## Phase 2 Roadmap

Phase 1 focuses on accurate, consistent cataloging. Future phases can extend the system:

| Phase | Focus | Key Addition |
|---|---|---|
| Phase 1 | Cataloging | CLAP + templates (this system) |
| Phase 2 | Richer descriptions | Audio LLM (Qwen-Audio, SALMONN) |
| Phase 3 | Search | CLAP embeddings → vector database |
| Phase 4 | Similarity | Nearest-neighbor audio retrieval |
| Phase 5 | Streaming | Real-time pipeline (WebSocket) |

---

## Technical Notes

**CLAP model size:** ~1.2 GB (downloaded from HuggingFace Hub on first run, cached locally).

**GPU memory usage:** ~4 GB VRAM in fp16 mode. Runs on any modern GPU (A10G, RTX 3090, A100).

**Processing speed (A100 40GB):** ~0.5s per clip (full-clip), ~2-3s per clip (with windowed analysis on a 10s file).

**CPU fallback:** The system runs on CPU if no GPU is available. Expect ~5-10x slower processing.

---

## License

Phase 1 — Internal Research Preview
