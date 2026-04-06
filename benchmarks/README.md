# Benchmark Fixture Scaffold

This directory is the landing zone for the evidence-first pipeline benchmark set.

Expected contents:

- `manifest.json` with 30-50 clips
- `audio/` containing the benchmark clips
- human-authored gold UCS metadata per clip
- reference descriptions per clip

Suggested manifest shape:

```json
[
  {
    "file_name": "metal_hit_01.wav",
    "audio_path": "audio/metal_hit_01.wav",
    "gold": {
      "category": "IMPACTS",
      "subcategory": "METAL",
      "cat_id": "IMPMtl",
      "category_full": "IMPACTS-METAL",
      "fx_name": "Metal Hit",
      "keywords": ["metal", "impact"],
      "sound_events": ["metal impact"],
      "description": "A short sharp metallic impact with a brief ring."
    }
  }
]
```

The benchmark harness is intentionally not coupled to a specific provider. Use
this manifest to compare the current pipeline output against the gold metadata
and reference descriptions before broad rollout changes.
