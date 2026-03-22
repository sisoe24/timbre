"""
clap_tagger.py
--------------
Zero-shot audio classification via CLAP (Contrastive Language-Audio Pretraining).

Model: laion/larger_clap_general
  — Trained on 633K audio-text pairs (AudioCaps, FreeSound, etc.)
  — Produces a shared embedding space for audio and text
  — Allows zero-shot classification: score any audio against any text label

Usage pattern
-------------
1. Full-clip classification  → overall category + tag scores
2. Sliding-window inference  → temporal event timeline

Reference
---------
  LAION CLAP: https://huggingface.co/laion/larger_clap_general
  Paper: https://arxiv.org/abs/2211.06687
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default model — can be overridden via config
DEFAULT_MODEL_ID = "laion/larger_clap_general"

# Fallback (smaller, faster) model if the larger one is unavailable
FALLBACK_MODEL_ID = "laion/clap-htsat-unfused"

# CLAP was trained at 48 kHz — always resample to this before passing audio
CLAP_SAMPLE_RATE = 48_000

# Maximum audio length CLAP processes in one forward pass (in seconds)
# Longer clips are chunked automatically
CLAP_MAX_SECONDS = 10.0


class CLAPTagger:
    """
    Wraps a CLAP model for zero-shot audio classification.

    Parameters
    ----------
    model_id  : HuggingFace model identifier
    device    : "cuda", "cpu", or None (auto-detect)
    fp16      : use float16 on GPU for lower memory usage (default True on CUDA)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and (self.device == "cuda")

        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights from HuggingFace Hub (or local cache)."""
        from transformers import ClapModel, ClapProcessor

        logger.info("Loading CLAP model: %s on %s", self.model_id, self.device)

        try:
            self._processor = ClapProcessor.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id)
        except Exception as exc:
            logger.warning(
                "Failed to load '%s' (%s). Falling back to '%s'.",
                self.model_id,
                exc,
                FALLBACK_MODEL_ID,
            )
            self._processor = ClapProcessor.from_pretrained(FALLBACK_MODEL_ID)
            self._model = ClapModel.from_pretrained(FALLBACK_MODEL_ID)

        if self.fp16:
            self._model = self._model.half()

        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("CLAP model loaded.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Core inference methods
    # ------------------------------------------------------------------

    def classify(
        self,
        waveform: np.ndarray,
        sr: int,
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        """
        Zero-shot classification of an audio clip against a list of text labels.

        Returns a dict mapping each label to its probability score (sum to 1.0).

        If the waveform is longer than CLAP_MAX_SECONDS, it is split into
        non-overlapping chunks and scores are averaged across chunks.

        Parameters
        ----------
        waveform         : mono float32 waveform, shape (num_samples,)
        sr               : sample rate (must be CLAP_SAMPLE_RATE = 48000 Hz)
        candidate_labels : list of text descriptions to score against
        """
        self._ensure_loaded()
        assert sr == CLAP_SAMPLE_RATE, (
            f"CLAP requires 48 kHz audio, got {sr} Hz. "
            "Pass audio through load_audio(target_sr=48000) first."
        )

        chunks = _split_waveform(waveform, sr, max_seconds=CLAP_MAX_SECONDS)
        chunk_scores: List[Dict[str, float]] = []

        for chunk in chunks:
            scores = self._classify_chunk(chunk, sr, candidate_labels)
            chunk_scores.append(scores)

        # Average probability across chunks
        averaged = {
            label: float(np.mean([s[label] for s in chunk_scores]))
            for label in candidate_labels
        }
        # Re-normalize so probabilities sum to 1
        total = sum(averaged.values())
        if total > 0:
            averaged = {k: v / total for k, v in averaged.items()}

        return averaged

    def classify_windowed(
        self,
        waveform: np.ndarray,
        sr: int,
        candidate_labels: List[str],
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
    ) -> List[Tuple[float, float, Dict[str, float]]]:
        """
        Sliding-window classification to capture temporal event changes.

        Returns a list of (start_time, end_time, score_dict) tuples,
        one per window.

        Parameters
        ----------
        waveform         : mono float32 waveform
        sr               : sample rate (48000)
        candidate_labels : labels to score
        window_seconds   : window size in seconds
        hop_seconds      : hop between windows in seconds
        """
        self._ensure_loaded()

        window_samples = int(window_seconds * sr)
        hop_samples = int(hop_seconds * sr)
        total_samples = len(waveform)

        results: List[Tuple[float, float, Dict[str, float]]] = []

        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            chunk = waveform[start:end]

            # Pad short final window
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))

            scores = self._classify_chunk(chunk, sr, candidate_labels)
            t_start = start / sr
            t_end = end / sr
            results.append((t_start, t_end, scores))

            if end >= total_samples:
                break
            start += hop_samples

        return results

    def embed_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Return the CLAP audio embedding for a waveform.
        Useful for similarity search in future phases.
        """
        self._ensure_loaded()
        inputs = self._processor(
            audios=waveform,
            return_tensors="pt",
            sampling_rate=sr,
        ).to(self.device)

        if self.fp16:
            inputs = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            audio_features = self._model.get_audio_features(**inputs)

        return audio_features.cpu().float().numpy()[0]  # shape: (512,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_chunk(
        self,
        waveform: np.ndarray,
        sr: int,
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        """Run CLAP inference on a single chunk ≤ CLAP_MAX_SECONDS."""
        inputs = self._processor(
            audios=waveform,
            text=candidate_labels,
            return_tensors="pt",
            padding=True,
            sampling_rate=sr,
        ).to(self.device)

        if self.fp16:
            inputs = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_audio shape: (1, num_labels)
            logits = outputs.logits_per_audio[0]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        return {label: float(prob) for label, prob in zip(candidate_labels, probs)}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _split_waveform(
    waveform: np.ndarray,
    sr: int,
    max_seconds: float,
) -> List[np.ndarray]:
    """Split a waveform into non-overlapping chunks of at most max_seconds."""
    max_samples = int(max_seconds * sr)
    if len(waveform) <= max_samples:
        return [waveform]

    chunks = []
    start = 0
    while start < len(waveform):
        end = min(start + max_samples, len(waveform))
        chunks.append(waveform[start:end])
        start = end

    return chunks
