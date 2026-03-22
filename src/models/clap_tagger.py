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
    device    : "cuda", "mps", "cpu", or None (auto-detect)
    fp16      : use float16 for lower memory usage (only applied on CUDA;
                disabled automatically on MPS and CPU)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device or _resolve_device()
        # fp16 only on CUDA — MPS has incomplete float16 kernel support
        # and CPU has no benefit from fp16
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
        Return the L2-normalised CLAP audio embedding for a waveform.

        If the waveform is longer than CLAP_MAX_SECONDS it is split into
        chunks and the mean embedding is returned.

        Shape: (512,) float32.
        """
        self._ensure_loaded()
        chunks = _split_waveform(waveform, sr, max_seconds=CLAP_MAX_SECONDS)
        embeddings = []
        for chunk in chunks:
            inputs = self._processor(
                audio=chunk,
                return_tensors="pt",
                sampling_rate=sr,
            ).to(self.device)
            if self.fp16:
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }
            with torch.no_grad():
                audio_out = self._model.audio_model(**inputs)
                feat = self._model.audio_projection(
                    audio_out.pooler_output
                )
            embeddings.append(feat.cpu().float().numpy()[0])

        mean_embed = np.mean(embeddings, axis=0)
        # L2-normalise so cosine similarity == dot product
        norm = np.linalg.norm(mean_embed)
        if norm > 0:
            mean_embed = mean_embed / norm
        return mean_embed  # shape: (512,)

    def embed_text(
        self,
        labels: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a list of text labels into L2-normalised CLAP embeddings.

        Parameters
        ----------
        labels     : list of text descriptions
        batch_size : number of labels per forward pass (reduce if OOM)

        Returns
        -------
        np.ndarray of shape (N, D) float32, each row L2-normalised.
        """
        self._ensure_loaded()
        all_embeddings = []

        for i in range(0, len(labels), batch_size):
            batch = labels[i : i + batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                # Use text_model + text_projection directly to avoid
                # version-dependent behaviour of get_text_features()
                text_out = self._model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                # pooler_output is the [CLS] representation
                feat = self._model.text_projection(text_out.pooler_output)
            # L2-normalise each row
            norms = feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            feat = feat / norms
            all_embeddings.append(feat.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)  # (N, D)

    @property
    def logit_scale(self) -> float:
        """Temperature scaling factor used by CLAP (exp of learned log-scale)."""
        self._ensure_loaded()
        # ClapModel uses logit_scale_a (audio) and logit_scale_t (text).
        # For audio-text similarity we use the audio scale.
        scale_attr = "logit_scale_a" if hasattr(self._model, "logit_scale_a") else "logit_scale"
        return float(getattr(self._model, scale_attr).exp().item())

    def score_audio_vs_embeddings(
        self,
        audio_embedding: np.ndarray,
        text_embeddings: np.ndarray,
        labels: List[str],
    ) -> Dict[str, float]:
        """
        Score a pre-computed audio embedding against pre-computed text embeddings.

        Both embeddings must be L2-normalised (as returned by embed_audio /
        embed_text).  Uses the model's logit scale to match the behaviour of
        the original classify() method.

        Parameters
        ----------
        audio_embedding : shape (D,)
        text_embeddings : shape (N, D)
        labels          : list of N label strings

        Returns
        -------
        dict mapping each label to its softmax probability.
        """
        scale = self.logit_scale
        logits = scale * (text_embeddings @ audio_embedding)  # (N,)
        # Numerically stable softmax
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return {label: float(p) for label, p in zip(labels, probs)}

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
            audio=waveform,
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

def _resolve_device() -> str:
    """
    Auto-detect the best available compute device.

    Priority: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU

    Notes
    -----
    - CUDA: full fp16 support, fastest
    - MPS:  Apple Metal (M1/M2/M3/M4), fp32 only for CLAP compatibility
    - CPU:  fallback, slowest (~5–10x vs GPU)
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


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
