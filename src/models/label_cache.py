"""
label_cache.py
--------------
Pre-computes and caches CLAP text embeddings for the full UCS vocabulary so
that they are encoded only once rather than on every audio file.

At inference time classification becomes:
  1. Encode audio → one forward pass
  2. Two-stage cosine similarity against cached matrix:
       Stage 1 — score audio against 82 category centroids → top-K categories
       Stage 2 — score audio against ~50-400 labels inside those categories
  vs. the old approach: re-encode all ~5 500 text labels on every call.

Cache file format  (PyTorch .pt dict)
--------------------------------------
  "labels"                  : List[str]           — N labels (flat)
  "embeddings"              : ndarray (N, D)       — L2-normalised text embeds
  "logit_scale"             : float                — model temperature
  "categories"              : List[str]            — M unique categories
  "category_centroids"      : ndarray (M, D)       — L2-normalised mean embeds
  "category_label_indices"  : Dict[str, List[int]] — category → label indices
  "label_to_category"       : Dict[str, str]
  "label_to_subcategory"    : Dict[str, str]
  "label_to_cat_id"         : Dict[str, str]
  "label_to_category_full"  : Dict[str, str]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# How many top-level UCS categories to explore in Stage 2.
# Higher = more accurate but slower (5 is a good balance).
DEFAULT_TOP_K_CATEGORIES = 5


class LabelEmbeddingCache:
    """
    Manages pre-computed CLAP text embeddings for the UCS vocabulary.

    Parameters
    ----------
    cache_path : path to the .pt cache file (read or written)
    """

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self._data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        tagger,  # CLAPTagger (already loaded)
        candidate_labels: List[str],
        label_to_category: Dict[str, str],
        label_to_subcategory: Dict[str, str],
        label_to_cat_id: Dict[str, str],
        label_to_category_full: Dict[str, str],
        batch_size: int = 64,
    ) -> None:
        """
        Encode all labels, compute category centroids, and save to disk.

        Parameters
        ----------
        tagger            : loaded CLAPTagger instance
        candidate_labels  : flat list of all text labels
        label_to_*        : lookup dicts from config_loader
        batch_size        : labels per text-encoder forward pass
        """
        n = len(candidate_labels)
        logger.info(
            "Building label embedding cache: %d labels → %s",
            n,
            self.cache_path,
        )

        # --- Encode all labels in batches --------------------------------
        embeddings = tagger.embed_text(candidate_labels, batch_size=batch_size)
        # embeddings: (N, D) float32, L2-normalised

        # --- Build per-category index ------------------------------------
        categories: List[str] = list(dict.fromkeys(
            label_to_category[lbl] for lbl in candidate_labels
            if lbl in label_to_category
        ))

        category_label_indices: Dict[str, List[int]] = {c: [] for c in categories}
        for i, lbl in enumerate(candidate_labels):
            cat = label_to_category.get(lbl)
            if cat and cat in category_label_indices:
                category_label_indices[cat].append(i)

        # --- Compute category centroids (mean of member embeddings) ------
        centroids = []
        for cat in categories:
            indices = category_label_indices[cat]
            centroid = embeddings[indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
        category_centroids = np.stack(centroids, axis=0)  # (M, D)

        logit_scale = tagger.logit_scale

        self._data = {
            "labels": candidate_labels,
            "embeddings": embeddings,
            "logit_scale": logit_scale,
            "categories": categories,
            "category_centroids": category_centroids,
            "category_label_indices": category_label_indices,
            "label_to_category": label_to_category,
            "label_to_subcategory": label_to_subcategory,
            "label_to_cat_id": label_to_cat_id,
            "label_to_category_full": label_to_category_full,
        }

        # --- Persist to disk ---------------------------------------------
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._data, self.cache_path)
        logger.info(
            "Cache saved: %d labels, %d categories, logit_scale=%.2f",
            n,
            len(categories),
            logit_scale,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load pre-computed embeddings from disk."""
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Label cache not found: {self.cache_path}\n"
                "Run  python scripts/build_label_cache.py  to build it."
            )
        logger.info("Loading label cache from %s", self.cache_path)
        self._data = torch.load(self.cache_path, weights_only=False)
        logger.info(
            "Cache loaded: %d labels, %d categories",
            len(self._data["labels"]),
            len(self._data["categories"]),
        )

    def is_valid(self, expected_label_count: Optional[int] = None) -> bool:
        """
        Return True if the cache file exists and (optionally) contains the
        expected number of labels.
        """
        if not self.cache_path.exists():
            return False
        if expected_label_count is not None:
            try:
                data = torch.load(self.cache_path, weights_only=False)
                return len(data.get("labels", [])) == expected_label_count
            except Exception:
                return False
        return True

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        audio_embedding: np.ndarray,
        top_k_categories: int = DEFAULT_TOP_K_CATEGORIES,
    ) -> Dict[str, float]:
        """
        Two-stage zero-shot classification using pre-computed embeddings.

        Stage 1: score audio_embedding against category centroids → top-K
        Stage 2: score audio_embedding against all labels in top-K categories

        Parameters
        ----------
        audio_embedding   : L2-normalised audio embedding, shape (D,)
        top_k_categories  : number of UCS categories to expand in stage 2

        Returns
        -------
        dict mapping label → softmax probability for all labels in top-K
        categories. Labels outside top-K categories are omitted (treat as 0).
        """
        if self._data is None:
            raise RuntimeError("Cache not loaded. Call load() or build() first.")

        data = self._data
        scale = data["logit_scale"]
        audio = audio_embedding  # already L2-normalised by embed_audio()

        # --- Stage 1: category-level scoring ----------------------------
        centroids = data["category_centroids"]          # (M, D)
        cat_logits = scale * (centroids @ audio)        # (M,)
        top_cat_idx = np.argsort(cat_logits)[::-1][:top_k_categories]
        top_categories = [data["categories"][i] for i in top_cat_idx]

        # --- Stage 2: label-level scoring within top-K categories -------
        gathered_indices: List[int] = []
        for cat in top_categories:
            gathered_indices.extend(data["category_label_indices"].get(cat, []))

        if not gathered_indices:
            return {}

        sub_embeds = data["embeddings"][gathered_indices]   # (K, D)
        sub_labels = [data["labels"][i] for i in gathered_indices]

        logits = scale * (sub_embeds @ audio)               # (K,)
        # Numerically stable softmax
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        return {lbl: float(p) for lbl, p in zip(sub_labels, probs)}

    # ------------------------------------------------------------------
    # Lookup helpers (forwarded from cached metadata)
    # ------------------------------------------------------------------

    @property
    def label_to_category(self) -> Dict[str, str]:
        return self._data["label_to_category"]

    @property
    def label_to_subcategory(self) -> Dict[str, str]:
        return self._data["label_to_subcategory"]

    @property
    def label_to_cat_id(self) -> Dict[str, str]:
        return self._data["label_to_cat_id"]

    @property
    def label_to_category_full(self) -> Dict[str, str]:
        return self._data["label_to_category_full"]

    @property
    def candidate_labels(self) -> List[str]:
        return self._data["labels"]
