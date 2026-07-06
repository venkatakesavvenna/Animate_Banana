"""Visual similarity via DINOv2 embeddings: cosine similarity between the
ground-truth diagram image and the PNG rendered from a model's generated
TikZ. Higher = more visually similar.

Loaded once per benchmark run (not per-sample) since the model load is the
expensive part; `DinoScorer.score` is cheap per call.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_DINO_REPO = "facebook/dinov2-base"


class DinoScorer:
    def __init__(self, repo_id: str = DEFAULT_DINO_REPO, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModel.from_pretrained(repo_id).to(self.device).eval()

    @torch.no_grad()
    def _embed(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # CLS token of the last hidden state is the standard DINO image embedding.
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    @torch.no_grad()
    def score(self, gt_image_path: Path, rendered_image_path: Path) -> float:
        gt_emb = self._embed(gt_image_path)
        rendered_emb = self._embed(rendered_image_path)
        sim = F.cosine_similarity(gt_emb.unsqueeze(0), rendered_emb.unsqueeze(0))
        return sim.item()
