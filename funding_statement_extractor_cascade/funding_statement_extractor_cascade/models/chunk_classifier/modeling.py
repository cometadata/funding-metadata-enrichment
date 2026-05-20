# Vendored from https://huggingface.co/cometadata/funding-chunk-classifier-modernbert-base
# Source: modeling.py at commit f69ad93c22f3b7e76544eb8fe15ba4af69cd1f2e on 2026-05-19
# If upstream changes the ChunkClassifier architecture, re-vendor and update this header.

"""Custom model class for funding-chunk-classifier-modernbert-base."""
import torch.nn as nn
from transformers import AutoModel


class ChunkClassifier(nn.Module):
    """ModernBERT-base encoder + mean-pool + binary head for funding-chunk detection."""

    def __init__(self, base: str = "answerdotai/ModernBERT-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over real (non-padding) tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)  # one logit per chunk
