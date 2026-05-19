# Vendored from https://huggingface.co/cometadata/funding-extraction-modernbert-base-spanhead
# Source: modeling.py at commit 1e00313fe334fbe8ef93f48e17db3a7be6456789 on 2026-05-19
# If upstream changes the SpanHead architecture, re-vendor and update this header.

"""Custom model class for funding-extraction-modernbert-base-spanhead."""
import torch
import torch.nn as nn
from transformers import AutoModel


class SpanHead(nn.Module):
    """ModernBERT-base encoder + start/end/no-answer heads for funding span extraction."""

    def __init__(self, base: str = "answerdotai/ModernBERT-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size  # 768
        self.start_head = nn.Linear(h, 1)
        self.end_head = nn.Linear(h, 1)
        self.no_answer_head = nn.Linear(h, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)
        start_logits = self.start_head(hidden).squeeze(-1)
        end_logits = self.end_head(hidden).squeeze(-1)
        # Mean-pool for no-answer
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        no_answer = self.no_answer_head(pooled).squeeze(-1)
        return start_logits, end_logits, no_answer
