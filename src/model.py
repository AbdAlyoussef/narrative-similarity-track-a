import torch
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderScorer(nn.Module):
    """
    Cross-encoder scoring model:
    - Input: (anchor_text, candidate_text) tokenized together
    - Output: a single scalar score s(anchor, candidate)
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [batch, hidden]
        cls = self.dropout(cls)
        score = self.head(cls).squeeze(-1)  # [batch]
        return score
