import torch
import torch.nn as nn
from transformers import AutoModel

class SingleHeadCrossEncoder(nn.Module):
    """
    Baseline single-head cross-encoder (no aspect heads).
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_size, 1)

    def _encode_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.dropout(cls)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self._encode_cls(input_ids, attention_mask)
        return self.head(h).squeeze(-1)
