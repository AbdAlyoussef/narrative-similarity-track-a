import torch
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderScorer(nn.Module):
    """
    Cross-encoder with multi-head aspect scoring.
    - forward() returns ONLY the combined score (for unchanged eval/inference).
    - forward_aspects() returns (combined, theme, action, outcome) for auxiliary training.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # 3 aspect heads
        self.head_theme = nn.Linear(hidden_size, 1)
        self.head_action = nn.Linear(hidden_size, 1)
        self.head_outcome = nn.Linear(hidden_size, 1)

        # learnable weights, normalized with softmax
        self.aspect_logits = nn.Parameter(torch.zeros(3))  # [wT, wA, wO] before softmax

    def _encode_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.dropout(cls)

    def forward_aspects(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Returns:
          s_combined: [batch]
          s_theme:    [batch]
          s_action:   [batch]
          s_outcome:  [batch]
        """
        h = self._encode_cls(input_ids, attention_mask)
        s_theme = self.head_theme(h).squeeze(-1)
        s_action = self.head_action(h).squeeze(-1)
        s_outcome = self.head_outcome(h).squeeze(-1)

        w = torch.softmax(self.aspect_logits, dim=0)  # [3]
        s_combined = w[0] * s_theme + w[1] * s_action + w[2] * s_outcome
        return s_combined, s_theme, s_action, s_outcome

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Unchanged external behavior:
        returns only the final scalar score per pair.
        """
        s_combined, _, _, _ = self.forward_aspects(input_ids, attention_mask)
        return s_combined
