import json
from typing import Dict, List
from torch.utils.data import Dataset

REQUIRED_KEYS = ["anchor_text", "text_a", "text_b", "text_a_is_closer"]

def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # synthetic files may contain extra fields (e.g., model_name); we ignore them.
            for k in REQUIRED_KEYS:
                if k not in obj or obj[k] is None:
                    print(f"Skipping line {line_idx} in {path}: missing or None key '{k}'")
                    break
            else:
                rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows

class TrackAPairwiseDataset(Dataset):
    """
    Returns (anchor, pos, neg) where pos is the narratively closer story to anchor.
    """
    def __init__(self, path: str):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        anchor = r["anchor_text"]
        a = r["text_a"]
        b = r["text_b"]
        a_is_closer = bool(r["text_a_is_closer"])
        pos, neg = (a, b) if a_is_closer else (b, a)
        return {"anchor": anchor, "pos": pos, "neg": neg}

class TrackATriplesDataset(Dataset):
    """
    Keeps the original triple format for evaluation/prediction.
    """
    def __init__(self, path: str):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        return {
            "anchor_text": r["anchor_text"],
            "text_a": r["text_a"],
            "text_b": r["text_b"],
            "text_a_is_closer": bool(r["text_a_is_closer"]),
        }
