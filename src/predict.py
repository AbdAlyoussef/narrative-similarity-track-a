import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer
from tqdm import tqdm

from src.config import TrainConfig
from src.data import TrackATriplesDataset
from src.model import CrossEncoderScorer
from src.utils import ensure_dir

def collate_triples(batch):
    keys = batch[0].keys()
    return {k: [x[k] for x in batch] for k in keys}

@torch.no_grad()
def score_batch(model, tokenizer, anchors, cands, device, max_length: int):
    tok = tokenizer(
        anchors, cands,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt"
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    return model(tok["input_ids"], tok["attention_mask"]).detach().cpu().tolist()

def run(cfg: TrainConfig, input_path: str, ckpt_path: str, output_path: str):
    ensure_dir("output")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "deberta" in cfg.model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    model = CrossEncoderScorer(cfg.model_name).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = TrackATriplesDataset(input_path)
    loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_triples)

    with open(output_path, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="predict"):
            anchors = batch["anchor_text"]
            a = batch["text_a"]
            b = batch["text_b"]

            s_a = score_batch(model, tokenizer, anchors, a, device, cfg.max_length)
            s_b = score_batch(model, tokenizer, anchors, b, device, cfg.max_length)

            for i in range(len(anchors)):
                pred_a_closer = (s_a[i] > s_b[i])
                out = {
                    "anchor_text": anchors[i],
                    "text_a": a[i],
                    "text_b": b[i],
                    "text_a_is_closer": bool(pred_a_closer),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[Done] Wrote: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/dev_track_a.jsonl")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="output/track_a.jsonl")
    args = parser.parse_args()

    cfg = TrainConfig()
    run(cfg, args.input, args.ckpt, args.output)
