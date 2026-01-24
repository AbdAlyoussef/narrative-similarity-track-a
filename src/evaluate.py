import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer
from tqdm import tqdm

from src.config import TrainConfig
from src.data import TrackATriplesDataset, read_jsonl
from src.model import CrossEncoderScorer

@torch.no_grad()
def evaluate_accuracy(model, tokenizer, ds, device, max_length: int, batch_size: int):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    def score(anchors, cands):
        tok = tokenizer(
            anchors, cands,
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        return model(tok["input_ids"], tok["attention_mask"]).detach().cpu()

    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Evaluating"):
        anchors = batch["anchor_text"]
        a = batch["text_a"]
        b = batch["text_b"]
        gold = batch["text_a_is_closer"].numpy()

        s_a = score(anchors, a).numpy()
        s_b = score(anchors, b).numpy()

        pred = (s_a > s_b)
        correct += (pred == gold).sum()
        total += len(gold)

    accuracy = correct / max(total, 1)
    return accuracy.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dev_track_a.jsonl",
                       help="Path to evaluation data")
    parser.add_argument("--train_path", type=str, default=None,
                       help="Optional path to training data to check for overlap")
    parser.add_argument("--fail_on_leak", type=int, default=1,
                       help="1 to abort if eval overlaps training data")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/best.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    args = parser.parse_args()

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    if "deberta" in cfg.model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)

    model = CrossEncoderScorer(cfg.model_name).to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Optional overlap check to prevent leakage
    if not args.train_path:
        last_split = "output/last_split.json"
        if os.path.exists(last_split):
            with open(last_split, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            args.train_path = split_info.get("train_path")
            if args.data_path == "data/dev_track_a.jsonl":
                args.data_path = split_info.get("eval_path", args.data_path)
            print(f"[Info] Using split info from {last_split}")
    if args.train_path:
        train_rows = read_jsonl(args.train_path)
        eval_rows = read_jsonl(args.data_path)
        train_keys = {
            (r["anchor_text"], r["text_a"], r["text_b"], bool(r["text_a_is_closer"]))
            for r in train_rows
        }
        eval_keys = {
            (r["anchor_text"], r["text_a"], r["text_b"], bool(r["text_a_is_closer"]))
            for r in eval_rows
        }
        overlap = train_keys.intersection(eval_keys)
        if overlap:
            print(f"[Warn] Overlap detected between train and eval: {len(overlap)} rows")
            if args.fail_on_leak:
                print("[Warn] Aborting evaluation due to leakage. Use a held-out eval set.")
                sys.exit(1)

    # Load data
    ds = TrackATriplesDataset(args.data_path)

    # Evaluate
    accuracy = evaluate_accuracy(model, tokenizer, ds, device, cfg.max_length, args.batch_size)

    print(f"{accuracy:.4f}")

if __name__ == "__main__":
    main()
