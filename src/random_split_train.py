import argparse
import random

from src.config import TrainConfig
from src.data import read_jsonl, TrackAPairwiseRowsDataset, TrackATriplesRowsDataset
from src.train import run_with_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train_track_a.jsonl")
    parser.add_argument("--dev_path", type=str, default="data/dev_track_a.jsonl")
    parser.add_argument("--sample_path", type=str, default="data/sample_track_a.jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    rows.extend(read_jsonl(args.train_path))
    rows.extend(read_jsonl(args.dev_path))
    rows.extend(read_jsonl(args.sample_path))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    split_idx = int(len(rows) * args.train_ratio)
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    cfg = TrainConfig()
    train_ds = TrackAPairwiseRowsDataset(train_rows)
    eval_ds = TrackATriplesRowsDataset(eval_rows)

    print(f"[Split] Total rows: {len(rows)} | Train: {len(train_rows)} | Eval: {len(eval_rows)}")
    run_with_datasets(cfg, train_ds, eval_ds, sample_ds=None)

if __name__ == "__main__":
    main()
