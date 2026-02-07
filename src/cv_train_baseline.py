import argparse
import copy
import os
import json
import random

from src.config import TrainConfig
from src.data import read_jsonl, TrackAPairwiseRowsDataset, TrackATriplesRowsDataset
from src.train_baseline import run as run_baseline

def make_folds(rows, k, seed):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    fold_sizes = [len(rows) // k + (1 if i < len(rows) % k else 0) for i in range(k)]
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(idx[start:start + size])
        start += size
    return folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train_track_a.jsonl")
    parser.add_argument("--dev_path", type=str, default="data/dev_track_a.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig()
    train_rows = read_jsonl(args.train_path)
    dev_rows = read_jsonl(args.dev_path)

    folds = make_folds(dev_rows, args.folds, args.seed)
    scores = []

    for i, fold_idx in enumerate(folds, start=1):
        fold_set = set(fold_idx)
        dev_fold = [dev_rows[j] for j in fold_idx]
        train_fold = train_rows + [r for j, r in enumerate(dev_rows) if j not in fold_set]

        # Write temporary splits for leakage-safe baseline training
        out_dir = os.path.join(cfg.ckpt_dir, "cv_baseline_splits")
        os.makedirs(out_dir, exist_ok=True)
        train_path = os.path.join(out_dir, f"train_fold{i}.jsonl")
        eval_path = os.path.join(out_dir, f"eval_fold{i}.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_fold:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for r in dev_fold:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        cfg_fold = copy.deepcopy(cfg)
        cfg_fold.ckpt_dir = os.path.join(cfg.ckpt_dir, "cv_baseline")
        cfg_fold.best_ckpt_name = f"best_baseline_fold{i}.pt"

        print(f"[Fold {i}/{args.folds}] Train rows: {len(train_fold)} | Dev rows: {len(dev_fold)}")
        best_acc = run_baseline(cfg_fold, train_path, eval_path)
        scores.append(best_acc)

    avg = sum(scores) / max(len(scores), 1)
    print(f"[CV-Baseline] Fold accuracies: {', '.join([f'{s:.4f}' for s in scores])}")
    print(f"[CV-Baseline] Mean accuracy: {avg:.4f}")

if __name__ == "__main__":
    main()
