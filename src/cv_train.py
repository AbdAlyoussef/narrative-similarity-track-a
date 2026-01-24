import argparse
import copy
import os
import random

from src.config import TrainConfig
from src.data import read_jsonl, TrackAPairwiseRowsDataset, TrackATriplesRowsDataset
from src.train import run_with_datasets

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

        cfg_fold = copy.deepcopy(cfg)
        cfg_fold.ckpt_dir = os.path.join(cfg.ckpt_dir, "cv")
        cfg_fold.best_ckpt_name = f"best_fold{i}.pt"

        print(f"[Fold {i}/{args.folds}] Train rows: {len(train_fold)} | Dev rows: {len(dev_fold)}")
        train_ds = TrackAPairwiseRowsDataset(train_fold)
        dev_ds = TrackATriplesRowsDataset(dev_fold)
        best_acc = run_with_datasets(cfg_fold, train_ds, dev_ds, sample_ds=None)
        scores.append(best_acc)

    avg = sum(scores) / max(len(scores), 1)
    print(f"[CV] Fold accuracies: {', '.join([f'{s:.4f}' for s in scores])}")
    print(f"[CV] Mean accuracy: {avg:.4f}")

if __name__ == "__main__":
    main()
