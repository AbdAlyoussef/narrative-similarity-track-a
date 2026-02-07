import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import TrainConfig
from src.data import TrackAPairwiseDataset, TrackATriplesDataset, read_jsonl
from src.model_single_head import SingleHeadCrossEncoder
from src.utils import set_seed, ensure_dir
from src.train import collate_pairwise, pairwise_ranking_loss, evaluate_accuracy

def run(cfg: TrainConfig, train_path: str, eval_path: str):
    ensure_dir(cfg.ckpt_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    model = SingleHeadCrossEncoder(cfg.model_name).to(device)

    # Leak check: ensure eval rows do not appear in training rows
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
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
        raise ValueError(f"Leakage detected: {len(overlap)} eval rows appear in training set.")

    train_ds = TrackAPairwiseDataset(train_path)
    dev_ds = TrackATriplesDataset(eval_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pairwise(b, tokenizer, cfg.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_amp = cfg.use_fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    best_path = f"{cfg.ckpt_dir}/best_baseline.pt"

    print(f"[Info] Train size: {len(train_ds)} | Dev size: {len(dev_ds)}")
    print(f"[Info] Model: {cfg.model_name} | max_length={cfg.max_length}")
    print(f"[Info] epochs={cfg.epochs} | train_batch_size={cfg.train_batch_size} | fp16={use_amp}")

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{cfg.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, (tok_pos, tok_neg, _) in enumerate(pbar, start=1):
            tok_pos = {k: v.to(device) for k, v in tok_pos.items()}
            tok_neg = {k: v.to(device) for k, v in tok_neg.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                s_pos = model(tok_pos["input_ids"], tok_pos["attention_mask"])
                s_neg = model(tok_neg["input_ids"], tok_neg["attention_mask"])
                loss = pairwise_ranking_loss(s_pos, s_neg)

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        dev_acc = evaluate_accuracy(model, tokenizer, dev_ds, device, cfg.max_length, cfg.eval_batch_size)
        print(f"[Eval] Dev accuracy   : {dev_acc:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(
                {"model_state": model.state_dict(), "model_name": cfg.model_name, "max_length": cfg.max_length},
                best_path
            )
            print(f"[Info] Saved new best checkpoint -> {best_path}")

    print(f"[Done] Best dev accuracy: {best_acc:.4f}")
    print(f"[Done] Best checkpoint  : {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    args = parser.parse_args()

    cfg = TrainConfig()
    run(cfg, args.train_path, args.eval_path)
