import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, DebertaV2Tokenizer
from tqdm import tqdm

from src.config import TrainConfig
from src.data import TrackAPairwiseDataset, TrackATriplesDataset
from src.model import CrossEncoderScorer
from src.utils import set_seed, ensure_dir

def collate_pairwise(batch, tokenizer, max_length: int):
    anchors = [x["anchor"] for x in batch]
    pos = [x["pos"] for x in batch]
    neg = [x["neg"] for x in batch]

    tok_pos = tokenizer(
        anchors, pos,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt"
    )
    tok_neg = tokenizer(
        anchors, neg,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt"
    )
    return tok_pos, tok_neg

def pairwise_ranking_loss(s_pos, s_neg):
    # L = -log(sigmoid(s_pos - s_neg))
    return -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()

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
    for batch in loader:
        anchors = batch["anchor_text"]
        a = batch["text_a"]
        b = batch["text_b"]
        gold = batch["text_a_is_closer"].numpy()

        s_a = score(anchors, a).numpy()
        s_b = score(anchors, b).numpy()

        pred = (s_a > s_b)
        correct += (pred == gold).sum()
        total += len(gold)

    return correct / max(total, 1)

def run(cfg: TrainConfig):
    ensure_dir(cfg.ckpt_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Windows-safe: force slow tokenizer for DeBERTa models (avoids DebertaV2TokenizerFast bug)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    model = CrossEncoderScorer(cfg.model_name).to(device)

    train_ds = TrackAPairwiseDataset(cfg.train_path)
    dev_ds = TrackATriplesDataset(cfg.dev_path)
    sample_ds = TrackATriplesDataset(cfg.sample_path)

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
    best_path = f"{cfg.ckpt_dir}/{cfg.best_ckpt_name}"

    print(f"[Info] Train size: {len(train_ds)} | Dev size: {len(dev_ds)} | Sample size: {len(sample_ds)}")
    print(f"[Info] Model: {cfg.model_name} | max_length={cfg.max_length}")
    print(f"[Info] epochs={cfg.epochs} | train_batch_size={cfg.train_batch_size} | fp16={use_amp}")

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{cfg.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, (tok_pos, tok_neg) in enumerate(pbar, start=1):
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
        sample_acc = evaluate_accuracy(model, tokenizer, sample_ds, device, cfg.max_length, cfg.eval_batch_size)

        print(f"[Eval] Dev accuracy   : {dev_acc:.4f}")
        print(f"[Eval] Sample accuracy: {sample_acc:.4f}")

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
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--fp16", type=int, default=None, help="1 to enable fp16, 0 to disable")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.model_name: cfg.model_name = args.model_name
    if args.max_length: cfg.max_length = args.max_length
    if args.epochs: cfg.epochs = args.epochs
    if args.train_batch_size: cfg.train_batch_size = args.train_batch_size
    if args.eval_batch_size: cfg.eval_batch_size = args.eval_batch_size
    if args.fp16 is not None: cfg.use_fp16 = bool(args.fp16)

    run(cfg)
