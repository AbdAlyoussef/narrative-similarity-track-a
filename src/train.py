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
from src.aspects import aspect_views

# Add imports for auxiliary losses
import re
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

RUBRIC = (
    "Judge narrative similarity based on:\n"
    "1) Abstract Theme (ideas/motives),\n"
    "2) Course of Action (key events/turning points),\n"
    "3) Outcomes (results).\n"
    "Ignore names, writing style, and setting unless essential.\n\n"
)

def format_pair(anchor: str, cand: str) -> str:
    return f"{RUBRIC}ANCHOR:\n{anchor}\n\nCANDIDATE:\n{cand}"

def collate_pairwise(batch, tokenizer, max_length: int):
    anchors = [x["anchor"] for x in batch]
    pos = [x["pos"] for x in batch]
    neg = [x["neg"] for x in batch]

    pos_pairs = [format_pair(a, p) for a, p in zip(anchors, pos)]
    neg_pairs = [format_pair(a, n) for a, n in zip(anchors, neg)]

    tok_pos = tokenizer(
        pos_pairs,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt"
    )
    tok_neg = tokenizer(
        neg_pairs,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt"
    )
    return tok_pos, tok_neg, batch  # <- add raw batch

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

# Add helper functions for aspect-based auxiliary losses
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

# OLD aspect_views function - replaced by src.aspects.aspect_views (Stage 2 improvement)
# def aspect_views(text: str):
#     """
#     Deterministic, lightweight aspect extraction (no LLM, no heavy summarizer):
#       - Theme: first 1 sentence
#       - Outcome: last 1 sentence
#       - Action: up to 3 middle sentences (or next sentences)
#     """
#     sents = split_sentences(text)
#     if len(sents) == 0:
#         return ("", "", "")
#     if len(sents) == 1:
#         return (sents[0], sents[0], sents[0])

#     theme = sents[0]
#     outcome = sents[-1]

#     middle = sents[1:-1]
#     if len(middle) == 0:
#         action = sents[0]  # fallback
#     else:
#         action = " ".join(middle[:3])  # first 3 middle sentences
#     return (theme, action, outcome)

class MiniLMPseudoTargets:
    """
    Builds pseudo-targets y_T, y_A, y_O using MiniLM cosine similarity
    between aspect views of anchor vs candidate.

    Uses embedding cache for speed.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.cache = {}  # maps aspect_text -> embedding tensor (on CPU for memory stability)

    @torch.no_grad()
    def _embed_cached(self, texts):
        # returns a list of CPU tensors
        embs = []
        missing = []
        missing_idx = []

        for i, t in enumerate(texts):
            key = t
            if key in self.cache:
                embs.append(self.cache[key])
            else:
                embs.append(None)
                missing.append(key)
                missing_idx.append(i)

        if missing:
            # encode returns numpy or tensor depending on settings; we convert to torch
            new = self.embedder.encode(
                missing,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            new = new.detach().cpu()
            for k, vec in zip(missing, new):
                self.cache[k] = vec
            for idx, vec in zip(missing_idx, new):
                embs[idx] = vec

        return embs

    @staticmethod
    def _cosine_batch(a_cpu, b_cpu):
        # both are lists of CPU tensors normalized; cosine = dot
        a = torch.stack(a_cpu, dim=0)
        b = torch.stack(b_cpu, dim=0)
        return (a * b).sum(dim=1).clamp(-1.0, 1.0)

    @torch.no_grad()
    def targets_for_pairs(self, anchors, cands):
        """
        anchors, cands: lists[str] same length
        Returns:
          yT, yA, yO as torch.FloatTensor on the *training device* (GPU if available)
        """
        a_theme, a_action, a_out = [], [], []
        c_theme, c_action, c_out = [], [], []

        for a, c in zip(anchors, cands):
            at, aa, ao = aspect_views(a)
            ct, ca, co = aspect_views(c)
            a_theme.append(at); a_action.append(aa); a_out.append(ao)
            c_theme.append(ct); c_action.append(ca); c_out.append(co)

        aT = self._embed_cached(a_theme)
        cT = self._embed_cached(c_theme)
        aA = self._embed_cached(a_action)
        cA = self._embed_cached(c_action)
        aO = self._embed_cached(a_out)
        cO = self._embed_cached(c_out)

        yT = self._cosine_batch(aT, cT)
        yA = self._cosine_batch(aA, cA)
        yO = self._cosine_batch(aO, cO)

        # map cosine [-1,1] -> [0,1] (helps stabilize MSE)
        yT = (yT + 1.0) / 2.0
        yA = (yA + 1.0) / 2.0
        yO = (yO + 1.0) / 2.0

        # move to training device for loss computation
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return yT.to(train_device), yA.to(train_device), yO.to(train_device)

def run(cfg: TrainConfig):
    ensure_dir(cfg.ckpt_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Windows-safe: force slow tokenizer for DeBERTa models (avoids DebertaV2TokenizerFast bug)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    model = CrossEncoderScorer(cfg.model_name).to(device)

    # Initialize pseudo-target builder for auxiliary losses
    pseudo = MiniLMPseudoTargets(model_name=cfg.minilm_model, device=cfg.minilm_device)

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

        for step, (tok_pos, tok_neg, batch_raw) in enumerate(pbar, start=1):
            tok_pos = {k: v.to(device) for k, v in tok_pos.items()}
            tok_neg = {k: v.to(device) for k, v in tok_neg.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Get combined score AND aspect scores for pos/neg
                s_pos, t_pos, a_pos, o_pos = model.forward_aspects(
                    tok_pos["input_ids"], tok_pos["attention_mask"]
                )
                s_neg, t_neg, a_neg, o_neg = model.forward_aspects(
                    tok_neg["input_ids"], tok_neg["attention_mask"]
                )

                # Main ranking loss unchanged
                rank_loss = pairwise_ranking_loss(s_pos, s_neg)

                # Pseudo targets (MiniLM cosine on aspect views)
                # NOTE: anchors are the same; candidates differ for pos/neg
                anchors = [x["anchor"] for x in batch_raw]   # we'll define batch_raw below
                pos_txt = [x["pos"] for x in batch_raw]
                neg_txt = [x["neg"] for x in batch_raw]

                yT_pos, yA_pos, yO_pos = pseudo.targets_for_pairs(anchors, pos_txt)
                yT_neg, yA_neg, yO_neg = pseudo.targets_for_pairs(anchors, neg_txt)

                # Normalize model aspect scores to [0,1] with sigmoid to match targets
                t_pos_n = torch.sigmoid(t_pos)
                a_pos_n = torch.sigmoid(a_pos)
                o_pos_n = torch.sigmoid(o_pos)
                t_neg_n = torch.sigmoid(t_neg)
                a_neg_n = torch.sigmoid(a_neg)
                o_neg_n = torch.sigmoid(o_neg)

                mse = torch.nn.functional.mse_loss
                aux_loss = (
                    mse(t_pos_n, yT_pos) + mse(a_pos_n, yA_pos) + mse(o_pos_n, yO_pos) +
                    mse(t_neg_n, yT_neg) + mse(a_neg_n, yA_neg) + mse(o_neg_n, yO_neg)
                ) / 2.0

                loss = rank_loss + cfg.aux_lambda * aux_loss

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
