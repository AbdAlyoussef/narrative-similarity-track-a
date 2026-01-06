from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Paths
    train_path: str = "data/train_track_a.jsonl"   # we set this to your 1900 synthetic triples
    dev_path: str = "data/dev_track_a.jsonl"       # 200 labeled dev triples (used ONLY for evaluation)
    sample_path: str = "data/sample_track_a.jsonl" # optional sanity-check set (39 labeled)

    # Model
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 384

    # Training
    seed: int = 42
    epochs: int = 3

    train_batch_size: int = 8
    eval_batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0

    use_fp16: bool = True

    # Aux (multi-head + pseudo targets)
    aux_lambda: float = 0.2
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    minilm_device: str = "cuda"

    # Output
    ckpt_dir: str = "checkpoints"
    best_ckpt_name: str = "best.pt"
