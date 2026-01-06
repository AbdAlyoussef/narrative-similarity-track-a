# Narrative Similarity - SemEval 2026 Task 4 Track A (Local, no paid APIs)

This project trains a cross-encoder ranking model to solve Track A:
given (anchor, story A, story B), predict whether A is narratively closer to the anchor than B.

## Key Features
- Multi-head aspect scoring (theme, action, outcome)
- Auxiliary losses from MiniLM cosine pseudo-targets
- DeBERTa-v3-base backbone
- GPU-friendly training with optional FP16
- Fully local training and inference (no paid APIs)

## What is included
- `data/train_track_a.jsonl` (1900 synthetic triples)
- `data/dev_track_a.jsonl` (200 labeled dev triples)
- `data/sample_track_a.jsonl` (39 labeled examples)
- `checkpoints/` (several checkpoints, including `best_stage4.pt`)

## Environment Setup

### Conda (recommended)
```bash
conda create -n narrative-sim python=3.12 -y
conda activate narrative-sim
```

### Install PyTorch with CUDA (for GPU support)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Install other dependencies
```bash
pip install -r requirements.txt
```

### Alternative: pip only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Verify CUDA
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 1) Verify the data schema (recommended)
```bash
python -m src.verify_data --path data/train_track_a.jsonl
python -m src.verify_data --path data/dev_track_a.jsonl
```

## 2) Train
```bash
python -m src.train
```

If you get CUDA OOM, reduce batch size:
```bash
python -m src.train --train_batch_size 4
```

Training will:
- Load DeBERTa-v3-base with multi-head aspect scoring
- Train with pairwise ranking loss + auxiliary MSE losses
- Use MiniLM embeddings as pseudo-targets
- Save the best checkpoint to `checkpoints/best_stage4.pt` (default in `src/config.py`)

## 3) Predict (write submission JSONL)
```bash
python -m src.predict --input data/dev_track_a.jsonl --ckpt checkpoints/best_stage4.pt --output output/track_a.jsonl
```

For a quick sanity run:
```bash
python -m src.predict --input data/sample_track_a.jsonl --ckpt checkpoints/best_stage4.pt --output output/sample_predictions.jsonl
```

The output JSONL contains:
- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (predicted boolean)

## 4) Evaluate
```bash
python -m src.evaluate --data_path data/dev_track_a.jsonl --ckpt_path checkpoints/best_stage4.pt
```

## Implementation Notes

### Multi-Head Aspect Scoring
The model has three parallel scoring heads:
- Theme head: story themes and topics
- Action head: events and turning points
- Outcome head: conclusions and resolutions

These are combined with learnable weights for the final score.

### Auxiliary Training with Pseudo-Targets
MiniLM computes cosine similarities between aspect views of anchor and candidate:
- Aspect views are extracted with simple sentence rules plus TF-IDF selection
- Cosine targets are mapped to [0, 1] and used in MSE losses

### Scoring Inputs
Training and inference prepend a rubric to each anchor/candidate pair, then score:
`s(anchor, A)` vs `s(anchor, B)` and predict A is closer if `s(anchor, A) > s(anchor, B)`.

## Configuration
Defaults live in `src/config.py`:
- `model_name`, `max_length`
- `epochs`, `train_batch_size`, `eval_batch_size`
- `aux_lambda`, `minilm_model`, `minilm_device`
- `best_ckpt_name` (default: `best_stage4.pt`)

Most values can be overridden via `src.train` CLI flags.

## Project Structure
```
narrative-similarity-track-a/
  README.md
  requirements.txt
  data/
    train_track_a.jsonl
    dev_track_a.jsonl
    sample_track_a.jsonl
  src/
    config.py
    model.py
    train.py
    evaluate.py
    predict.py
    data.py
    aspects.py
    utils.py
    verify_data.py
  checkpoints/
    best.pt
    best_stage2.pt
    best_stage4.pt
```

## Notes
- No paid APIs are used.
- GPU is recommended for reasonable training time.
- On Windows, the training code forces `use_fast=False` for DeBERTa tokenizers.
