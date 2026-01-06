# Narrative Similarity — SemEval 2026 Task 4 — Track A (Local, No Paid APIs)

This project trains a **cross-encoder ranking model** locally (GPU) to solve Track A:
Given (anchor, story A, story B), predict whether A is narratively closer to the anchor than B.

## What is included
- `data/train_track_a.jsonl`  -> **your 1900 synthetic triples** (used as training)
- `data/dev_track_a.jsonl`    -> 200 labeled dev triples (used only for evaluation / model selection)
- `data/sample_track_a.jsonl` -> 39 labeled examples (optional sanity-check)

## 0) Environment Setup

### Using Conda (Recommended)
Create and activate conda environment:
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

### Alternative: Using pip only
If you prefer pip, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Verify CUDA
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 1) Verify the data schema (recommended)
```powershell
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
- Load DeBERTa-v3-base model
- Train for 3 epochs with gradient accumulation
- Save best checkpoint to `checkpoints/best.pt`
- Show progress with loss and validation accuracy

Expected output includes:
- Device: cuda (if GPU available)
- Train size: 1900, Dev size: 200, Sample size: 39
- Final dev accuracy: ~47.5%

## 3) Predict (write submission JSONL)
```bash
python -m src.predict --input data/dev_track_a.jsonl --ckpt checkpoints/best.pt --output output/track_a.jsonl
```

For testing with sample data:
```bash
python -m src.predict --input data/sample_track_a.jsonl --ckpt checkpoints/best.pt --output output/sample_predictions.jsonl
```

The output file will contain predictions in JSONL format with fields:
- `anchor_text`
- `text_a`
- `text_b` 
- `text_a_is_closer` (predicted boolean)

## Notes
- No paid APIs are used.
- The model learns a scoring function s(anchor, story). We predict A is closer if s(anchor,A) > s(anchor,B).
- Training uses pairwise ranking loss for contrastive learning.
- Model: DeBERTa-v3-base with max sequence length 384 tokens.
- GPU training is recommended for reasonable speed.
- Data loading skips any entries with missing/null required fields.

## Troubleshooting
- If you get "ModuleNotFoundError: No module named 'torch'", ensure PyTorch is installed with CUDA support.
- For CUDA issues, verify installation with: `python -c "import torch; print(torch.cuda.is_available())"`
- If training fails, try reducing batch size or using CPU-only mode.
- Data verification: Use `python -m src.verify_data` to check data integrity.
