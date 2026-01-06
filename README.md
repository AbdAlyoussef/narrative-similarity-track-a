# Narrative Similarity — SemEval 2026 Task 4 — Track A (Local, No Paid APIs)

This project trains an **advanced cross-encoder ranking model** locally (GPU) to solve Track A:
Given (anchor, story A, story B), predict whether A is narratively closer to the anchor than B.

## Key Features
- **Multi-Head Aspect Scoring**: Separate scoring heads for theme, action, and outcome aspects
- **Auxiliary Loss Training**: Uses MiniLM embeddings as pseudo-targets for better representation learning
- **DeBERTa-v3-base Backbone**: State-of-the-art transformer architecture
- **GPU-Accelerated Training**: Optimized for CUDA with mixed precision
- **No Paid APIs Required**: Fully local inference and training

## Performance
- **Baseline Accuracy**: ~47.5% (standard cross-encoder)
- **Enhanced Accuracy**: **56.5%** (with multi-head aspect scoring + auxiliary losses)
- **Improvement**: +9 percentage points

## What is included
- `data/train_track_a.jsonl`  -> **your 1900 synthetic triples** (used as training)
- `data/dev_track_a.jsonl`    -> 200 labeled dev triples (used only for evaluation / model selection)
- `data/sample_track_a.jsonl` -> 39 labeled examples (optional sanity-check)
- `checkpoints/best.pt`       -> Pre-trained model checkpoint (56.5% dev accuracy)

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
- Load DeBERTa-v3-base model with multi-head aspect scoring
- Train for 3 epochs with gradient accumulation and auxiliary losses
- Use MiniLM embeddings for pseudo-target computation
- Save best checkpoint to `checkpoints/best.pt`
- Show progress with loss and validation accuracy

Expected output includes:
- Device: cuda (if GPU available)
- Train size: 1900, Dev size: 200, Sample size: 39
- Final dev accuracy: **~56.5%** (with aspect scoring enhancement)

### Training Configuration
The model uses advanced training techniques:
- **Multi-Head Architecture**: Three parallel scoring heads (theme/action/outcome)
- **Auxiliary Losses**: MSE losses against MiniLM pseudo-targets (λ=0.2)
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Accumulation**: Configurable accumulation steps
- **Warmup Scheduling**: Linear warmup for stable training

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

## Technical Implementation

### Multi-Head Aspect Scoring
The model employs three parallel scoring heads that specialize in different narrative aspects:
- **Theme Head**: Captures story themes and topics
- **Action Head**: Focuses on events and character actions  
- **Outcome Head**: Emphasizes story conclusions and resolutions

Each head produces an independent similarity score, which are then combined with learned weights for the final prediction.

### Auxiliary Training with Pseudo-Targets
During training, auxiliary MSE losses guide the aspect heads using MiniLM embeddings as weak supervision:
- MiniLM computes embeddings for aspect-specific text views
- MSE losses encourage aspect heads to match these embeddings
- Improves representation learning without requiring labeled aspect data

### Architecture Details
- **Backbone**: DeBERTa-v3-base (183M parameters)
- **Max Sequence Length**: 384 tokens
- **Output**: Single similarity score (backward compatible)
- **Training**: Pairwise ranking loss + auxiliary MSE losses
- **Optimization**: AdamW with linear warmup and weight decay

## Notes
- No paid APIs are used.
- The model learns a scoring function s(anchor, story). We predict A is closer if s(anchor,A) > s(anchor,B).
- Training uses pairwise ranking loss for contrastive learning + auxiliary losses for aspect guidance.
- Model: DeBERTa-v3-base with multi-head aspect scoring and max sequence length 384 tokens.
- GPU training is recommended for reasonable speed.
- Data loading skips any entries with missing/null required fields.
- The enhanced model achieves 56.5% dev accuracy vs 47.5% baseline.

## Reproducibility & GitHub Setup

### Reproducing the 56.5% Results
```bash
# Train with the same configuration that achieved 56.5% accuracy
python -m src.train --train_batch_size 4

# Evaluate on dev set
python -m src.evaluate --data_path data/dev_track_a.jsonl --ckpt_path checkpoints/best.pt
```

### Project Structure
```
narrative-similarity-track-a/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Dataset files
│   ├── train_track_a.jsonl   # Training data (1900 triples)
│   ├── dev_track_a.jsonl     # Dev evaluation (200 triples)
│   └── sample_track_a.jsonl  # Sanity check (39 triples)
├── src/                      # Source code
│   ├── config.py             # Training configuration
│   ├── model.py              # Multi-head cross-encoder model
│   ├── train.py              # Training script with auxiliary losses
│   ├── evaluate.py           # Evaluation script
│   ├── predict.py            # Prediction/inference script
│   ├── data.py               # Data loading utilities
│   ├── utils.py              # Helper functions
│   └── verify_data.py        # Data validation
└── checkpoints/              # Model checkpoints
    └── best.pt              # Pre-trained model (56.5% accuracy)
```

### GitHub Repository Setup
This project is ready for GitHub upload. Key files to include:
- All source code in `src/`
- `README.md` and `requirements.txt`
- Example data files (or instructions to download)
- Pre-trained checkpoint `checkpoints/best.pt`

### Citation
If you use this implementation, please cite:
```
Multi-Head Aspect Scoring for Narrative Similarity
- Baseline: 47.5% dev accuracy
- Enhanced: 56.5% dev accuracy (+9pp improvement)
- Technique: Auxiliary losses with MiniLM pseudo-targets
```
