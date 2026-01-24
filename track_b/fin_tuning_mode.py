import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

print("preparing data ")
data = pd.read_json(
    "generated_data/output_transformed.jsonl",
    lines=True
)
print(data.head())
print(f"Loaded {len(data)} training examples.")



train_examples = [
    InputExample(texts=[row["view1"], row["view2"]], label=1.0)
    for _, row in data.iterrows()
]
print(f"Prepared {len(train_examples)} training examples.")
print(len(train_examples))

print("loading model")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = SentenceTransformer(
    #"sentence-transformers/gtr-t5-base",
    "all-MiniLM-L12-v2",
    device=device
)

print("data loader")

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=128   # you can go higher than contrastive
)

train_loss = losses.CosineSimilarityLoss(model)


print("start training")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=0,
    show_progress_bar=True
)

model.save("fine_tuned_model_L12")

