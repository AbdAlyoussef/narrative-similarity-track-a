import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import gc  # Add garbage collection

data = pd.read_json("generated_data/story_views_local_postProcessed.jsonl", lines=True)
print (f"Loaded {len(data)} training examples.")
print(data.head())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear CUDA cache at start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

def info_nce_loss(z1, z2, temperature=0.05):
    """
    z1, z2: tensors of shape (N, D)
    """
    N = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T)
    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    pos_idx = torch.arange(N, device=z.device)
    positives = torch.cat([pos_idx + N, pos_idx])
    loss = F.cross_entropy(sim / temperature, positives)
    return loss

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def train_contrastive(model, epochs=10, batch_size=32, learning_rate=1e-4, temperature=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tokenizer = model.tokenizer
    model.train()
    
    for epoch in range(epochs):
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(shuffled_data), batch_size):
            batch_data = shuffled_data.iloc[i:i + batch_size]
            views_1 = batch_data['view1'].tolist()
            views_2 = batch_data['view2'].tolist()
            
            encoded_1 = tokenizer(views_1, padding=True, truncation=True, return_tensors='pt').to(device)
            encoded_2 = tokenizer(views_2, padding=True, truncation=True, return_tensors='pt').to(device)
            
            model_output_1 = model[0].auto_model(**encoded_1)
            model_output_2 = model[0].auto_model(**encoded_2)
            
            z1 = mean_pooling(model_output_1, encoded_1['attention_mask'])
            z2 = mean_pooling(model_output_2, encoded_2['attention_mask'])
            
            optimizer.zero_grad()
            loss = info_nce_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear memory after each batch
            del encoded_1, encoded_2, model_output_1, model_output_2, z1, z2, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Clear cache after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    return model

model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Reduce batch size significantly
model = train_contrastive(
    model=model,
    epochs=10,
    batch_size=128,  # Reduced from 240
    learning_rate=1e-5,
    temperature=0.05
)

model.save("fine_tuned_model_L6_with_google_gemma_data")
print("Model saved to 'fine_tuned_model_L6_with_google_gemma_data' directory")
