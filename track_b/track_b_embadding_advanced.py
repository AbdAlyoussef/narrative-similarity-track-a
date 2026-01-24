import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import gc  # Add garbage collection

data = pd.read_json("generated_data/output_combined.jsonl", lines=True)
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

def info_nce_loss_multi_view(embeddings, temperature=0.05):
    """
    Compute InfoNCE loss for multiple views
    embeddings: list of tensors, each of shape (N, D)
    """
    # Normalize all embeddings
    embeddings = [F.normalize(emb, dim=1) for emb in embeddings]
    
    # Concatenate all views
    z = torch.cat(embeddings, dim=0)  # Shape: (num_views * N, D)
    
    N = embeddings[0].size(0)
    num_views = len(embeddings)
    
    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature
    
    # Mask out self-similarities
    mask = torch.eye(num_views * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    
    # Create positive pairs: all views of the same sample are positives
    total_loss = 0
    for i in range(num_views * N):
        view_idx = i // N
        sample_idx = i % N
        
        # All other views of the same sample are positives
        positive_indices = []
        for v in range(num_views):
            if v != view_idx:
                positive_indices.append(v * N + sample_idx)
        
        # Compute loss for this anchor against all positives
        logits = sim[i]
        labels = torch.tensor(positive_indices, device=z.device)
        
        # Use log-sum-exp trick for multiple positives
        pos_sim = logits[labels]
        total_loss += -torch.logsumexp(pos_sim, dim=0) + torch.logsumexp(logits, dim=0)
    
    return total_loss / (num_views * N)

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
            
            # Get all 4 views
            views_1 = batch_data['view1'].tolist()
            views_2 = batch_data['view2'].tolist()
            views_3 = batch_data['view3'].tolist()
            views_4 = batch_data['view4'].tolist()
            
            # Encode all views
            encoded_1 = tokenizer(views_1, padding=True, truncation=True, return_tensors='pt').to(device)
            encoded_2 = tokenizer(views_2, padding=True, truncation=True, return_tensors='pt').to(device)
            encoded_3 = tokenizer(views_3, padding=True, truncation=True, return_tensors='pt').to(device)
            encoded_4 = tokenizer(views_4, padding=True, truncation=True, return_tensors='pt').to(device)
            
            # Get model outputs
            model_output_1 = model[0].auto_model(**encoded_1)
            model_output_2 = model[0].auto_model(**encoded_2)
            model_output_3 = model[0].auto_model(**encoded_3)
            model_output_4 = model[0].auto_model(**encoded_4)
            
            # Pool embeddings
            z1 = mean_pooling(model_output_1, encoded_1['attention_mask'])
            z2 = mean_pooling(model_output_2, encoded_2['attention_mask'])
            z3 = mean_pooling(model_output_3, encoded_3['attention_mask'])
            z4 = mean_pooling(model_output_4, encoded_4['attention_mask'])
            
            optimizer.zero_grad()
            
            # Compute loss for all 4 views together
            loss = info_nce_loss_multi_view([z1, z2, z3, z4], temperature=temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear memory after each batch
            del encoded_1, encoded_2, encoded_3, encoded_4
            del model_output_1, model_output_2, model_output_3, model_output_4
            del z1, z2, z3, z4, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Clear cache after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    return model

model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Reduce batch size for 4 views
model = train_contrastive(
    model=model,
    epochs=10,
    batch_size=64,  # Further reduced due to 4 views
    learning_rate=1e-5,
    temperature=0.05
)

model.save("fine_tuned_model_L6_with_google_gemma_data")
print("Model saved to 'fine_tuned_model_L6_with_google_gemma_data' directory")
