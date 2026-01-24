"""
Compute cosine similarity between view1 and view2 from story_views.jsonl
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the story views data
data = pd.read_json("generated_data/output_transformed copy.jsonl", lines=True)
print(f"Loaded {len(data)} stories with views")

# Initialize the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded: all-MiniLM-L6-v2")

# Encode view1 and view2
print("\nEncoding view1...")
view1_embeddings = model.encode(data["view1"].tolist(), show_progress_bar=True)

print("Encoding view2...")
view2_embeddings = model.encode(data["view2"].tolist(), show_progress_bar=True)

# Compute cosine similarity between view1 and view2 for each story
print("\nComputing cosine similarities...")
similarities = []
for i in range(len(data)):
    sim = cos_sim(view1_embeddings[i], view2_embeddings[i]).item()
    similarities.append(sim)

# Add similarities to dataframe
data["cosine_similarity"] = similarities

# Display results
print("\n" + "="*80)
print("COSINE SIMILARITY RESULTS")
print("="*80)
for idx, row in data.iterrows():
    print(f"\nStory {idx + 1}:")
    print(f"  Cosine Similarity: {row['cosine_similarity']:.4f}")
    print(f"  Original Story (first 100 chars): {row['story'][:100]}...")

print("\n" + "="*80)
print(f"Average Cosine Similarity: {np.mean(similarities):.4f}")
print(f"Min Cosine Similarity: {np.min(similarities):.4f}")
print(f"Max Cosine Similarity: {np.max(similarities):.4f}")
print(f"Std Deviation: {np.std(similarities):.4f}")
print("="*80)


def compute_cross_story_similarities(view1_embeddings, view2_embeddings):
    """
    Compute cosine similarity between all views across different stories.
    Returns a matrix where each cell [i,j] is the similarity between views from different stories.
    """
    # Combine all views into a single array
    all_views = np.vstack([view1_embeddings, view2_embeddings])
    n_stories = len(view1_embeddings)
    
    # Create labels: 0 to n-1 for view1s, 0 to n-1 for view2s (story indices)
    story_indices = list(range(n_stories)) + list(range(n_stories))
    
    # Compute all pairwise similarities
    similarity_matrix = cos_sim(all_views, all_views).cpu().numpy()
 
    
    # Extract cross-story similarities (excluding same-story comparisons)
    cross_story_sims = []
    same_story_sims = []
    
    for i in range(len(all_views)):
        for j in range(i+1, len(all_views)):
            story_i = story_indices[i]
            story_j = story_indices[j]
            
            if story_i == story_j:
                # Same story comparison (view1 vs view2)
                same_story_sims.append(similarity_matrix[i, j])
            else:
                # Cross-story comparison
                cross_story_sims.append(similarity_matrix[i, j])
    
    return np.array(cross_story_sims), np.array(same_story_sims), similarity_matrix


# Compute cross-story similarities
print("\n" + "="*80)
print("COMPUTING CROSS-STORY SIMILARITIES")
print("="*80)

cross_story_sims, same_story_sims, full_matrix = compute_cross_story_similarities(
    view1_embeddings, view2_embeddings
)

print(f"\nTotal comparisons:")
print(f"  Same-story comparisons (view1 vs view2): {len(same_story_sims)}")
print(f"  Cross-story comparisons: {len(cross_story_sims)}")

print("\n" + "="*80)
print("COMPARISON: SAME-STORY vs CROSS-STORY SIMILARITIES")
print("="*80)

print("\nSame-Story Similarities (View1 vs View2 from same story):")
print(f"  Average: {np.mean(same_story_sims):.4f}")
print(f"  Min: {np.min(same_story_sims):.4f}")
print(f"  Max: {np.max(same_story_sims):.4f}")
print(f"  Std: {np.std(same_story_sims):.4f}")
print(f"  Count < 0.4: {np.sum(same_story_sims > 0.9)}")

print("\nCross-Story Similarities (All views from different stories):")
print(f"  Average: {np.mean(cross_story_sims):.4f}")
print(f"  Min: {np.min(cross_story_sims):.4f}")
print(f"  Max: {np.max(cross_story_sims):.4f}")
print(f"  Std: {np.std(cross_story_sims):.4f}")

print("\n" + "-"*80)
print(f"Difference (Same-Story - Cross-Story): {np.mean(same_story_sims) - np.mean(cross_story_sims):.4f}")
print("-"*80)

# Show top cross-story similarities (potential confusions)
print("\n" + "="*80)
print("TOP 5 CROSS-STORY SIMILARITIES (Potential Confusions)")
print("="*80)

# Get indices of top cross-story similarities
n_stories = len(view1_embeddings)
cross_story_details = []

for i in range(len(view1_embeddings) + len(view2_embeddings)):
    for j in range(i+1, len(view1_embeddings) + len(view2_embeddings)):
        story_i = i if i < n_stories else i - n_stories
        story_j = j if j < n_stories else j - n_stories
        
        if story_i != story_j:
            view_i_type = "view1" if i < n_stories else "view2"
            view_j_type = "view1" if j < n_stories else "view2"
            sim = full_matrix[i, j]
            cross_story_details.append((story_i, view_i_type, story_j, view_j_type, sim))

# Sort by similarity (descending)
cross_story_details.sort(key=lambda x: x[4], reverse=True)

for idx, (story_i, view_i_type, story_j, view_j_type, sim) in enumerate(cross_story_details[:5]):
    print(f"\n{idx+1}. Similarity: {sim:.4f}")
    print(f"   Story {story_i+1} ({view_i_type}) <-> Story {story_j+1} ({view_j_type})")

print("\n" + "="*80)

