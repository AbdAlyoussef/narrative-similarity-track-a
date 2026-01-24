import re

import torch
import pandas as pd
import json
import os
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from huggingface_hub import login

login("Your_HuggingFace_Token_Here")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 1. Configure Model


model_id = "google/gemma-3-1b-it"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading model and tokenizer...")
# 2. Initialize Tokenizer and Model

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Model loaded: {model_id}")

# Load the data
data = pd.read_json("data/dev_track_b.jsonl", lines=True)
print(f"Loaded {len(data)} stories")

# Create generated_data folder if it doesn't exist
os.makedirs('generated_data', exist_ok=True)

def parse_views(text):
    """Extract View 1 and View 2 from the model's response"""
    view1 = ""
    view2 = ""
    
    if "View 1:" in text and "View 2:" in text:
        parts = text.split("View 2:")
        view1 = parts[0].replace("View 1:", "").strip()
        view2 = parts[1].strip()
    
    # Remove thinking tags if present
    view1 = re.sub(r"<think>.*?</think>", "", view1, flags=re.DOTALL).strip()
    view2 = re.sub(r"<think>.*?</think>", "", view2, flags=re.DOTALL).strip()
    
    return view1, view2
def generate_data(tokenizer, model, story):
    prompt_view1 = f"""You are given a narrative story. Rewrite it as a DIFFERENT narrative expression of the same story.

STRICT RULES (follow all):
- Keep the same events in the same order
- Keep the same ending and final outcome
- Keep approximately the same length
- Change sentence structure in EVERY sentence
- Replace metaphors, descriptions, and phrasing with different expressions
- Do NOT reuse phrases longer than 3 consecutive words
- Do NOT introduce new events, characters, or outcomes
- Character names may be changed
- Preserve the original emotional tone and theme

Rewrite Rules:
- Split long sentences into shorter ones OR merge short sentences
- Change passive voice to active OR active to passive where possible
- Reorder clauses inside sentences

Story:
{story}

Output:
<rewritten narrative text only>"""

    prompt_view2 = f"""You are given a narrative story. Rewrite it as a FACTUAL REPORT.

STRICT RULES (follow all):
- Describe only observable actions and outcomes
- Remove emotional language, metaphors, and inner thoughts
- Use simple, direct sentences
- Use past tense only
- Keep the same events in the same order
- Keep the same ending and final outcome
- Keep approximately the same length
- Do NOT introduce new events, characters, or outcomes
- Do NOT reuse phrases longer than 3 consecutive words
- Character names may be changed

Writing Style:
- Neutral, objective tone
- External observer point of view
- No adjectives unless required for clarity
- No figurative language

Story:
{story}

Output:
<factual rewritten text only without line breaks>"""
    
    # Generate View 1
    messages1 = [[{"role": "user", "content": [{"type": "text", "text": prompt_view1}]}]]
    inputs1 = tokenizer.apply_chat_template(
        messages1, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device).to(torch.bfloat16)
    
    with torch.inference_mode():
        story_word_count = len(story.split())
        max_tokens = int((story_word_count * 1.3) + 100)
        outputs1 = model.generate(**inputs1, max_new_tokens=max_tokens)
    
    view1 = tokenizer.batch_decode(outputs1)[0].split("model")[-1].strip()
    view1 = re.sub(r"<think>.*?</think>", "", view1, flags=re.DOTALL).strip()
    # Generate View 2
    messages2 = [[{"role": "user", "content": [{"type": "text", "text": prompt_view2}]}]]
    inputs2 = tokenizer.apply_chat_template(
        messages2, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device).to(torch.bfloat16)
    
    with torch.inference_mode():
        outputs2 = model.generate(**inputs2, max_new_tokens=max_tokens)
    
    view2 = tokenizer.batch_decode(outputs2)[0].split("model")[-1].strip()
    view2 = re.sub(r"<think>.*?</think>", "", view2, flags=re.DOTALL).strip()
    return view1, view2


with open('generated_data/story_views_local.jsonl', 'w', encoding='utf-8') as jsonfile:
    total_stories = len(data)
    
    # Process each story
    for idx, row in data.iterrows():
        story = row['text']
        print(f"\nProcessing story {idx + 1}/{total_stories}...")
        
        view1, view2 = generate_data(tokenizer, model, story)
    
        
        # Save to JSONL
        output_data = {'story': story, 'view1': view1, 'view2': view2}
        jsonfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        
        print(f"✓ Story {idx + 1}/{total_stories} completed")
        # print(f" story length: {len(story)} chars")
        # print(f"  View1 length: {len(view1)} chars")
        # print(f"  View2 length: {len(view2)} chars")
print(f"\n✓ All {total_stories} stories processed successfully!")
print(f"✓ Output saved to: generated_data/story_views_local.jsonl")