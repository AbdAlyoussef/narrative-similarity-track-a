from vllm import LLM
from vllm.sampling_params import SamplingParams
from datetime import datetime, timedelta
from huggingface_hub import login

login("Your api key here")

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


def generate_data(tokenizer, model, story):
    prompt_view1 = f"""You are given a narrative story. Rewrite it with a focus on internal perspective .

Requirements:
- Preserve the abstract theme (core ideas and motivations)
- Preserve the sequence of key events exactly as they occur
- Preserve the final outcome
- Maintain approximately the same length as the original
- Do NOT introduce new events, characters, or outcomes
- Do NOT change the ending

Focus on:
- Internal thoughts, feelings, and reflections of characters
- Emotional responses to events
- Psychological motivations and inner conflicts
- Subjective interpretation of actions

Writing style:
- Use introspective, reflective narrative voice
- Emphasize emotions over bare facts
- Use varied sentence structures (avoid copying original phrasing)
- Focus on causes and internal reasons

Story:
{story}

Output format:
<full rewritten view without titles or headers>"""

    prompt_view2 = f"""You are given a narrative story. Rewrite it with a focus on external perspective and FACTUAL description.

Requirements:
- Preserve the abstract theme (core ideas and motivations)
- Preserve the sequence of key events exactly as they occur
- Preserve the final outcome
- Maintain approximately the same length as the original
- Do NOT introduce new events, characters, or outcomes
- Do NOT change the ending

Focus on:
- Observable actions and behaviors
- External descriptions and physical details
- Objective narration of what happens
- Consequences and outcomes of actions

Writing style:
- Use objective, factual narrative voice
- Emphasize actions over emotions
- Use varied sentence structures (avoid copying original phrasing)
- Focus on consequences and visible results

Story:
{story}

Output format:
<full rewritten view without titles or headers>"""
    
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
    print(view1)
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
    print(view2)
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
        print(f" story length: {len(story)} chars")
        print(f"  View1 length: {len(view1)} chars")
        print(f"  View2 length: {len(view2)} chars")
        if idx == 10:  # Remove this line to process all stories
            break

print(f"\n✓ All {total_stories} stories processed successfully!")
print(f"✓ Output saved to: generated_data/story_views_local.jsonl")