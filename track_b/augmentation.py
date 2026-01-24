import pandas as pd
from google import genai
import csv
import json
import os
api_key = "your_api_key_here"

data = pd.read_json("data/dev_track_b.jsonl", lines=True)
client = genai.Client(api_key=api_key)

# Create generated_data folder if it doesn't exist
os.makedirs('generated_data', exist_ok=True)

# Open output file once
with open('generated_data/story_views_step_wise.jsonl', 'w', encoding='utf-8') as jsonfile:
    total_stories = len(data)
    
    # Process each story
    for idx, row in data.iterrows():
        story = row['text']  # Adjust column name if different
        print(f"Processing story {idx + 1}/{total_stories}...")
        
        prompt = f"""You are given a single narrative story.

Create TWO DIFFERENT VIEWS of the same story.

The two views MUST:
- Preserve the abstract theme (core ideas and motivations)
- Preserve the sequence of key events
- Preserve the final outcome
- Remain approximately the same length
- Not introduce new events, characters, or outcomes

The two views MUST DIFFER in:
- Narrative perspective or voice (e.g., internal vs external, reflective vs factual)
- Sentence structure and phrasing (avoid close paraphrasing)
- Descriptive focus (e.g., emotions vs actions, causes vs consequences)

Do NOT:
- Add or remove events
- Change the ending
- Reuse large portions of identical phrasing
- Add titles or headers beyond the labels below

Story:
{story}

Output format:
View 1:
<full rewritten view>

View 2:
<full rewritten view>"""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        # Parse the response to extract View 1 and View 2
        response_text = response.text
        view1 = ""
        view2 = ""

        if "View 1:" in response_text and "View 2:" in response_text:
            parts = response_text.split("View 2:")
            view1 = parts[0].replace("View 1:", "").strip()
            view2 = parts[1].strip()

        # Save to JSONL
        output_data = {'story': story, 'view1': view1, 'view2': view2}
        jsonfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        
        print(f"✓ Story {idx + 1}/{total_stories} completed")

print(f"\n✓ All {total_stories} stories processed successfully!")