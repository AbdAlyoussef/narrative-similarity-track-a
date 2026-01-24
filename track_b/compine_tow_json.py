import pandas as pd
import json

# Load both files
output_file = "generated_data/output.jsonl"
story_views_file = "generated_data/story_views_local_postProcessed.jsonl"

# Read the files
output_data = []
with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        output_data.append(json.loads(line))

story_views_data = []
with open(story_views_file, 'r', encoding='utf-8') as f:
    for line in f:
        story_views_data.append(json.loads(line))

# Combine the data
combined_data = []
for idx, story_view in enumerate(story_views_data):
    # Find matching entry in output_data by index
    output_entry = output_data[idx] if idx < len(output_data) else None
    
    if output_entry:
        # Get the nested data from output.jsonl
        nested_data = output_entry.get(str(idx), {})
        
        combined_entry = {
            "story": story_view.get("story", ""),
            "view1": story_view.get("view1", ""),
            "view2": story_view.get("view2", ""),
            "view3": nested_data.get("view1", ""),
            "view4": nested_data.get("view2", "")
        }
        combined_data.append(combined_entry)

# Save combined data
output_combined = "generated_data/output_combined.jsonl"
with open(output_combined, 'w', encoding='utf-8') as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Combined {len(combined_data)} stories")
print(f"Saved to: {output_combined}")

# Create file in story_views format (single JSON object per line)
output_story_views_format = "generated_data/output_story_views_format.jsonl"
with open(output_story_views_format, 'w', encoding='utf-8') as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Saved story_views format to: {output_story_views_format}")
