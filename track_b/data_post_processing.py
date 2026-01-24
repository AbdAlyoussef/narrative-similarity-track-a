import re
import json

def remove_pattern_from_jsonl(input_file):
    """
    Reads a JSONL file, removes lines matching the pattern "Okay, (*) \n\n",
    and saves the cleaned data to a new file with '_postProcessed' suffix.
    
    Args:
        input_file: Path to the input JSONL file
    """
    # Pattern to match "Here's a rewritten narrative..." at the start
    pattern2 = r'Here.*?rewritten.*?\n\n'
    # Pattern to match the word "rewritten" followed by \n\n
    pattern3 = r'\brewritten\b.*?\n\n'
    
    # Create output filename
    output_file = input_file.replace('.jsonl', '_postProcessed.jsonl')
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line)
                
                # Check each field in the JSON object
                for key, value in data.items():
                    if isinstance(value, str):
                        # First apply pattern2 (Here's...)
                        matches2 = re.findall(pattern2, value, re.IGNORECASE | re.MULTILINE)
                        if matches2:
                            print(f"Line {line_num}, Field '{key}':")
                            for match in matches2:
                                print(f"  Removed (pattern2): {repr(match)}")
                        value = re.sub(pattern2, "", value, flags=re.IGNORECASE | re.MULTILINE)
                        
                        # Then apply pattern3
                        matches = re.findall(pattern3, value, re.IGNORECASE)
                        if matches:
                            print(f"Line {line_num}, Field '{key}':")
                            for match in matches:
                                print(f"  Removed: {repr(match)}")
                        value = re.sub(pattern3, "", value, flags=re.IGNORECASE)
                        
                        data[key] = value
                
                # Write the cleaned JSON to output file
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

# Usage
remove_pattern_from_jsonl('generated_data/story_views_local_postProcessed.jsonl')