import json

def transform_output_to_views_format(input_file, output_file):
    """
    Transform output.jsonl format to dev_track_b_views format.
    Every two consecutive entries become view1 and view2 of the same output.
    
    Input format: {"0": {"generated_story": "story text"}}
    Output format: {"story": "", "view1": "first story", "view2": "second story"}
    """
    
    def extract_story_text(data):
        """Extract story text from the nested structure"""
        story_text = ""
        
        if data:
            first_key = list(data.keys())[0]
            entry = data[first_key]
            
            if "generated_story" in entry:
                story_text = entry["generated_story"]
            elif "error" in entry and "raw" in entry:
                # Extract from raw field when there's an error
                raw_text = entry.get("raw", "")
                # Try to parse as JSON first
                try:
                    raw_data = json.loads(raw_text)
                    if "generated_story" in raw_data:
                        story_text = raw_data["generated_story"]
                    else:
                        # If no generated_story key, use the whole raw content
                        story_text = raw_text
                except:
                    # If parsing fails, use the raw string as is
                    story_text = raw_text
        
        return story_text
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        stories = []
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                story_text = extract_story_text(data)
                stories.append(story_text)
                
                # When we have two stories, create an output entry
                if len(stories) == 2:
                    output_data = {
                        "story": "",
                        "view1": stories[0],
                        "view2": stories[1]
                    }
                    
                    outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    stories = []  # Reset for next pair
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}... - {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
        
        # Handle remaining story if odd number of entries
        if len(stories) == 1:
            print(f"Warning: Odd number of entries. Last entry duplicated in both views.")
            output_data = {
                "story": "",
                "view1": stories[0],
                "view2": stories[0]
            }
            outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    
    print(f"Transformation complete! Output written to: {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = "generated_data/output.jsonl"
    output_file = "generated_data/output_transformed.jsonl"
    
    transform_output_to_views_format(input_file, output_file)
    
    print(f"\nTo transform different files, you can call:")
    print(f'transform_output_to_views_format("generated_data/output (3).jsonl", "generated_data/output_3_transformed.jsonl")')
