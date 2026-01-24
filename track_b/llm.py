import ollama
import pandas as pd
import json
import time


model="llama3.1:8b"
OUTPUT_JSONL = "output.jsonl"
id=0
df_dev = pd.read_json("dev_track_b.jsonl", lines=True).dropna(how="all")
df_sample = pd.read_json("sample_track_b.jsonl", lines=True).dropna(how="all")
df=pd.concat([df_dev,df_sample])


SYSTEM_PROMPT = (
    "You are a precise assistant. "
    "Return strictly valid JSON. No extra text."
)

prompt= f"""generate a different stories that are sim
ilar to the anchor story ,that I give you,in terms of main idea of the story
, the sequence of actions and the final outcomes and change anything else Re
turn only JSON exactly like: "generated_story": "the story that you have generated" 
Return only a JSON object. No prose. No Markdown. No code fences. donot add title for the story
"""

def parse_model_json(content: str) -> dict:
    s = content.strip()

    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()

    # First try strict JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # If it looks like an object body:  "key": ...
    # Example: '"view": "..."'  or '"a":1, "b":2'
    if s.startswith('"') and '":' in s:
        candidate = "{" + s
        if not candidate.rstrip().endswith("}"):
            candidate = candidate.rstrip() + "}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # keep going to other salvage options
            pass

    # Try to find a JSON object embedded somewhere in the text
    decoder = json.JSONDecoder()
    brace = s.find("{")
    if brace != -1:
        try:
            obj, _ = decoder.raw_decode(s[brace:])
            return obj
        except Exception:
            pass

    # No JSON recovered: return a structured error with raw output for traceability
    return {
        "error": "model_returned_non_json",
        "raw": s[:2000],  # cap to avoid huge logs
    }

def call_llama(prompt: str) -> dict:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.4},
    )
    content = response["message"]["content"]
    print("RAW MODEL OUTPUT:")
    print(repr(content))
    return parse_model_json(content) 


with open(OUTPUT_JSONL, "a", encoding="utf-8") as out:
    
        for idx, row in df[30:50].iterrows():
            text = str(row.get("text", "")).strip()
            result1=None
            result2=None
            try:
                
                result1=call_llama(text+prompt)
                result2=call_llama(text+prompt)
            except Exception as e:
                err = {"error": str(e)}
                if result1 is None:
                    result1 = err
                if result2 is None:
                    result2 = err
            record1 = {
                 
                id:result1,
                
            }
            id+=1
            record2 = {
                id:result2,
                
            }
            id+=1
            out.write(json.dumps(record1, ensure_ascii=False) + "\n")
            out.write(json.dumps(record2, ensure_ascii=False) + "\n")
            time.sleep(0.1)
        out.flush()

        
        time.sleep(0.0)

print("Done.")