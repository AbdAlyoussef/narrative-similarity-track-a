import argparse
from src.data import read_jsonl, REQUIRED_KEYS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(args.path)
    print(f"[OK] Loaded {len(rows)} rows from {args.path}")
    print(f"[Info] Required keys: {REQUIRED_KEYS}")
    print("[Preview]")
    for i in range(min(args.n, len(rows))):
        r = rows[i]
        print(f"Row {i+1} keys: {list(r.keys())}")
        print(f"  anchor_text chars: {len(r['anchor_text'])}")
        print(f"  text_a chars: {len(r['text_a'])}")
        print(f"  text_b chars: {len(r['text_b'])}")
        print(f"  text_a_is_closer: {r['text_a_is_closer']}")
        print("")

if __name__ == "__main__":
    main()
