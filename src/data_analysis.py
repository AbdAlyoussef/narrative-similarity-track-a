
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
TRAIN_PATH = DATA_DIR / "train_track_a.jsonl"
DEV_PATH = DATA_DIR / "dev_track_a.jsonl"
SAMPLE_PATH = DATA_DIR / "sample_track_a.jsonl"
OUTPUT_DIR = Path(__file__).parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# 1.  DATA LOADING

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_df(rows, source_label: str) -> pd.DataFrame:
    records = []
    for r in rows:
        records.append({
            "source": source_label,
            "anchor_text": r["anchor_text"],
            "text_a": r["text_a"],
            "text_b": r["text_b"],
            "text_a_is_closer": bool(r["text_a_is_closer"]),
            "model_name": r.get("model_name", None),
        })
    return pd.DataFrame(records)


train_rows = load_jsonl(TRAIN_PATH)
dev_rows = load_jsonl(DEV_PATH)
sample_rows = load_jsonl(SAMPLE_PATH)

df_train = rows_to_df(train_rows, "train")
df_dev = rows_to_df(dev_rows, "dev")
df_sample = rows_to_df(sample_rows, "sample")

df_all = pd.concat([df_train, df_dev, df_sample], ignore_index=True)


# 2.  TEXT HELPERS

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "who", "whom", "which", "what", "where", "when",
    "how", "not", "no", "nor", "as", "if", "then", "so", "up", "out",
    "about", "into", "over", "after", "before", "between", "under",
    "again", "further", "than", "too", "very", "just", "also", "more",
    "most", "other", "some", "such", "only", "own", "same", "all", "each",
    "every", "both", "few", "s", "t", "re", "ve", "d", "ll", "m",
}


def _safe(text) -> str:
    return str(text) if text else ""


def word_count(text) -> int:
    return len(_safe(text).split())


def char_count(text) -> int:
    return len(_safe(text))


def sentence_count(text) -> int:
    t = _safe(text).strip()
    if not t:
        return 0
    sents = SENT_SPLIT.split(t)
    return len([s for s in sents if s.strip()])


def avg_word_length(text) -> float:
    words = _safe(text).split()
    if not words:
        return 0.0
    return np.mean([len(w) for w in words])


def vocabulary(text) -> set:
    return set(
        w.lower().strip(string.punctuation)
        for w in _safe(text).split()
        if w.lower().strip(string.punctuation)
    )


def clean_tokens(text):
    """Lowercase, strip punctuation, remove stop words."""
    words = _safe(text).lower().split()
    return [
        w.strip(string.punctuation)
        for w in words
        if w.strip(string.punctuation) and w.strip(string.punctuation) not in STOP_WORDS
    ]


# 3.  BASIC STATISTICS

def compute_text_stats(series: pd.Series, label: str) -> pd.DataFrame:
    stats = {
        "word_count": series.apply(word_count),
        "char_count": series.apply(char_count),
        "sentence_count": series.apply(sentence_count),
        "avg_word_len": series.apply(avg_word_length),
    }
    stats_df = pd.DataFrame(stats)
    summary = stats_df.describe().T
    summary["field"] = label
    return stats_df, summary


print("=" * 72)
print("DATA SOURCES AND BACKGROUND")
print("=" * 72)
print("""
Dataset : SemEval-2026 Task 4 — Narrative Story Similarity, Track A
Task    : Given (anchor, story_A, story_B), predict which story is
          narratively closer to the anchor across three aspects:
          abstract theme, course of action, and outcomes.
Format  : JSONL — each line has keys:
            anchor_text, text_a, text_b, text_a_is_closer
            (train also has model_name for the generating LLM)

Training data (1 900 triples)
  • Source: LLM-generated synthetic narrative triples.
  • LLMs used: Meta-Llama-3.1-8B-Instruct-Turbo, GPT-4o, GPT-4o Mini,
    and others (see model_name field).
  • Obtained by prompting LLMs to produce anchor stories with two
    candidate variants differing in narrative similarity.

Dev data (200 triples)
  • Source: Real Wikipedia movie / book plot summaries.
  • Curated and labeled by the task organizers.

Sample data (39 triples)
  • Source: Same as dev — real Wikipedia summaries.
  • A small held-out set for sanity-checking predictions.

CRITICAL NOTE — Domain Gap:
  The synthetic training data differs significantly from the real
  dev/sample data in vocabulary, writing style, and text length.
  This is the central modeling challenge.
""")

print("=" * 72)
print("DATASET SIZE OVERVIEW")
print("=" * 72)
size_table = pd.DataFrame({
    "Split": ["Train", "Dev", "Sample"],
    "Triples": [len(df_train), len(df_dev), len(df_sample)],
    "Source": ["LLM-generated", "Wikipedia", "Wikipedia"],
    "Total texts": [len(df_train) * 3, len(df_dev) * 3, len(df_sample) * 3],
})
print(size_table.to_string(index=False))
print()

# Gather all texts per field per source
field_stats = {}
for source, df in [("train", df_train), ("dev", df_dev), ("sample", df_sample)]:
    for field in ["anchor_text", "text_a", "text_b"]:
        key = f"{source}_{field}"
        stats_df, summary = compute_text_stats(df[field], key)
        field_stats[key] = stats_df
        df[f"{field}_wc"] = stats_df["word_count"]
        df[f"{field}_cc"] = stats_df["char_count"]
        df[f"{field}_sc"] = stats_df["sentence_count"]
        df[f"{field}_awl"] = stats_df["avg_word_len"]

print("=" * 72)
print("TEXT STATISTICS PER FIELD (word count)")
print("=" * 72)
for source, df in [("train", df_train), ("dev", df_dev), ("sample", df_sample)]:
    print(f"\n--- {source.upper()} ({len(df)} triples) ---")
    for field in ["anchor_text", "text_a", "text_b"]:
        wc = df[f"{field}_wc"]
        print(f"  {field:15s}  mean={wc.mean():.1f}  med={wc.median():.0f}  "
              f"std={wc.std():.1f}  min={wc.min()}  max={wc.max()}")

print()
print("=" * 72)
print("TEXT STATISTICS PER FIELD (sentence count)")
print("=" * 72)
for source, df in [("train", df_train), ("dev", df_dev), ("sample", df_sample)]:
    print(f"\n--- {source.upper()} ({len(df)} triples) ---")
    for field in ["anchor_text", "text_a", "text_b"]:
        sc = df[f"{field}_sc"]
        print(f"  {field:15s}  mean={sc.mean():.1f}  med={sc.median():.0f}  "
              f"std={sc.std():.1f}  min={sc.min()}  max={sc.max()}")


# 4.  LABEL DISTRIBUTION

print()
print("=" * 72)
print("LABEL DISTRIBUTION (text_a_is_closer)")
print("=" * 72)
for source, df in [("train", df_train), ("dev", df_dev), ("sample", df_sample)]:
    counts = df["text_a_is_closer"].value_counts()
    total = len(df)
    a_closer = counts.get(True, 0)
    b_closer = counts.get(False, 0)
    print(f"  {source:8s}  A closer: {a_closer:4d} ({a_closer/total*100:.1f}%)  "
          f"B closer: {b_closer:4d} ({b_closer/total*100:.1f}%)")


# 5.  MOST COMMON WORDS

print()
print("=" * 72)
print("TOP 20 MOST COMMON CONTENT WORDS")
print("=" * 72)

for source, df in [("train", df_train), ("dev", df_dev)]:
    all_texts = (
        df["anchor_text"].tolist() + df["text_a"].tolist() + df["text_b"].tolist()
    )
    all_tokens = []
    for t in all_texts:
        all_tokens.extend(clean_tokens(t))
    freq = Counter(all_tokens).most_common(20)
    words_str = ", ".join(f"{w} ({c})" for w, c in freq)
    print(f"\n  {source.upper()}:  {words_str}")


# 6.  VOCABULARY OVERLAP  (synthetic vs real)

print()
print("=" * 72)
print("VOCABULARY OVERLAP: TRAIN (synthetic) vs DEV (real)")
print("=" * 72)

train_vocab = set()
for t in df_train["anchor_text"].tolist() + df_train["text_a"].tolist() + df_train["text_b"].tolist():
    train_vocab.update(vocabulary(t))

dev_vocab = set()
for t in df_dev["anchor_text"].tolist() + df_dev["text_a"].tolist() + df_dev["text_b"].tolist():
    dev_vocab.update(vocabulary(t))

overlap = train_vocab & dev_vocab
only_train = train_vocab - dev_vocab
only_dev = dev_vocab - train_vocab
print(f"  Train vocabulary size : {len(train_vocab):,}")
print(f"  Dev vocabulary size   : {len(dev_vocab):,}")
print(f"  Overlap               : {len(overlap):,} ({len(overlap)/len(dev_vocab)*100:.1f}% of dev)")
print(f"  Dev-only words        : {len(only_dev):,} ({len(only_dev)/len(dev_vocab)*100:.1f}% of dev)")
print(f"  Train-only words      : {len(only_train):,}")


# 7.  LLM SOURCE MODEL DISTRIBUTION (train only)

print()
print("=" * 72)
print("GENERATING LLM DISTRIBUTION (train set)")
print("=" * 72)
model_counts = df_train["model_name"].value_counts()
for model, cnt in model_counts.items():
    print(f"  {model:50s}  {cnt:4d} ({cnt/len(df_train)*100:.1f}%)")


# 8.  VISUALIZATIONS

print()
print("Generating visualizations ...")


# ---------- Figure 1: Label distribution ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Label Distribution (text_a_is_closer)", fontsize=14, fontweight="bold", y=1.02)

for ax, (name, df) in zip(axes, [("Train", df_train), ("Dev", df_dev), ("Sample", df_sample)]):
    counts = df["text_a_is_closer"].value_counts().sort_index()
    labels = ["B closer (False)", "A closer (True)"]
    values = [counts.get(False, 0), counts.get(True, 0)]
    colors = ["#e74c3c", "#27ae60"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", fontweight="bold", fontsize=11)
    ax.set_title(f"{name}  (n={len(df)})", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) * 1.18)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 01_label_distribution.png")


# ---------- Figure 2: Word count distributions ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Word Count Distributions by Text Field", fontsize=14, fontweight="bold", y=1.02)

for ax, field, color in zip(axes,
                             ["anchor_text", "text_a", "text_b"],
                             ["#3498db", "#e67e22", "#9b59b6"]):
    for source, df, ls in [("train", df_train, "-"), ("dev", df_dev, "--")]:
        data = df[f"{field}_wc"]
        ax.hist(data, bins=30, alpha=0.5, label=f"{source} (n={len(df)})",
                edgecolor="black", linewidth=0.5)
    ax.set_title(field, fontweight="bold")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_word_count_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 02_word_count_distributions.png")


# ---------- Figure 3: Sentence count distributions ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Sentence Count Distributions by Text Field", fontsize=14, fontweight="bold", y=1.02)

for ax, field in zip(axes, ["anchor_text", "text_a", "text_b"]):
    for source, df in [("train", df_train), ("dev", df_dev)]:
        data = df[f"{field}_sc"]
        ax.hist(data, bins=20, alpha=0.5, label=f"{source} (n={len(df)})",
                edgecolor="black", linewidth=0.5)
    ax.set_title(field, fontweight="bold")
    ax.set_xlabel("Sentence count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_sentence_count_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 03_sentence_count_distributions.png")


# ---------- Figure 4: Train vs Dev text length comparison (box plot) ----------
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Word Count Comparison: Train (Synthetic) vs Dev (Real)",
             fontsize=14, fontweight="bold")

plot_data = []
for source, df in [("Train", df_train), ("Dev", df_dev)]:
    for field in ["anchor_text", "text_a", "text_b"]:
        for val in df[f"{field}_wc"]:
            plot_data.append({"Source": source, "Field": field, "Word Count": val})

plot_df = pd.DataFrame(plot_data)
sns.boxplot(data=plot_df, x="Field", y="Word Count", hue="Source", ax=ax,
            palette={"Train": "#3498db", "Dev": "#e74c3c"})
ax.set_xlabel("")
ax.set_ylabel("Word Count")
ax.legend(title="Source")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_train_vs_dev_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 04_train_vs_dev_boxplot.png")


# ---------- Figure 5: Top 20 words comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Top 20 Content Words (stop words removed)",
             fontsize=14, fontweight="bold", y=1.02)

for ax, (source, df) in zip(axes, [("Train (Synthetic)", df_train), ("Dev (Real)", df_dev)]):
    all_texts = df["anchor_text"].tolist() + df["text_a"].tolist() + df["text_b"].tolist()
    all_tokens = []
    for t in all_texts:
        all_tokens.extend(clean_tokens(t))
    freq = Counter(all_tokens).most_common(20)
    words = [w for w, _ in freq]
    counts = [c for _, c in freq]
    ax.barh(words[::-1], counts[::-1], color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_title(source, fontweight="bold")
    ax.set_xlabel("Frequency")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "05_top_words_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05_top_words_comparison.png")


# ---------- Figure 6: LLM model distribution (train) ----------
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Generating LLM Distribution (Train Set)",
             fontsize=14, fontweight="bold")

model_counts = df_train["model_name"].value_counts()
short_names = [n.split("/")[-1] if "/" in str(n) else str(n) for n in model_counts.index]
colors = sns.color_palette("Set2", len(model_counts))
bars = ax.barh(short_names[::-1], model_counts.values[::-1],
               color=colors[::-1], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, model_counts.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontweight="bold", fontsize=10)
ax.set_xlabel("Number of triples")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "06_llm_model_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 06_llm_model_distribution.png")


# ---------- Figure 7: Vocabulary Venn-style bar chart ----------
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Vocabulary Overlap: Train vs Dev", fontsize=14, fontweight="bold")

categories = ["Train-only", "Shared", "Dev-only"]
sizes = [len(only_train), len(overlap), len(only_dev)]
colors_v = ["#3498db", "#2ecc71", "#e74c3c"]
bars = ax.bar(categories, sizes, color=colors_v, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}", ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Unique words")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "07_vocabulary_overlap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 07_vocabulary_overlap.png")


# ---------- Figure 8: Avg word length distribution ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Average Word Length Distributions", fontsize=14, fontweight="bold", y=1.02)

for ax, field in zip(axes, ["anchor_text", "text_a", "text_b"]):
    for source, df in [("train", df_train), ("dev", df_dev)]:
        data = df[f"{field}_awl"]
        ax.hist(data, bins=25, alpha=0.5, label=f"{source}",
                edgecolor="black", linewidth=0.5)
    ax.set_title(field, fontweight="bold")
    ax.set_xlabel("Avg word length (chars)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "08_avg_word_length.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 08_avg_word_length.png")


# ---------- Figure 9: Lexical overlap between closer vs farther story ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lexical Overlap with Anchor: Closer vs Farther Story",
             fontsize=14, fontweight="bold", y=1.02)

for ax, (source, df) in zip(axes, [("Train", df_train), ("Dev", df_dev)]):
    closer_overlaps = []
    farther_overlaps = []
    for _, row in df.iterrows():
        anchor_v = vocabulary(row["anchor_text"])
        a_v = vocabulary(row["text_a"])
        b_v = vocabulary(row["text_b"])
        if not anchor_v:
            continue
        overlap_a = len(anchor_v & a_v) / len(anchor_v)
        overlap_b = len(anchor_v & b_v) / len(anchor_v)
        if row["text_a_is_closer"]:
            closer_overlaps.append(overlap_a)
            farther_overlaps.append(overlap_b)
        else:
            closer_overlaps.append(overlap_b)
            farther_overlaps.append(overlap_a)

    ax.hist(closer_overlaps, bins=25, alpha=0.6, label="Closer story", color="#27ae60",
            edgecolor="black", linewidth=0.5)
    ax.hist(farther_overlaps, bins=25, alpha=0.6, label="Farther story", color="#e74c3c",
            edgecolor="black", linewidth=0.5)
    ax.set_title(f"{source}", fontweight="bold")
    ax.set_xlabel("Word overlap ratio with anchor")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.axvline(np.mean(closer_overlaps), color="#27ae60", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(farther_overlaps), color="#e74c3c", linestyle="--", linewidth=1.5)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "09_lexical_overlap_closer_vs_farther.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 09_lexical_overlap_closer_vs_farther.png")


# ---------- Figure 10: Combined summary dashboard ----------
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Narrative Similarity Track A — Data Analysis Dashboard",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

# (0,0) dataset sizes
ax = fig.add_subplot(gs[0, 0])
splits = ["Train\n(1900)", "Dev\n(200)", "Sample\n(39)"]
sizes_bar = [1900, 200, 39]
ax.bar(splits, sizes_bar, color=["#3498db", "#e74c3c", "#f39c12"],
       edgecolor="black", linewidth=0.8)
for i, v in enumerate(sizes_bar):
    ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
ax.set_title("Dataset Sizes", fontweight="bold")
ax.set_ylabel("Triples")

# (0,1) label balance
ax = fig.add_subplot(gs[0, 1])
for i, (name, df) in enumerate([("Train", df_train), ("Dev", df_dev), ("Sample", df_sample)]):
    a_pct = df["text_a_is_closer"].sum() / len(df) * 100
    b_pct = 100 - a_pct
    ax.barh(name, a_pct, color="#27ae60", edgecolor="black", linewidth=0.5, label="A closer" if i == 0 else "")
    ax.barh(name, -b_pct, color="#e74c3c", edgecolor="black", linewidth=0.5, label="B closer" if i == 0 else "")
ax.set_xlim(-60, 60)
ax.set_title("Label Balance (%)", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlabel("Percentage")

# (0,2) vocabulary overlap pie
ax = fig.add_subplot(gs[0, 2])
wedges, texts, autotexts = ax.pie(
    [len(only_train), len(overlap), len(only_dev)],
    labels=["Train-only", "Shared", "Dev-only"],
    colors=["#3498db", "#2ecc71", "#e74c3c"],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 9}
)
ax.set_title("Vocabulary Overlap", fontweight="bold")

# (1,0) word count box
ax = fig.add_subplot(gs[1, 0])
bp_data = []
for source, df in [("Train", df_train), ("Dev", df_dev)]:
    combined_wc = pd.concat([df[f"{f}_wc"] for f in ["anchor_text", "text_a", "text_b"]])
    bp_data.append(combined_wc.values)
bp = ax.boxplot(bp_data, labels=["Train", "Dev"], patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.6),
                medianprops=dict(color="black", linewidth=2))
bp["boxes"][1].set_facecolor("#e74c3c")
ax.set_title("Overall Word Count", fontweight="bold")
ax.set_ylabel("Words")

# (1,1) sentence count box
ax = fig.add_subplot(gs[1, 1])
bp_data = []
for source, df in [("Train", df_train), ("Dev", df_dev)]:
    combined_sc = pd.concat([df[f"{f}_sc"] for f in ["anchor_text", "text_a", "text_b"]])
    bp_data.append(combined_sc.values)
bp = ax.boxplot(bp_data, labels=["Train", "Dev"], patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.6),
                medianprops=dict(color="black", linewidth=2))
bp["boxes"][1].set_facecolor("#e74c3c")
ax.set_title("Overall Sentence Count", fontweight="bold")
ax.set_ylabel("Sentences")

# (1,2) avg word length
ax = fig.add_subplot(gs[1, 2])
for source, df, c in [("Train", df_train, "#3498db"), ("Dev", df_dev, "#e74c3c")]:
    all_awl = pd.concat([df[f"{f}_awl"] for f in ["anchor_text", "text_a", "text_b"]])
    ax.hist(all_awl, bins=30, alpha=0.5, label=source, color=c, edgecolor="black", linewidth=0.5)
ax.set_title("Avg Word Length", fontweight="bold")
ax.set_xlabel("Characters per word")
ax.legend(fontsize=9)

# (2,0-1) top words train
ax = fig.add_subplot(gs[2, 0])
all_tokens_train = []
for t in df_train["anchor_text"].tolist() + df_train["text_a"].tolist() + df_train["text_b"].tolist():
    all_tokens_train.extend(clean_tokens(t))
freq_train = Counter(all_tokens_train).most_common(15)
ax.barh([w for w, _ in freq_train][::-1], [c for _, c in freq_train][::-1],
        color="#3498db", edgecolor="black", linewidth=0.5)
ax.set_title("Top Words (Train)", fontweight="bold")

ax = fig.add_subplot(gs[2, 1])
all_tokens_dev = []
for t in df_dev["anchor_text"].tolist() + df_dev["text_a"].tolist() + df_dev["text_b"].tolist():
    all_tokens_dev.extend(clean_tokens(t))
freq_dev = Counter(all_tokens_dev).most_common(15)
ax.barh([w for w, _ in freq_dev][::-1], [c for _, c in freq_dev][::-1],
        color="#e74c3c", edgecolor="black", linewidth=0.5)
ax.set_title("Top Words (Dev)", fontweight="bold")

# (2,2) LLM model dist
ax = fig.add_subplot(gs[2, 2])
model_counts = df_train["model_name"].value_counts()
short_names = [n.split("/")[-1] if "/" in str(n) else str(n) for n in model_counts.index]
ax.pie(model_counts.values, labels=short_names, autopct="%1.1f%%",
       colors=sns.color_palette("Set2", len(model_counts)),
       textprops={"fontsize": 8})
ax.set_title("LLM Sources (Train)", fontweight="bold")

fig.savefig(OUTPUT_DIR / "10_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 10_dashboard.png")


# 9.  PREPROCESSING DOCUMENTATION

print()
print("=" * 72)
print("DATA PREPROCESSING STEPS (as used in this project)")
print("=" * 72)
print("""
The project applies the following preprocessing at various stages:

A. TOKENIZATION (src/train.py, src/predict.py)
   - Uses HuggingFace AutoTokenizer with the DeBERTa-v3-base tokenizer.
   - use_fast=False to avoid DebertaV2TokenizerFast issues across platforms.
   - max_length=384 tokens with truncation and dynamic padding.
   - Each anchor-candidate pair is prepended with a RUBRIC prompt:
       "Judge narrative similarity based on:
        1) Abstract Theme (ideas/motives),
        2) Course of Action (key events/turning points),
        3) Outcomes (results).
        Ignore names, writing style, and setting unless essential."
     This guides the cross-encoder to focus on the three narrative aspects.

B. ASPECT EXTRACTION (src/aspects.py)
   - Sentence splitting via regex: (?<=[.!?])\\s+
   - Theme  = first sentence of the text
   - Outcome = last sentence of the text
   - Action  = top-3 TF-IDF-ranked sentences from the middle
     (using sklearn TfidfVectorizer with English stop words removed,
      ranked by squared L2 norm of TF-IDF vectors)
   - Used to generate MiniLM cosine pseudo-targets, NOT as direct model input.

C. PSEUDO-TARGET GENERATION (src/train.py — MiniLMPseudoTargets)
   - Embeds aspect views with sentence-transformers/all-MiniLM-L6-v2.
   - Computes cosine similarity between anchor and candidate aspect views.
   - Maps cosine [-1, 1] to [0, 1] for MSE loss stability.
   - Embedding cache prevents redundant computation.

D. DATA FORMAT HANDLING (src/data.py)
   - Reads JSONL; skips lines with missing/None required keys.
   - TrackAPairwiseDataset: converts (anchor, A, B, label) to (anchor, pos, neg)
     based on the label — used for training with pairwise ranking loss.
   - TrackATriplesDataset: preserves original triple format — used for evaluation.

E. DATA AUGMENTATION (track-a-second-try/training/train.py)
   - Swap augmentation: randomly swaps text_a and text_b (flips label).
   - Random truncation: keeps 80-100% of words (simulates length variation).
   - Applied online during training with 50% probability.

F. NO ADDITIONAL TEXT CLEANING IS APPLIED:
   - No lowercasing, stemming, or lemmatization on model inputs.
   - The pretrained DeBERTa tokenizer handles subword segmentation (BPE).
   - Preserving original casing and morphology is important for the
     pretrained language model to leverage its learned representations.
""")


# 10. MERGED DATA — TOP WORDS BAR CHART + WORD CLOUD

from wordcloud import WordCloud

print()
print("=" * 72)
print("MERGED DATA: ALL SPLITS COMBINED (stop words removed)")
print("=" * 72)

# Merge every text field from every split into one token stream
all_merged_texts = []
for df in [df_train, df_dev, df_sample]:
    for field in ["anchor_text", "text_a", "text_b"]:
        all_merged_texts.extend(df[field].tolist())

all_merged_tokens = []
for t in all_merged_texts:
    all_merged_tokens.extend(clean_tokens(t))

merged_freq = Counter(all_merged_tokens)
top_30 = merged_freq.most_common(30)

print(f"\n  Total tokens (after stop-word removal): {len(all_merged_tokens):,}")
print(f"  Unique words: {len(merged_freq):,}")
print(f"\n  Top 30 words:")
for rank, (word, count) in enumerate(top_30, 1):
    print(f"    {rank:2d}. {word:20s}  {count:,}")

# ---------- Figure 11: Top 30 words (merged, bar chart) ----------
fig, ax = plt.subplots(figsize=(12, 9))
fig.suptitle("Top 30 Content Words — All Data Merged (stop words removed)",
             fontsize=14, fontweight="bold")

words_list = [w for w, _ in top_30]
counts_list = [c for _, c in top_30]

# Colour gradient from dark to light
cmap = plt.cm.Blues
norm_vals = np.linspace(0.35, 0.95, len(top_30))[::-1]
bar_colors = [cmap(v) for v in norm_vals]

bars = ax.barh(words_list[::-1], counts_list[::-1],
               color=bar_colors[::-1], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, counts_list[::-1]):
    ax.text(bar.get_width() + max(counts_list) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Frequency", fontsize=12)
ax.set_title(f"({len(all_merged_tokens):,} total tokens from {len(all_merged_texts):,} text segments)",
             fontsize=10, style="italic")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "11_top30_words_merged.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 11_top30_words_merged.png")

# ---------- Figure 12: Word Cloud (merged) ----------
fig, axes = plt.subplots(1, 3, figsize=(26, 10))
fig.suptitle("Word Clouds — All Data Merged (stop words removed)",
             fontsize=16, fontweight="bold", y=1.02)

# (a) Full merged word cloud
wc_all = WordCloud(
    width=800, height=800,
    background_color="white",
    max_words=150,
    colormap="viridis",
    prefer_horizontal=0.7,
    min_font_size=8,
    random_state=42,
).generate_from_frequencies(merged_freq)

axes[0].imshow(wc_all, interpolation="bilinear")
axes[0].set_title("All Data Combined", fontsize=13, fontweight="bold")
axes[0].axis("off")

# (b) Train-only word cloud
train_tokens = []
for t in df_train["anchor_text"].tolist() + df_train["text_a"].tolist() + df_train["text_b"].tolist():
    train_tokens.extend(clean_tokens(t))
train_freq = Counter(train_tokens)

wc_train = WordCloud(
    width=800, height=800,
    background_color="white",
    max_words=150,
    colormap="Blues",
    prefer_horizontal=0.7,
    min_font_size=8,
    random_state=42,
).generate_from_frequencies(train_freq)

axes[1].imshow(wc_train, interpolation="bilinear")
axes[1].set_title("Train (Synthetic)", fontsize=13, fontweight="bold")
axes[1].axis("off")

# (c) Dev-only word cloud
dev_tokens = []
for t in df_dev["anchor_text"].tolist() + df_dev["text_a"].tolist() + df_dev["text_b"].tolist():
    dev_tokens.extend(clean_tokens(t))
dev_freq = Counter(dev_tokens)

wc_dev = WordCloud(
    width=800, height=800,
    background_color="white",
    max_words=150,
    colormap="Reds",
    prefer_horizontal=0.7,
    min_font_size=8,
    random_state=42,
).generate_from_frequencies(dev_freq)

axes[2].imshow(wc_dev, interpolation="bilinear")
axes[2].set_title("Dev (Real / Wikipedia)", fontsize=13, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "12_word_clouds.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 12_word_clouds.png")


print()
print("=" * 72)
print(f"All plots saved to: {OUTPUT_DIR.resolve()}")
print("=" * 72)
