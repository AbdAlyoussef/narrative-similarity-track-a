import re
from sklearn.feature_extraction.text import TfidfVectorizer

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]

def _topk_tfidf_sentences(sentences, k=3):
    if len(sentences) <= k:
        return sentences

    # TF-IDF over sentences; select by L2 norm as "importance"
    vec = TfidfVectorizer(min_df=1, stop_words="english")
    X = vec.fit_transform(sentences)  # [n_sent, vocab]
    scores = (X.multiply(X)).sum(axis=1).A1  # squared L2 norm

    top_idx = scores.argsort()[::-1][:k]
    top_idx = sorted(top_idx.tolist())  # keep narrative order
    return [sentences[i] for i in top_idx]

def aspect_views(text: str):
    """
    Theme: first sentence
    Outcome: last sentence
    Action: top-3 TF-IDF sentences from the middle (excluding first/last)
    """
    sents = split_sentences(text)
    if len(sents) == 0:
        return ("", "", "")
    if len(sents) == 1:
        return (sents[0], sents[0], sents[0])

    theme = sents[0]
    outcome = sents[-1]

    middle = sents[1:-1]
    if len(middle) >= 1:
        action_sents = _topk_tfidf_sentences(middle, k=min(3, len(middle)))
        action = " ".join(action_sents)
    else:
        action = theme

    return (theme, action, outcome)