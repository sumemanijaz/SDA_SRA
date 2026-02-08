"""Preprocessing utilities for requirements text files.

This module reads plain text requirement documents or meeting notes and
produces a cleaned list of sentences ready for LLM analysis.

Primary functions:
- `read_text_file(path)` - read file contents
- `clean_text(text, ...)` - remove headers/page numbers and normalize whitespace
- `split_sentences(text)` - split cleaned text into sentences
- `remove_empty_or_meaningless(sentences, ...)` - filter out short/empty items
- `preprocess_file(path, ...)` - orchestrator returning list of sentences

The implementation uses simple, explainable heuristics and no external
dependencies so it integrates easily with other modules in the project.
"""

from typing import List, Optional
import os
import re
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple

# reuse tokenizer from classifier to avoid duplicate tokenization logic
try:
    from . import classifier
    _tokenize = classifier.tokenize
except Exception:
    # fallback: very small local tokenizer
    _WORD_RE = re.compile(r"\b[a-zA-Z0-9\-]+\b")

    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in _WORD_RE.findall(text)]

__all__ = [
    "read_text_file",
    "clean_text",
    "split_sentences",
    "remove_empty_or_meaningless",
    "preprocess_file",
    "compute_sentence_vectors",
    "group_sentences_by_similarity",
]


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Read and return text content from `path`.

    Raises `FileNotFoundError` if the file does not exist.
    """
    with open(path, "r", encoding=encoding) as fh:
        return fh.read()


def clean_text(
    text: str,
    remove_headers: bool = True,
    remove_page_numbers: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """Return a cleaned version of `text`.

    Cleaning performed:
    - Remove obvious page number lines like "Page 1" or standalone digits
    - Drop simple headers (short lines with many uppercase characters)
    - Remove horizontal separators like '----' or '***'
    - Optionally normalize repeated whitespace and blank lines
    """
    lines = text.splitlines()
    cleaned_lines: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            # preserve empty lines (we'll collapse later) to keep paragraph boundaries
            cleaned_lines.append("")
            continue

        # horizontal separators
        if re.match(r"^[-*_]{3,}$", s):
            continue

        # common page number patterns: "Page 1" or "1" on its own line
        if remove_page_numbers and re.match(r"^page\s*\d+(\s*of\s*\d+)?$", s, re.I):
            continue
        if remove_page_numbers and re.match(r"^\d+$", s):
            continue

        # heuristic for headers: short lines with mostly uppercase letters
        if remove_headers:
            words = s.split()
            if 1 <= len(words) <= 6:
                uppercase_chars = sum(1 for ch in s if ch.isupper())
                upper_ratio = uppercase_chars / max(1, len(s))
                if upper_ratio > 0.6:
                    continue

        cleaned_lines.append(s)

    cleaned = "\n".join(cleaned_lines)

    if normalize_whitespace:
        # collapse multiple blank lines to a single blank line
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        # normalize internal whitespace to single spaces on lines
        cleaned = "\n".join(re.sub(r"\s+", " ", l).strip() for l in cleaned.splitlines())

    return cleaned.strip()


def split_sentences(text: str) -> List[str]:
    """Split `text` into sentences using a lightweight regex.

    This function is intentionally simple and avoids heavy NLP
    dependencies. It splits on sentence-ending punctuation followed by
    whitespace and a capital letter or digit, and also splits on
    line-breaks. The output sentences are stripped.
    """
    if not text:
        return []

    # replace newlines with spaces to avoid keeping line-based fragments
    single = text.replace("\n", " ")

    # split on punctuation (.!?), keep abbreviations handling minimal
    parts = re.split(r'(?<=[\.\!\?])\s+(?=[A-Z0-9\"\'\(])', single)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


def remove_empty_or_meaningless(
    sentences: List[str], min_chars: int = 10, min_words: int = 2
) -> List[str]:
    """Filter out sentences that are too short or otherwise meaningless.

    - `min_chars` excludes very short fragments
    - `min_words` excludes fragments with too few words
    """
    out: List[str] = []
    for s in sentences:
        if len(s) < min_chars:
            continue
        if len(s.split()) < min_words:
            continue
        out.append(s)
    return out


def preprocess_file(
    path: str,
    remove_headers: bool = True,
    remove_page_numbers: bool = True,
    normalize_whitespace: bool = True,
    filter_short: bool = True,
    min_chars: int = 10,
    min_words: int = 2,
) -> List[str]:
    """Read `path`, clean the text, split into sentences and return them.

    Typical usage from `app.py`:
    ```py
    sentences = preprocess_file("data/meeting_notes.txt")
    ```
    """
    raw = read_text_file(path)
    cleaned = clean_text(
        raw,
        remove_headers=remove_headers,
        remove_page_numbers=remove_page_numbers,
        normalize_whitespace=normalize_whitespace,
    )
    sentences = split_sentences(cleaned)
    if filter_short:
        sentences = remove_empty_or_meaningless(sentences, min_chars=min_chars, min_words=min_words)
    return sentences


def compute_sentence_vectors(sentences: List[str]) -> Tuple[List[Dict[int, float]], Dict[str, int]]:
    """Compute lightweight TF-IDF-like vectors for `sentences`.

    Returns a tuple (vectors, vocab) where `vectors` is a list of
    dicts mapping token-index to float weight and `vocab` maps token->index.

    This is intentionally simple and dependency-free; it provides a
    semantic grouping signal suitable for approximate similarity and
    downstream clustering or nearest-neighbour lookups.
    """
    # tokenize and collect document frequencies
    tokenized = [list(filter(None, _tokenize(s))) for s in sentences]
    df: Counter = Counter()
    for toks in tokenized:
        df.update(set(toks))

    # build vocabulary
    vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(df.keys()))}
    n = max(1, len(sentences))

    vectors: List[Dict[int, float]] = []
    for toks in tokenized:
        tf = Counter(toks)
        vec: Dict[int, float] = {}
        for term, freq in tf.items():
            idx = vocab.get(term)
            if idx is None:
                continue
            # tf * idf weighting
            idf = math.log((n / (1 + df[term])))
            vec[idx] = freq * idf
        # normalize vector length
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in list(vec.keys()):
                vec[k] = vec[k] / norm
        vectors.append(vec)

    return vectors, vocab


def _cosine_sim(a: Dict[int, float], b: Dict[int, float]) -> float:
    # efficient sparse dot product
    if not a or not b:
        return 0.0
    # iterate over smaller dict
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, v in a.items():
        s += v * b.get(k, 0.0)
    return s


def group_sentences_by_similarity(sentences: List[str], threshold: float = 0.65) -> List[List[int]]:
    """Group sentence indices by vector similarity.

    Returns a list of clusters, each cluster is a list of sentence
    indices (into the input `sentences` list). This algorithm is a
    simple greedy clustering using centroid similarity; it is designed
    to be fast and interpretable rather than perfectly optimal.
    """
    vectors, vocab = compute_sentence_vectors(sentences)
    clusters: List[Dict[str, object]] = []

    for idx, vec in enumerate(vectors):
        placed = False
        best_cluster = None
        best_sim = 0.0
        for c in clusters:
            centroid: Dict[int, float] = c["centroid"]
            sim = _cosine_sim(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c

        if best_cluster and best_sim >= threshold:
            # add to cluster and update centroid (simple average)
            members: List[int] = best_cluster["members"]
            m = len(members)
            # update centroid: new_centroid = (centroid * m + vec) / (m+1)
            centroid = best_cluster["centroid"]
            # combine sparse representations
            new_centroid: Dict[int, float] = dict(centroid)
            for k, v in vec.items():
                new_centroid[k] = new_centroid.get(k, 0.0) * m / (m + 1) + v / (m + 1)
            best_cluster["centroid"] = new_centroid
            best_cluster["members"].append(idx)
            placed = True

        if not placed:
            # create new cluster with this vector as centroid
            clusters.append({"centroid": dict(vec), "members": [idx]})

    # return list of member index lists
    return [c["members"] for c in clusters]


if __name__ == "__main__":
    # Simple demo for manual testing
    demo_text = (
        "PROJECT XYZ\n"  # header that should be removed
        "Page 1\n\n"
        "The system shall allow users to register and login using email and password.\n"
        "Responses to user queries must be returned within 2 seconds under normal load.\n\n"
        "1\n"
        "All sensitive data must be encrypted at rest and in transit.\n"
    )

    # write demo to a temporary file in data/
    demo_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(demo_dir, exist_ok=True)
    demo_path = os.path.join(demo_dir, "tmp_demo_requirements.txt")
    with open(demo_path, "w", encoding="utf-8") as fh:
        fh.write(demo_text)

    print("Preprocessing demo file:", demo_path)
    out = preprocess_file(demo_path)
    for i, s in enumerate(out, 1):
        print(i, s)
