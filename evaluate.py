"""
Evaluation script for UZLLC Labor-Law QA supplementary materials.

This version assumes all repository files are stored at the repository root, for example:

    corpus_units.csv
    questions.csv
    gold_mapping.csv
    parameters.json
    bm25_results.csv
    lsa_results.csv
    hybrid_results.csv

Run:
    pip install pandas numpy scikit-learn
    python evaluate.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, Normalizer


ROOT = Path(__file__).resolve().parent


def normalize_text(text: str) -> str:
    """Normalize Uzbek legal/question text without changing legal meaning."""
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = {
        "ʼ": "'",
        "‘": "'",
        "’": "'",
        "`": "'",
        "´": "'",
        "ʻ": "'",
        "ʻ": "'",
        "ʼ": "'",
        "“": '"',
        "”": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z0-9_'ʻʼʻ\-]+", text)


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 2.0, b: float = 0.9):
        self.k1 = k1
        self.b = b
        self.docs_tokens = [tokenize(d) for d in documents]
        self.doc_lens = np.array([len(x) for x in self.docs_tokens], dtype=float)
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 0.0
        self.N = len(self.docs_tokens)
        self.df: Dict[str, int] = {}
        for toks in self.docs_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def score(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        scores = np.zeros(self.N, dtype=float)
        for i, doc in enumerate(self.docs_tokens):
            if not doc:
                continue
            tf: Dict[str, int] = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            dl = self.doc_lens[i]
            for t in q_terms:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    if float(np.max(scores)) == float(np.min(scores)):
        return np.zeros(len(scores), dtype=float)
    return MinMaxScaler().fit_transform(scores).ravel()


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    corpus_path = ROOT / "corpus_units.csv"
    questions_path = ROOT / "questions.csv"

    if not corpus_path.exists():
        raise FileNotFoundError("corpus_units.csv not found at repository root.")
    if not questions_path.exists():
        raise FileNotFoundError("questions.csv not found at repository root.")

    corpus = pd.read_csv(corpus_path)
    questions = pd.read_csv(questions_path)

    if "normalized_text" not in corpus.columns:
        corpus["normalized_text"] = corpus["raw_text"].map(normalize_text)
    if "gold_corpus_ids" not in questions.columns:
        raise ValueError("questions.csv must include gold_corpus_ids.")

    return corpus, questions


def gold_set(value: object) -> set:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}


def reciprocal_rank(ranked_ids: Sequence[str], gold: set, k: int = 10) -> float:
    if not gold:
        return 0.0
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], gold: set, k: int) -> int:
    if not gold:
        return 0
    return int(bool(set(ranked_ids[:k]) & gold))


def evaluate_rankings(rankings: Dict[str, List[str]], questions: pd.DataFrame, method: str) -> Dict[str, object]:
    q_eval = questions[(questions["split"] == "test") & (questions["answerability"] == "answerable")].copy()
    n = len(q_eval)

    r1 = r3 = r5 = 0
    mrr_sum = 0.0
    for _, row in q_eval.iterrows():
        qid = row["question_id"]
        ranked = rankings[qid]
        gold = gold_set(row["gold_corpus_ids"])
        r1 += recall_at_k(ranked, gold, 1)
        r3 += recall_at_k(ranked, gold, 3)
        r5 += recall_at_k(ranked, gold, 5)
        mrr_sum += reciprocal_rank(ranked, gold, 10)

    return {
        "method": method,
        "test_answerable_n": n,
        "recall@1": r1 / n,
        "recall@1_count": f"{r1}/{n}",
        "recall@3": r3 / n,
        "recall@3_count": f"{r3}/{n}",
        "recall@5": r5 / n,
        "recall@5_count": f"{r5}/{n}",
        "mrr@10": mrr_sum / n,
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin


def main() -> None:
    corpus, questions = load_data()

    corpus_ids = corpus["corpus_id"].astype(str).tolist()
    texts = (
        corpus.get("article_title", "").fillna("").astype(str)
        + " "
        + corpus["normalized_text"].fillna(corpus["raw_text"]).astype(str)
    ).map(normalize_text).tolist()

    test_questions = questions[(questions["split"] == "test")].copy()

    params = {}
    param_path = ROOT / "parameters.json"
    if param_path.exists():
        with open(param_path, "r", encoding="utf-8") as f:
            params = json.load(f)

    bm25_k1 = params.get("bm25", {}).get("k1", 2.0)
    bm25_b = params.get("bm25", {}).get("b", 0.9)
    lsa_components = params.get("lsa", {}).get("n_components", 200)
    alpha = params.get("hybrid", {}).get("alpha", 0.6)

    # BM25
    bm25 = BM25(texts, k1=bm25_k1, b=bm25_b)

    # TF-IDF + LSA
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        tokenizer=tokenize,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(texts)
    n_components = int(min(lsa_components, max(2, min(X.shape) - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    X_lsa = normalizer.fit_transform(svd.fit_transform(X))

    bm25_rankings: Dict[str, List[str]] = {}
    lsa_rankings: Dict[str, List[str]] = {}
    hybrid_rankings: Dict[str, List[str]] = {}

    bm25_rows = []
    lsa_rows = []
    hybrid_rows = []

    for _, row in test_questions.iterrows():
        qid = row["question_id"]
        qtext = row["question_text"]

        b_scores = bm25.score(qtext)

        q_vec = vectorizer.transform([normalize_text(qtext)])
        q_lsa = normalizer.transform(svd.transform(q_vec))
        s_scores = (X_lsa @ q_lsa.T).ravel()

        h_scores = alpha * minmax(b_scores) + (1 - alpha) * minmax(s_scores)

        for method, scores, rankings, out_rows in [
            ("bm25", b_scores, bm25_rankings, bm25_rows),
            ("lsa", s_scores, lsa_rankings, lsa_rows),
            ("hybrid", h_scores, hybrid_rankings, hybrid_rows),
        ]:
            order = np.argsort(-scores)
            ranked_ids = [corpus_ids[i] for i in order]
            rankings[qid] = ranked_ids

            gold = gold_set(row["gold_corpus_ids"])
            for rank, idx in enumerate(order[:10], start=1):
                cid = corpus_ids[idx]
                out_rows.append({
                    "question_id": qid,
                    "rank": rank,
                    "corpus_id": cid,
                    "score": float(scores[idx]),
                    "is_gold": int(cid in gold),
                })

    summary = pd.DataFrame([
        evaluate_rankings(bm25_rankings, questions, "BM25"),
        evaluate_rankings(lsa_rankings, questions, "TF-IDF + LSA"),
        evaluate_rankings(hybrid_rankings, questions, "BM25 + LSA hybrid"),
    ])
    summary.to_csv(ROOT / "retrieval_results_summary_recomputed.csv", index=False)

    pd.DataFrame(bm25_rows).to_csv(ROOT / "bm25_results_recomputed.csv", index=False)
    pd.DataFrame(lsa_rows).to_csv(ROOT / "lsa_results_recomputed.csv", index=False)
    pd.DataFrame(hybrid_rows).to_csv(ROOT / "hybrid_results_recomputed.csv", index=False)

    ci_rows = []
    for _, row in summary.iterrows():
        n = int(row["test_answerable_n"])
        for metric in ["recall@1", "recall@3", "recall@5"]:
            successes = int(str(row[f"{metric}_count"]).split("/")[0])
            lo, hi = wilson_ci(successes, n)
            ci_rows.append({
                "method": row["method"],
                "metric": metric,
                "successes": successes,
                "n": n,
                "estimate": row[metric],
                "ci95_low": lo,
                "ci95_high": hi,
            })
    pd.DataFrame(ci_rows).to_csv(ROOT / "confidence_intervals_recomputed.csv", index=False)

    print("Evaluation completed.")
    print(summary.to_string(index=False))
    print("\nGenerated:")
    print("- retrieval_results_summary_recomputed.csv")
    print("- bm25_results_recomputed.csv")
    print("- lsa_results_recomputed.csv")
    print("- hybrid_results_recomputed.csv")
    print("- confidence_intervals_recomputed.csv")


if __name__ == "__main__":
    main()
