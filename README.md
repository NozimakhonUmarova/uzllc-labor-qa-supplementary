# UZLLC Labor-Law QA Supplementary Materials

This repository contains supplementary materials for the paper:

**An Uzbek Labor-Law Corpus and Hybrid Semantic Retrieval Model for Source-Grounded Legal Question Answering**

## Overview

The materials support a source-grounded Uzbek legal information retrieval experiment focused on labor-law questions. The dataset contains corpus units from:

- the Labor Code of the Republic of Uzbekistan;
- Article 42 of the Constitution of the Republic of Uzbekistan.

The system evaluates three retrieval methods:

1. BM25 lexical retrieval;
2. TF-IDF + LSA latent semantic retrieval;
3. BM25 + LSA hybrid retrieval.

## Dataset summary

- Corpus retrieval units: 245
- Questions: 120
- Development split: 30 questions
- Test split: 90 questions
- Answerable test questions used for primary Recall/MRR evaluation: 78
- Unanswerable test questions retained for future abstention experiments: 12

## Repository structure

```text
data/
  corpus_units.csv
  corpus_units.jsonl
  questions.csv
  questions.jsonl
  gold_mapping.csv
  dev_test_split.csv

config/
  parameters.json

results/
  retrieval_results_summary.csv
  bm25_results.csv
  lsa_results.csv
  hybrid_results.csv
  confidence_intervals.csv
  paired_bootstrap_differences.csv
  error_examples.csv
  strict_all_test_metrics.csv
  bm25_dev_grid.csv
  lsa_dev_grid.csv
  hybrid_alpha_dev_grid.csv

docs/
  corpus_audit.md
  dataset_audit.md
  validation_report.md
  retrieval_evaluation_report.md

src/
  evaluate.py
```

## Main reported test results

Primary retrieval metrics are computed on the 78 answerable test questions.

| Method | Recall@1 | Count | Recall@3 | Count | Recall@5 | Count | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.603 | 47/78 | 0.808 | 63/78 | 0.897 | 70/78 | 0.718 |
| TF-IDF + LSA | 0.526 | 41/78 | 0.808 | 63/78 | 0.872 | 68/78 | 0.669 |
| BM25 + LSA hybrid | 0.603 | 47/78 | 0.821 | 64/78 | 0.872 | 68/78 | 0.724 |

## Selected parameters

The selected settings are stored in `config/parameters.json`.

Key settings:

- BM25: `k1 = 2.0`, `b = 0.9`
- TF-IDF: word-level, 1–2 grams, `min_df = 1`, `max_df = 0.95`, `sublinear_tf = true`
- Stopword handling: no stopword removal
- LSA components: 200
- Hybrid alpha: 0.6
- Development-set selection: MRR@10, then Recall@3 and Recall@1

## Reproducing the evaluation

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn
```

Run:

```bash
python src/evaluate.py
```

The script computes BM25, TF-IDF+LSA, and BM25+LSA hybrid retrieval metrics and writes result files.

## Important limitations

The gold mappings were checked through internal textual consistency validation. No independent legal-expert adjudication is included in this version. The dataset is intended for research on source-grounded retrieval and is not legal advice. Users should verify legal provisions using the official legal sources.

## Citation

If this supplementary material is used, please cite the accompanying UBMK paper.
