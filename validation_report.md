# UZLLC Question Dataset Validation Report v1
## Scope
This file reports a model-assisted internal consistency validation of the 120-question dataset against the supplied Labor Code and Constitution corpus units. It does not claim independent legal-expert validation. In the article, this should be described as manual/internal consistency validation, not as legal-expert annotation.
## Summary
- total_questions: 120
- answerable_questions: 105
- unanswerable_questions: 15
- dev_questions: 30
- test_questions: 90
- gold_mapping_rows: 120
- unique_gold_corpus_units: 95
- legal_expert_validated: False
- validation_scope: Internal consistency validation against the supplied cleaned Labor Code and Constitution corpus units; not a substitute for independent legal-expert annotation.

## Question type distribution
- terminology: 45
- paraphrase: 45
- multi_gold: 15
- unanswerable: 15

## Split distribution
| split   |   multi_gold |   paraphrase |   terminology |   unanswerable |
|:--------|-------------:|-------------:|--------------:|---------------:|
| dev     |            4 |           11 |            12 |              3 |
| test    |           11 |           34 |            33 |             12 |

## Validation decisions
- validated_consistent: 120

## Gold mapping
- Normalized gold-mapping rows: 120
- Unique gold corpus units used: 95
- Multi-gold questions were kept with one primary and one secondary corpus unit.
- Unanswerable questions have no gold corpus unit and are intended for out-of-scope detection, not Recall@k calculation.

## Notes for paper wording
Recommended wording: "The gold mappings were manually checked for internal textual consistency against the selected Labor Code and Constitution corpus units. No independent legal-expert adjudication was available for this version; this limitation is reported explicitly."

## Files generated
- questions_v1_validated.csv
- questions_v1_validated.json
- questions_v1_validated.jsonl
- gold_mapping_v1_validated.csv
- dev_test_split_v1.csv
- validation_summary_v1.json
- validation_evidence_samples_v1.csv
