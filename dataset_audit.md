# UZLLC Question Dataset v1 Audit

This package contains a 120-question evaluation set linked to the UZLLC Labor Code + Constitution retrieval units.

## Files
- `questions_v1.csv` — spreadsheet-friendly question dataset.
- `questions_v1.json` — JSON version of the same dataset.
- `questions_v1.jsonl` — line-delimited JSON version.
- `gold_mapping_v1.csv` — normalized many-to-many mapping from questions to gold `corpus_id` values.

## Counts
- Total questions: 120
- Answerable questions: 105
- Unanswerable / out-of-scope questions: 15
- Gold mapping rows: 120
- Unique gold corpus units used: 95
- Split: dev=30, test=90

## Question-type distribution
- terminology: 45
- paraphrase: 45
- multi_gold: 15
- unanswerable: 15

## Topic distribution
- termination: 12
- leave_entitlement: 12
- wages_and_payments: 11
- employment_contract_general: 8
- rest_time_holidays: 8
- occupational_safety_health: 8
- employment_subjects_rights_duties: 6
- probation_period: 6
- working_time: 6
- family_pregnancy_childcare_guarantees: 6
- constitutional_labor_rights: 4
- labor_rights_principles: 4
- hiring_procedure: 4
- disciplinary_liability: 4
- contract_modification_transfer: 3
- labor_law_scope: 2
- fact_specific_legal_advice: 2
- suspension_from_work: 1
- out_of_scope_tax_law: 1
- out_of_scope_family_law: 1
- out_of_scope_housing_law: 1
- out_of_scope_customs_law: 1
- out_of_scope_administrative_law: 1
- out_of_scope_criminal_procedure: 1
- fact_specific_financial_civil_issue: 1
- foreign_jurisdiction: 1
- future_current_rate: 1
- legal_document_drafting: 1
- out_of_scope_property_law: 1
- out_of_scope_tax_employment_mixed: 1
- outside_corpus_social_insurance: 1

## Notes
- `gold_corpus_ids` in CSV is pipe-separated.
- Unanswerable questions intentionally have an empty `gold_corpus_ids` field and are excluded from retrieval-recall metrics unless an answerability-detection metric is added.
- Multi-gold questions have two gold units in `gold_mapping_v1.csv`.
- Dev/test split is fixed: 30 development questions and 90 held-out test questions.
