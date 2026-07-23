# UZLLC Labor + Constitution Corpus Units v1

## Summary
- Total retrieval units: 245
- Document counts: {'CON': 3, 'LC': 242}
- Unit level counts: {'clause': 60, 'article': 185}
- LC article coverage: 197 articles
- CON article coverage: Article 42 only (3 clause units)

## Selected LC article ranges
- 2–5: labor rights and principles
- 10–11: scope of labor legislation
- 19–25: employment subjects, employee/employer rights and duties
- 103–173: employment contract, hiring, probation, modification, suspension, termination
- 181–250, 253–254, 269–270: working time, rest time, leave, wages, wage payment, final settlement, deductions
- 312–315: disciplinary sanctions and procedure
- 351–366: occupational safety and health
- 392–410: pregnancy, childcare, and family-responsibility guarantees
- Constitution Article 42: constitutional right to work and labor guarantees

## Topic distribution
- termination: 46
- leave_entitlement: 30
- working_time: 20
- family_pregnancy_childcare_guarantees: 20
- wages_and_payments: 19
- contract_modification_transfer: 18
- rest_time_holidays: 17
- occupational_safety_health: 16
- employment_contract_general: 15
- hiring_procedure: 11
- probation_period: 8
- employment_subjects_rights_duties: 7
- disciplinary_liability: 5
- labor_rights_principles: 4
- suspension_from_work: 4
- constitutional_labor_rights: 3
- labor_law_scope: 2

## Schema
- `corpus_id`
- `document_code`
- `document_title`
- `article_number`
- `article_title`
- `unit_level`
- `unit_number`
- `raw_text`
- `normalized_text`
- `legal_topics`
- `source_url`
- `answer_source`

## Notes
- `raw_text` preserves the cleaned source text from the uploaded files.
- `normalized_text` lowercases text, normalizes repeated spaces, and unifies apostrophe-like characters for retrieval.
- For the Constitution, the official source text does not provide article titles; therefore `article_title` is null for Article 42 units.
- The subset is designed for the planned 120-question controlled benchmark.