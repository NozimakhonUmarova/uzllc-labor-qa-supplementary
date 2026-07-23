# UZLLC Retrieval Evaluation Report v1

## Dataset

- Corpus units: 245

- Development questions: 30 total, 27 answerable

- Test questions: 90 total, 78 answerable and 12 unanswerable

- Primary retrieval metrics are calculated on answerable test questions only, because BM25/LSA/hybrid ranking does not include an abstention threshold for unanswerable queries.


## Selected parameters

- BM25: k1=2.0, b=0.9

- TF-IDF: word-level, ngram_range=(1,2), min_df=1, max_df=0.95, sublinear_tf=True, no stopword removal

- LSA components: 200; explained variance sum=0.9042

- Hybrid alpha: 0.6; selected on dev set by MRR@10


## Test results on answerable questions

| method          |   test_questions_total |   answerable_test_questions |   unanswerable_test_questions |   Recall@1 | Recall@1_count   |   Recall@3 | Recall@3_count   |   Recall@5 | Recall@5_count   |   MRR@10 |   RR_sum |
|:----------------|-----------------------:|----------------------------:|------------------------------:|-----------:|:-----------------|-----------:|:-----------------|-----------:|:-----------------|---------:|---------:|
| BM25            |                     90 |                          78 |                            12 |   0.602564 | 47/78            |   0.807692 | 63/78            |   0.897436 | 70/78            | 0.718407 |  56.0357 |
| TF-IDF+LSA      |                     90 |                          78 |                            12 |   0.525641 | 41/78            |   0.807692 | 63/78            |   0.871795 | 68/78            | 0.66859  |  52.15   |
| BM25+LSA hybrid |                     90 |                          78 |                            12 |   0.602564 | 47/78            |   0.820513 | 64/78            |   0.871795 | 68/78            | 0.724161 |  56.4845 |


## 95% confidence intervals

| method          | metric   |   estimate | count   |   ci_95_low |   ci_95_high | ci_method                                                |
|:----------------|:---------|-----------:|:--------|------------:|-------------:|:---------------------------------------------------------|
| BM25            | Recall@1 |   0.602564 | 47/78   |    0.491617 |     0.703883 | Wilson score interval                                    |
| BM25            | Recall@3 |   0.807692 | 63/78   |    0.70665  |     0.879849 | Wilson score interval                                    |
| BM25            | Recall@5 |   0.897436 | 70/78   |    0.810454 |     0.947107 | Wilson score interval                                    |
| BM25            | MRR@10   |   0.718407 |         |    0.635836 |     0.798935 | Nonparametric bootstrap over questions, 10,000 resamples |
| TF-IDF+LSA      | Recall@1 |   0.525641 | 41/78   |    0.416246 |     0.632629 | Wilson score interval                                    |
| TF-IDF+LSA      | Recall@3 |   0.807692 | 63/78   |    0.70665  |     0.879849 | Wilson score interval                                    |
| TF-IDF+LSA      | Recall@5 |   0.871795 | 68/78   |    0.779839 |     0.928848 | Wilson score interval                                    |
| TF-IDF+LSA      | MRR@10   |   0.66859  |         |    0.584615 |     0.751496 | Nonparametric bootstrap over questions, 10,000 resamples |
| BM25+LSA hybrid | Recall@1 |   0.602564 | 47/78   |    0.491617 |     0.703883 | Wilson score interval                                    |
| BM25+LSA hybrid | Recall@3 |   0.820513 | 64/78   |    0.720974 |     0.889962 | Wilson score interval                                    |
| BM25+LSA hybrid | Recall@5 |   0.871795 | 68/78   |    0.779839 |     0.928848 | Wilson score interval                                    |
| BM25+LSA hybrid | MRR@10   |   0.724161 |         |    0.642536 |     0.804488 | Nonparametric bootstrap over questions, 10,000 resamples |


## Paired bootstrap differences

| comparison             | metric   |   difference |   ci_95_low |   ci_95_high |   bootstrap_p_approx |
|:-----------------------|:---------|-------------:|------------:|-------------:|---------------------:|
| BM25+LSA hybrid - BM25 | Recall@1 |   0          |  -0.0512821 |    0.0512821 |               1      |
| BM25+LSA hybrid - BM25 | Recall@3 |   0.0128205  |  -0.0384615 |    0.0641026 |               0.8072 |
| BM25+LSA hybrid - BM25 | Recall@5 |  -0.025641   |  -0.0641026 |    0         |               0.2774 |
| BM25+LSA hybrid - BM25 | MRR@10   |   0.00575397 |  -0.0223451 |    0.0342953 |               0.6794 |


## Strict all-test metrics including unanswerable as misses

| method          |   test_questions_total | note                                                                                  |   Strict_R@1_all_test | Strict_R@1_count   |   Strict_R@3_all_test | Strict_R@3_count   |   Strict_R@5_all_test | Strict_R@5_count   |   Strict_MRR_all_test |
|:----------------|-----------------------:|:--------------------------------------------------------------------------------------|----------------------:|:-------------------|----------------------:|:-------------------|----------------------:|:-------------------|----------------------:|
| BM25            |                     90 | Unanswerable questions treated as misses because no abstention threshold was applied. |              0.522222 | 47/90              |              0.7      | 63/90              |              0.777778 | 70/90              |              0.622619 |
| TF-IDF+LSA      |                     90 | Unanswerable questions treated as misses because no abstention threshold was applied. |              0.455556 | 41/90              |              0.7      | 63/90              |              0.755556 | 68/90              |              0.579444 |
| BM25+LSA hybrid |                     90 | Unanswerable questions treated as misses because no abstention threshold was applied. |              0.522222 | 47/90              |              0.711111 | 64/90              |              0.755556 | 68/90              |              0.627606 |


## Notes

- Exact counts are included to avoid overinterpreting small percentage differences.

- Unanswerable questions should be evaluated with a separate abstention or answerability threshold in a future experiment.
