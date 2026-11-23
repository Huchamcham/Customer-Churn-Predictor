# Customer churn evaluation

Dataset — Synthetic demo data

Stratified split with seed 42 — 1,502 training, 499 validation, and 499 holdout customers.

Both candidates were fitted on the training partition. The model with higher validation average precision was evaluated once on the holdout partition.

| Validation model | ROC AUC | Average precision |
| --- | ---: | ---: |
| Logistic regression | 0.763 | 0.445 |
| Naive Bayes | 0.756 | 0.424 |

**Selected model — Logistic regression**

The holdout churn rate was 19.6%. A contact limit of 10% put 50 customers on the list.

| Holdout measure | Result |
| --- | ---: |
| ROC AUC | 0.768 |
| Average precision | 0.467 |
| Churners in the contact list | 32 |
| Precision at contact limit | 64.0% |
| Recall at contact limit | 32.7% |
| Lift over the holdout churn rate | 3.26× |

Precision is the share of the contact list that churned. Recall is the share of all holdout churners on that list. Lift compares precision with the holdout churn rate.

## Logistic regression coefficients

Coefficients show the change in model log-odds for a one-standard-deviation increase in each input, holding the other inputs fixed.

| Input | Weight |
| --- | ---: |
| `usage_change_pct` | -0.740 |
| `active_days_last_30` | -0.400 |
| `tenure_months` | -0.307 |
| `autopay` | -0.302 |
| `monthly_plan` | +0.283 |
| `monthly_fee` | +0.260 |
| `support_tickets_last_30` | +0.147 |

Scores rank customers for this example and have not been calibrated. A live evaluation would need time-ordered snapshots and evidence that outreach changes retention outcomes.
