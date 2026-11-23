# Customer Churn Predictor

This Python project ranks subscription customers by their chance of leaving in the next 30 days. It compares logistic regression and naive Bayes and simulates a retention team that can contact only a fraction of the customer base.

## Open the dashboard

Extract the project folder. With Python 3.10 or later installed, double-click `Start_Churn_Predictor.bat` on Windows to open the dashboard in your default browser. Keep the launcher window open while using it. On any platform with Python, run `python server.py --open` from the project folder. If the browser does not open automatically, use the local address printed in the terminal.

The dashboard opens with a demo analysis using 2,500 synthetic customers. To try your own data, choose **Upload a CSV**, select a labeled CSV file, set the contact budget, and click **Run analysis**. You can download an example CSV from the same panel. The dashboard runs on your computer and holds the uploaded data in memory while it trains and evaluates the models. The results page includes the contact list preview, the full holdout ranking CSV, and a Markdown report.

## Run from the command line

From the project folder, run the following with Python 3.10 or later. The project uses the standard library.

```powershell
python run.py
```

The run generates 2,500 fictional customers and writes a model comparison, holdout scores, and the synthetic input CSV to `artifacts/`. See [the sample evaluation](sample_report.md) for an example.

Change the contact budget or provide a labeled dataset with the same schema.

```powershell
python run.py --contact-share 0.15 --output artifacts/budget-15
python run.py --input artifacts/demo_customers.csv --output artifacts/csv-check
```

## Evaluation

1. Split the data by outcome into approximately 60% training, 20% validation, and 20% held-out testing. An individual customer appears in one partition.
2. Fit both models on training rows. Logistic regression learns feature scaling on those rows. Naive Bayes uses Gaussian likelihoods for numeric features and Bernoulli likelihoods for binary flags.
3. Choose the model with the higher validation average precision and evaluate its ranking on the holdout set. The contact list takes the top 10% of holdout scores by default, rounded up to a whole customer.

The report shows ROC AUC, average precision, precision and recall at the contact limit, and lift over the holdout churn rate. The sample data was designed to contain learnable patterns. Its results describe the simulation and cannot establish what works for real customers.

## Input schema

| Field | Example | Meaning |
| --- | --- | --- |
| `customer_id` | `C-000001` | Unique ID, used only to identify records and break score ties |
| `tenure_months` | `18` | Months since signup at the snapshot date |
| `active_days_last_30` | `9` | Days active in the 30 days before the snapshot |
| `monthly_fee` | `44.90` | Monthly subscription price |
| `support_tickets_last_30` | `2` | Tickets in the 30 days before the snapshot |
| `usage_change_pct` | `-28.5` | Usage change compared with the previous period |
| `autopay` | `1` | Automatic payment enabled |
| `monthly_plan` | `0` | Month-to-month plan |
| `churned_next_30_days` | `1` | Whether the customer left in the 30 days after the snapshot |

Use one row per customer, 0 or 1 for flags and outcome, and at least 10 records from each outcome class. Invalid rows are reported with their CSV line numbers. Keep real customer inputs and results outside the repository.

## Tests

```powershell
python -m unittest discover -s tests -v
```
