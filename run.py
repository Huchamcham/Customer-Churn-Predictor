"""Compare two churn models and export a holdout report."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from data import Customer, FEATURES, read_customers, sample_customers, write_customers
from evaluation import evaluate, rank_rows, split_rows
from models import LogisticRegression, NaiveBayes


def report(partitions, comparison, selected, test, share, demo, model):
    train, validation, holdout = partitions
    label = "Synthetic demo data" if demo else "User-supplied CSV"
    base_rate = sum(row.churned for row in holdout) / len(holdout)
    lines = [
        "# Customer churn evaluation", "", f"Dataset — {label}", "",
        f"Stratified split with seed 42 — {len(train):,} training, {len(validation):,} validation, "
        f"and {len(holdout):,} holdout customers.", "",
        "Both candidates were fitted on the training partition. The model with higher validation average precision was evaluated once on the holdout partition.",
        "", "| Validation model | ROC AUC | Average precision |", "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {name} | {row['auc']:.3f} | {row['ap']:.3f} |" for name, row in comparison.items())
    lines.extend([
        "", f"**Selected model — {selected}**", "",
        f"The holdout churn rate was {base_rate:.1%}. A contact limit of {share * 100:.6g}% put {test['contacted']} customers on the list.", "",
        "| Holdout measure | Result |", "| --- | ---: |",
        f"| ROC AUC | {test['auc']:.3f} |", f"| Average precision | {test['ap']:.3f} |",
        f"| Churners in the contact list | {test['captured']} |", f"| Precision at contact limit | {test['precision']:.1%} |",
        f"| Recall at contact limit | {test['recall']:.1%} |", f"| Lift over the holdout churn rate | {test['lift']:.2f}× |", "",
        "Precision is the share of the contact list that churned. Recall is the share of all holdout churners on that list. Lift compares precision with the holdout churn rate.", "",
    ])
    if isinstance(model, LogisticRegression):
        lines.extend(["## Logistic regression coefficients", "",
                      "Coefficients show the change in model log-odds for a one-standard-deviation increase in each input, holding the other inputs fixed.",
                      "", "| Input | Weight |", "| --- | ---: |"])
        terms = sorted(zip(FEATURES, model.weights), key=lambda entry: -abs(entry[1]))
        lines.extend(f"| `{key}` | {weight:+.3f} |" for key, weight in terms)
        lines.append("")
    lines.append("Scores rank customers for this example and have not been calibrated. A live evaluation would need time-ordered snapshots and evidence that outreach changes retention outcomes.")
    return "\n".join(lines) + "\n"


@dataclass
class Analysis:
    partitions: tuple[list[Customer], list[Customer], list[Customer]]
    comparison: dict
    selected: str
    model: LogisticRegression | NaiveBayes
    ranked: list[tuple[Customer, float]]
    test: dict

    def write_ranking(self, handle):
        writer = csv.writer(handle)
        writer.writerow(["rank", "customer_id", "churn_score", "actual_churn", "contact_list"])
        writer.writerows((rank, row.customer_id, f"{score:.9f}", row.churned, int(rank <= self.test["contacted"]))
                         for rank, (row, score) in enumerate(self.ranked, start=1))

    def markdown(self, share, demo):
        return report(self.partitions, self.comparison, self.selected, self.test, share, demo, self.model)


def analyze(rows, share):
    partitions = split_rows(rows)
    train, validation, holdout = partitions
    binary = [i for i, rule in enumerate(FEATURES.values()) if rule == (0, 1, True)]
    models = {"Logistic regression": LogisticRegression(), "Naive Bayes": NaiveBayes(binary)}
    comparison = {}
    for name, model in models.items():
        model.fit(train)
        comparison[name] = evaluate(rank_rows(model, validation), share)
    selected = max(comparison, key=lambda name: comparison[name]["ap"])
    model = models[selected]
    ranked = rank_rows(model, holdout)
    test = evaluate(ranked, share)
    return Analysis(partitions, comparison, selected, model, ranked, test)


def run(input_path, output, share):
    rows = read_customers(input_path) if input_path else sample_customers()
    result = analyze(rows, share)
    output.mkdir(parents=True, exist_ok=True)
    if input_path is None:
        write_customers(output / "demo_customers.csv", rows)
    (output / "report.md").write_text(
        result.markdown(share, input_path is None), encoding="utf-8"
    )
    with (output / "holdout_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        result.write_ranking(handle)
    print(f"Wrote evaluation report and holdout ranking to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Read labeled customer snapshots from a CSV")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory for generated results")
    parser.add_argument("--contact-share", type=float, default=0.10, help="Fraction of customers on the contact list")
    args = parser.parse_args()
    if not 0 < args.contact_share <= 1:
        parser.error("--contact-share must be greater than 0 and at most 1")
    try:
        run(args.input, args.output, args.contact_share)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
