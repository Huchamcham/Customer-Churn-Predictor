import tempfile
import unittest
from pathlib import Path

from data import Customer, read_customers, sample_customers, write_customers
from evaluation import evaluate, rank_rows, split_rows
from models import LogisticRegression, NaiveBayes


class ChurnTests(unittest.TestCase):
    def test_split_is_disjoint_and_keeps_both_classes(self):
        rows = sample_customers(200)
        train, validation, holdout = split_rows(rows)
        self.assertEqual(len(train) + len(validation) + len(holdout), 200)
        self.assertEqual(len(set(train + validation + holdout)), 200)
        for partition in (train, validation, holdout):
            self.assertEqual({row.churned for row in partition}, {0, 1})
        self.assertEqual((train, validation, holdout), split_rows(rows))

    def test_metrics_for_known_ranking(self):
        labels = [1, 0, 1, 0]
        ranked = [(Customer(str(i), (), label), 1 - i / 4) for i, label in enumerate(labels)]
        values = evaluate(ranked, 0.25)
        self.assertAlmostEqual(values["auc"], 0.75)
        self.assertAlmostEqual(values["ap"], 5 / 6)
        self.assertAlmostEqual(values["lift"], 2)
        self.assertAlmostEqual(values["recall"], 0.5)

    def test_metrics_for_ties(self):
        ranked = [(Customer(str(i), (), label), 0.5) for i, label in enumerate([1, 0, 0, 0])]
        values = evaluate(ranked, 0.25)
        self.assertEqual(values["auc"], 0.5)
        self.assertEqual(values["ap"], 0.25)

    def test_csv_round_trip_and_duplicate_check(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "customers.csv"
            rows = sample_customers(4)
            write_customers(path, rows)
            self.assertEqual(read_customers(path), rows)
            write_customers(path, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate customer ID"):
                read_customers(path)

    def test_models_rank_separable_examples_with_constant_input(self):
        rows = [Customer(f"C-{i}", (i, 0), int(i >= 10)) for i in range(20)]
        for model in (LogisticRegression(), NaiveBayes([1])):
            model.fit(rows)
            with self.subTest(model=type(model).__name__):
                self.assertEqual(evaluate(rank_rows(model, rows), 0.1)["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
