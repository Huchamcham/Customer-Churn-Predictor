"""Create stratified partitions and score a model's customer ranking."""

import math
import random
from itertools import groupby


def split_rows(rows, seed=42):
    rng = random.Random(seed)
    partitions = ([], [], [])
    for label in (0, 1):
        group = [row for row in rows if row.churned == label]
        if len(group) < 10:
            raise ValueError("Use at least 10 customers from each outcome class.")
        rng.shuffle(group)
        held_out = len(group) // 5
        train_end = len(group) - 2 * held_out
        partitions[0].extend(group[:train_end])
        partitions[1].extend(group[train_end:train_end + held_out])
        partitions[2].extend(group[train_end + held_out:])
    for partition in partitions:
        rng.shuffle(partition)
    return partitions


def rank_rows(model, rows):
    return sorted(((row, model.predict(row.features)) for row in rows), key=lambda entry: (-entry[1], entry[0].customer_id))


def evaluate(ranked, contact_share):
    total = len(ranked)
    positive = sum(row.churned for row, _ in ranked)
    negative = total - positive
    seen_positive = seen_negative = favorable_pairs_twice = 0
    ap = 0.0

    for _, group in groupby(ranked, key=lambda entry: entry[1]):
        labels = [row.churned for row, _ in group]
        positives = sum(labels)
        negatives = len(labels) - positives
        seen_positive += positives
        seen_negative += negatives
        ap += positives / positive * seen_positive / (seen_positive + seen_negative)
        # Each negative is below the previous positives and tied with this group.
        favorable_pairs_twice += negatives * (2 * seen_positive - positives)

    contacted = math.ceil(total * contact_share)
    captured = sum(row.churned for row, _ in ranked[:contacted])
    precision = captured / contacted
    return {
        "auc": favorable_pairs_twice / (2 * positive * negative),
        "ap": ap, "contacted": contacted, "captured": captured,
        "precision": precision, "recall": captured / positive, "lift": precision / (positive / total),
    }
