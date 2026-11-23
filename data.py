"""Generate and validate customer snapshots for the churn demo."""

import csv
import math
import random
import re
from dataclasses import dataclass


FEATURES = {
    "tenure_months": (1, 240, True),
    "active_days_last_30": (0, 30, True),
    "monthly_fee": (0, 10000, False),
    "support_tickets_last_30": (0, 100, True),
    "usage_change_pct": (-100, 1000, False),
    "autopay": (0, 1, True),
    "monthly_plan": (0, 1, True),
}
TARGET = "churned_next_30_days"


@dataclass(frozen=True)
class Customer:
    customer_id: str
    features: tuple[float, ...]
    churned: int


def sample_customers(count=2500, seed=42):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        tenure = rng.randint(1, 84)
        active_days = max(0, min(30, round(rng.gauss(17, 7))))
        fee = round(rng.uniform(19, 99), 2)
        support = min(7, int(rng.expovariate(0.9)))
        usage_change = round(max(-100, min(150, rng.gauss(-8, 30))), 2)
        autopay = int(rng.random() < 0.57)
        monthly = int(rng.random() < 0.68)
        risk = (-1.55 - 0.018 * tenure - 0.058 * (active_days - 17) + 0.014 * (fee - 59)
                + 0.34 * support - 0.023 * usage_change - 0.60 * autopay + 0.55 * monthly
                + (0.72 if monthly and usage_change < -35 else 0) + rng.gauss(0, 0.45))
        churned = int(rng.random() < 1 / (1 + math.exp(-risk)))
        features = (tenure, active_days, fee, support, usage_change, autopay, monthly)
        rows.append(Customer(f"C-{i + 1:06d}", features, churned))
    return rows


def write_customers(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        write_customer_csv(handle, rows)


def write_customer_csv(handle, rows):
    writer = csv.writer(handle)
    writer.writerow(["customer_id", *FEATURES, TARGET])
    writer.writerows((row.customer_id, *row.features, row.churned) for row in rows)


def read_customers(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return parse_customers(handle)


def parse_customers(handle, max_rows=None):
    rows = []
    seen = set()
    reader = csv.DictReader(handle, strict=True)
    required = {"customer_id", *FEATURES, TARGET}
    header = reader.fieldnames or []
    if len(set(header)) != len(header) or not required.issubset(header):
        raise ValueError(f"CSV needs unique headers and these fields: {', '.join(sorted(required))}.")
    for row in reader:
        if max_rows is not None and len(rows) >= max_rows:
            raise ValueError(f"Choose a CSV with at most {max_rows:,} customers.")
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Wrong number of columns.")
            customer_id = row["customer_id"].strip()
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", customer_id):
                raise ValueError("Invalid customer ID.")
            if customer_id.casefold() in seen:
                raise ValueError("Duplicate customer ID.")
            values = []
            for field, (low, high, whole) in FEATURES.items():
                try:
                    value = float(row[field])
                except ValueError:
                    raise ValueError(f"{field} must be a number.") from None
                if not low <= value <= high or (whole and not value.is_integer()):
                    raise ValueError(f"{field} must be {'a whole number' if whole else 'a number'} between {low} and {high}.")
                values.append(value)
            if row[TARGET].strip() not in ("0", "1"):
                raise ValueError(f"{TARGET} must be 0 or 1.")
            rows.append(Customer(customer_id, tuple(values), int(row[TARGET])))
            seen.add(customer_id.casefold())
        except ValueError as error:
            raise ValueError(f"CSV line {reader.line_num}: {error}") from error
    return rows
