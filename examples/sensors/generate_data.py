#!/usr/bin/env python3
"""Regenerate the committed synthetic sensor CSVs (fixed random seed)."""

import csv
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(20261011)
    centers = {
        "idle": np.array([-1.5, -1.3, -1.0, -1.1]),
        "running": np.array([1.2, 1.3, 0.7, 1.1]),
        "high_load": np.array([2.8, 2.4, 2.3, 2.6]),
    }
    scenarios = {
        "baseline": [("idle", 48), ("running", 48)],
        "changed": [("idle", 36), ("running", 36), ("high_load", 36)],
    }
    folder = Path(__file__).resolve().parent / "data"
    folder.mkdir(exist_ok=True)
    for name, counts in scenarios.items():
        rows = []
        for state, count in counts:
            for values in rng.normal(centers[state], 0.20, (count, 4)):
                rows.append((state, *values))
        # Rare unusual readings: HDBSCAN may assign some to noise. This is
        # illustrative; the result is sensitive to the sample and parameters.
        for values in rng.uniform(-4.5, 4.5, (5, 4)):
            rows.append(("unusual", *values))
        rng.shuffle(rows)
        with (folder / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["state", "vibration", "motor_current", "temperature", "acoustic"])
            writer.writerows([[state, *[f"{v:.5f}" for v in values]]
                              for state, *values in rows])


if __name__ == "__main__":
    main()
