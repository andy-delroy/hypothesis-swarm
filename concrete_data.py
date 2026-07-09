"""
concrete_data.py — Loader for UCI Concrete Compressive Strength dataset.

Dependencies beyond stdlib:
  requests  — HTTP download
  xlrd<2    — .xls parsing; xlrd 2.x dropped legacy XLS support, and
              openpyxl only handles .xlsx, so xlrd 1.x is the single-package
              path for this .xls source file (vs pandas which pulls xlrd anyway
              plus a large transitive dependency tree).

Install: pip install requests "xlrd<2"
"""

from __future__ import annotations

import csv
import os
import random

_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "concrete/compressive/Concrete_Data.xls"
)

# Exact column headers from the UCI .xls file, mapped to clean snake_case.
# Verified against the UCI source: some headers have double spaces and a
# trailing space on the target column — matched literally so the map is robust.
_COL_MAP: dict[str, str] = {
    "Cement (component 1)(kg in a m^3 mixture)":              "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":  "blast_furnace_slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)":             "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)":              "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)":    "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":   "coarse_aggregate",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)":      "fine_aggregate",
    "Age (day)":                                               "age",
    # Age values in the source are whole-number days (1–365) stored as floats.
    "Concrete compressive strength(MPa, megapascals) ":       "strength",
}


def download_concrete_csv(dest_path: str = "data/concrete.csv") -> str:
    """Download the UCI Concrete dataset as CSV, caching after the first fetch.

    Returns dest_path immediately if the file already exists (cache hit).
    Raises RuntimeError with instructions on network failure.
    """
    if os.path.exists(dest_path):
        return dest_path

    try:
        import requests
        import xlrd
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency: {exc}\n"
            "Install with:  pip install requests 'xlrd<2'"
        ) from exc

    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    try:
        resp = requests.get(_URL, timeout=30)
        resp.raise_for_status()
        xls_bytes = resp.content
    except Exception as exc:
        clean_names = ", ".join(_COL_MAP.values())
        raise RuntimeError(
            f"Failed to download UCI Concrete dataset from:\n  {_URL}\n"
            f"Error: {exc}\n\n"
            f"To continue without network access, place a CSV manually at:\n"
            f"  {os.path.abspath(dest_path)}\n"
            f"Expected CSV columns (in order): {clean_names}"
        ) from exc

    wb = xlrd.open_workbook(file_contents=xls_bytes)
    ws = wb.sheet_by_index(0)

    raw_headers = [ws.cell_value(0, c) for c in range(ws.ncols)]
    mapped_headers = []
    for h in raw_headers:
        clean = _COL_MAP.get(h)
        if clean is None:
            # Unexpected header — keep original and warn so the mismatch is visible.
            print(f"  WARNING: unmapped column {h!r} — kept as-is. "
                  "Update _COL_MAP if this is a source-format change.")
            clean = h
        mapped_headers.append(clean)

    with open(dest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(mapped_headers)
        for r in range(1, ws.nrows):
            writer.writerow([ws.cell_value(r, c) for c in range(ws.ncols)])

    print(f"Saved {ws.nrows - 1} rows → {dest_path}")
    return dest_path


def load_concrete(test_frac: float = 0.2, seed: int = 0):
    """Load the Concrete dataset and return a reward.Dataset.

    Splits deterministically: shuffle with random.Random(seed), then hold out
    the first test_frac fraction as the test set (should be ~206 test / 824 train
    for the 1030-row UCI dataset at default settings).
    """
    from reward import Dataset

    csv_path = download_concrete_csv()

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    random.Random(seed).shuffle(rows)

    n_test = round(len(rows) * test_frac)
    test_rows = rows[:n_test]
    train_rows = rows[n_test:]

    return Dataset(
        task_type="regression",
        target="strength",
        train=train_rows,
        test=test_rows,
    )


if __name__ == "__main__":
    ds = load_concrete()
    total = len(ds.train) + len(ds.test)
    print(f"\nTotal rows : {total}")
    print(f"Train rows : {len(ds.train)}")
    print(f"Test rows  : {len(ds.test)}")
    print("\nSample (first 2 train rows):")
    for row in ds.train[:2]:
        print(" ", row)
