"""
ISL Dataset Validation Tool

Checks the dataset for completeness, balance, and quality.

Usage:
    python src/check_dataset.py
    python src/check_dataset.py --file dataset/isl_landmarks.csv
"""

import pandas as pd
import numpy as np
import os
import sys


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DEFAULT_CSV = "dataset/isl_landmarks.csv"
EXPECTED_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXPECTED_FEATURES = 63
EXPECTED_COLUMNS = 64  # 63 features + 1 label


# --------------------------------------------------
# PARSE ARGUMENTS
# --------------------------------------------------

def parse_args():

    csv_file = DEFAULT_CSV

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--file" and i + 1 < len(sys.argv):
            csv_file = sys.argv[i + 1]
            i += 2

        elif arg == "--help":
            print()
            print("ISL Dataset Validation Tool")
            print("===========================")
            print()
            print("Usage:")
            print("  python src/check_dataset.py")
            print("  python src/check_dataset.py --file path/to/data.csv")
            print()
            sys.exit(0)

        else:
            i += 1

    return csv_file


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    csv_file = parse_args()

    print()
    print("=" * 60)
    print("  ISL DATASET VALIDATION REPORT")
    print("=" * 60)
    print()

    # --------------------------------------------------
    # CHECK FILE EXISTS
    # --------------------------------------------------

    if not os.path.exists(csv_file):
        print(f"ERROR: Dataset file not found: {csv_file}")
        print()
        print("To create a dataset, use one of:")
        print("  python src/collect_data.py")
        print("  python src/prepare_kaggle_dataset.py")
        return False

    print(f"File: {csv_file}")
    file_size = os.path.getsize(csv_file)
    print(f"Size: {file_size / 1024 / 1024:.2f} MB")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        return False

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print()

    issues = []
    warnings = []

    # --------------------------------------------------
    # CHECK COLUMN COUNT
    # --------------------------------------------------

    print("-" * 60)
    print("COLUMN CHECK")
    print("-" * 60)

    if df.shape[1] != EXPECTED_COLUMNS:
        msg = f"Expected {EXPECTED_COLUMNS} columns, found {df.shape[1]}"
        issues.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  OK: {df.shape[1]} columns (63 features + 1 label)")

    # Check label column exists
    if "label" not in df.columns:
        issues.append("Missing 'label' column")
        print("  FAIL: Missing 'label' column")
        return False
    else:
        print("  OK: 'label' column present")

    print()

    # --------------------------------------------------
    # CHECK LABELS
    # --------------------------------------------------

    print("-" * 60)
    print("CLASS CHECK")
    print("-" * 60)

    unique_labels = sorted(df["label"].unique())
    print(f"  Found classes: {', '.join(unique_labels)}")
    print(f"  Count: {len(unique_labels)} classes")
    print()

    # Check for missing classes
    missing = [l for l in EXPECTED_LABELS if l not in unique_labels]
    extra = [l for l in unique_labels if l not in EXPECTED_LABELS]

    if missing:
        msg = f"Missing classes: {', '.join(missing)}"
        issues.append(msg)
        print(f"  FAIL: {msg}")

    if extra:
        msg = f"Unexpected classes: {', '.join(extra)}"
        warnings.append(msg)
        print(f"  WARN: {msg}")

    if not missing and not extra:
        print("  OK: All 26 classes (A-Z) present")

    print()

    # --------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------

    print("-" * 60)
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    label_counts = df["label"].value_counts().sort_index()

    min_count = label_counts.min()
    max_count = label_counts.max()
    mean_count = label_counts.mean()

    for label in sorted(unique_labels):
        count = label_counts[label]

        # Visual bar
        bar_length = int(count / max_count * 30)
        bar = "#" * bar_length

        status = ""
        if count < 100:
            status = " [LOW]"
            warnings.append(f"Class {label} has only {count} samples")

        print(f"  {label}: {count:>6}  {bar}{status}")

    print()
    print(f"  Min: {min_count}")
    print(f"  Max: {max_count}")
    print(f"  Mean: {mean_count:.0f}")
    print(f"  Total: {len(df)}")

    # Check balance
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    if imbalance_ratio > 2.0:
        msg = f"Class imbalance ratio: {imbalance_ratio:.1f}x"
        warnings.append(msg)
        print(f"  WARN: {msg}")
    else:
        print(f"  OK: Balanced (ratio: {imbalance_ratio:.2f}x)")

    print()

    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------

    print("-" * 60)
    print("DATA QUALITY")
    print("-" * 60)

    # Check for NaN/null
    null_count = df.isnull().sum().sum()

    if null_count > 0:
        msg = f"Found {null_count} missing values"
        issues.append(msg)
        print(f"  FAIL: {msg}")

        # Show which columns
        null_cols = df.columns[df.isnull().any()].tolist()
        print(f"  Columns with nulls: {null_cols[:10]}")
    else:
        print("  OK: No missing values")

    # Check for non-numeric features
    feature_cols = [c for c in df.columns if c != "label"]
    non_numeric = 0

    for col in feature_cols:
        try:
            pd.to_numeric(df[col])
        except (ValueError, TypeError):
            non_numeric += 1

    if non_numeric > 0:
        msg = f"Found {non_numeric} non-numeric feature columns"
        issues.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print("  OK: All features are numeric")

    # Check feature ranges
    feature_data = df[feature_cols].values.astype(np.float32)

    abs_max = np.max(np.abs(feature_data))
    abs_min = np.min(np.abs(feature_data))

    print(f"  Feature range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
    print(f"  Absolute max: {abs_max:.4f}")

    if abs_max > 2.0:
        warnings.append(f"Some feature values seem large ({abs_max:.2f})")
        print(f"  WARN: Some values seem large (max={abs_max:.2f})")

    # Check for duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate rows")
        print(f"  WARN: Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
    else:
        print("  OK: No duplicate rows")

    # Check for zero-only rows (wrist coordinates)
    zero_rows = (feature_data == 0).all(axis=1).sum()
    if zero_rows > 0:
        warnings.append(f"Found {zero_rows} all-zero rows")
        print(f"  WARN: Found {zero_rows} all-zero feature rows")

    print()

    # --------------------------------------------------
    # TRAIN/TEST SPLIT PREVIEW
    # --------------------------------------------------

    print("-" * 60)
    print("TRAIN/TEST SPLIT PREVIEW (80/20)")
    print("-" * 60)

    total = len(df)
    train_size = int(total * 0.8)
    test_size = total - train_size

    print(f"  Training: {train_size} samples")
    print(f"  Testing:  {test_size} samples")
    print(f"  Per-class test (approx): {test_size // len(unique_labels)} samples")

    print()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print()

    if not issues and not warnings:
        print("  STATUS: PASS")
        print("  Dataset is ready for training.")
    elif not issues:
        print(f"  STATUS: PASS (with {len(warnings)} warnings)")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"  STATUS: FAIL ({len(issues)} issues, {len(warnings)} warnings)")
        print()
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
        if warnings:
            print()
            print("  Warnings:")
            for w in warnings:
                print(f"    - {w}")

    print()

    return len(issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
