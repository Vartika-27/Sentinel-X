"""
network/preprocess.py

Preprocessing pipeline for the CICIDS2017 network intrusion detection dataset.

Pipeline steps:
    1. Load CSV
    2. Print all column names
    3. Replace ±Infinity with NaN and drop NaN rows
    4. Coerce all columns to numeric where possible
    5. Encode Label → binary (BENIGN=0, attack=1)
    6. Retain only numeric feature columns
    7. Drop highly correlated features (threshold > 0.9)
    8. Normalise with StandardScaler
    9. Persist scaler to disk

Usage:
    from network.preprocess import preprocess

    X, y, feature_names = preprocess()
"""

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH: str = "backend/network/data/cicids.csv"
SCALER_PATH: str = "backend/network/scaler.pkl"
LABEL_COLUMN: str = "Label"
BENIGN_CLASS: str = "BENIGN"
CORRELATION_THRESHOLD: float = 0.9


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    """Load dataset from *path* and print all column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")

    df = pd.read_csv(path, low_memory=False)

    # Strip accidental leading/trailing whitespace in column names
    df.columns = df.columns.str.strip()

    print(f"[load]  Loaded {len(df):,} rows × {len(df.columns)} columns from '{path}'")
    print("\n── Column names " + "─" * 60)
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col}")
    print("─" * 76 + "\n")

    return df


def _sanitise(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±Infinity with NaN, then drop all NaN rows."""
    before = len(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    dropped = before - len(df)
    print(f"[sanitise]  Replaced ±Inf with NaN → dropped {dropped:,} rows  "
          f"(remaining: {len(df):,})")
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce every non-Label column to numeric where possible.

    Columns that cannot be coerced at all (all-NaN after conversion, i.e.
    pure string / categorical columns) are dropped entirely.  Only then are
    rows with any remaining NaN removed.
    """
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns that are entirely NaN (non-numeric / pure-text columns)
    all_nan_cols = [col for col in feature_cols if df[col].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        print(f"[coerce]  Dropped {len(all_nan_cols)} non-numeric column(s): {all_nan_cols}")

    # Now drop rows with any remaining NaN (partial coercion failures)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)

    if dropped:
        print(f"[coerce]  Dropped {dropped:,} rows with residual non-numeric values.")
    else:
        print("[coerce]  All feature columns successfully coerced to numeric.")

    return df


def _encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Encode the Label column into binary targets.
        BENIGN → 0
        anything else → 1
    """
    if LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Expected a '{LABEL_COLUMN}' column, but it was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    unique_labels = df[LABEL_COLUMN].unique()
    print(f"\n[labels]  Unique raw labels ({len(unique_labels)}): "
          f"{sorted(str(l) for l in unique_labels)}")

    y: pd.Series = (df[LABEL_COLUMN].str.strip() != BENIGN_CLASS).astype(int)

    benign_count = int((y == 0).sum())
    attack_count = int((y == 1).sum())
    print(f"[labels]  BENIGN=0 → {benign_count:,} samples  |  "
          f"Attack=1 → {attack_count:,} samples")

    return df.drop(columns=[LABEL_COLUMN]), y


def _keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only columns with a numeric dtype."""
    numeric_df = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"[numeric]  Dropped non-numeric columns: {sorted(dropped)}")
    else:
        print("[numeric]  All remaining columns are numeric.")
    return numeric_df


def _drop_correlated(df: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """
    Remove features whose absolute pairwise Pearson correlation exceeds
    *threshold*.  For each correlated pair the second column is removed.
    """
    corr_matrix = df.corr(method="pearson").abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper_triangle.columns
        if (upper_triangle[col] > threshold).any()
    ]

    df = df.drop(columns=to_drop)
    print(f"[corr]  Dropped {len(to_drop)} highly correlated features "
          f"(threshold={threshold})  →  {len(df.columns)} features remain.")

    return df


def _normalise(df: pd.DataFrame, scaler_path: str) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on *df*, transform it, and persist the fitted
    scaler to *scaler_path*.
    """
    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(df)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[scaler]  Fitted StandardScaler saved → '{scaler_path}'")
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    data_path: str = DATA_PATH,
    scaler_path: str = SCALER_PATH,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Run the full preprocessing pipeline for CICIDS2017.

    Args:
        data_path (str): Path to the raw CICIDS CSV file.
        scaler_path (str): Destination path for the serialised StandardScaler.
        correlation_threshold (float): Pearson correlation cutoff (default 0.9).

    Returns:
        X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary label vector of shape (n_samples,).
        feature_names (list[str]): Ordered list of retained feature names.
    """
    print("=" * 76)
    print("  CICIDS2017 Preprocessing Pipeline")
    print("=" * 76 + "\n")

    # 1 ── Load ──────────────────────────────────────────────────────────────
    df = _load_csv(data_path)

    # 2 ── Sanitise ──────────────────────────────────────────────────────────
    df = _sanitise(df)

    # 3 ── Coerce to numeric ─────────────────────────────────────────────────
    df = _coerce_numeric(df)

    # 4 ── Encode labels ─────────────────────────────────────────────────────
    df, y = _encode_labels(df)

    # 5 ── Keep only numeric feature columns ─────────────────────────────────
    df = _keep_numeric(df)

    # 6 ── Remove highly correlated features ─────────────────────────────────
    df = _drop_correlated(df, threshold=correlation_threshold)

    # 7 ── Normalise ──────────────────────────────────────────────────────────
    X, _ = _normalise(df, scaler_path)

    feature_names: list = df.columns.tolist()
    y_array: np.ndarray = y.to_numpy()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  Pipeline complete")
    print("=" * 76)
    print(f"\n  Number of features  : {len(feature_names)}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label vector shape  : {y_array.shape}")
    print("\n  Selected feature names:")
    for i, name in enumerate(feature_names, 1):
        print(f"    {i:>3}. {name}")
    print()

    return X, y_array, feature_names