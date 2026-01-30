# To Split the Data into Train/Validation/Test Sets
# sort by Time + chronological split + save split statistics

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from fraud.data.ingest import IngestConfig
import json

# Find the Project Root Directory
project_root = Path(__file__).resolve().parents[3]


# Config for data splitting
@dataclass(frozen=True)
class SplitConfig:
    SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
    ARTIFACTS_DIR = project_root / "artifacts/shared/"
    SPLIT_STATS_PATH = ARTIFACTS_DIR / "split_stats.json"

def split_data_chronologically(df: pd.DataFrame) -> dict:
    """
    Split the DataFrame into train/val/test sets based on the defined split ratios.
    """
    # Ensure artifacts directory exists
    SplitConfig.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by Time
    df = df.sort_values(by="Time").reset_index(drop=True)

    # Split the data
    n = len(df)
    train_end = int(n * SplitConfig.SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SplitConfig.SPLIT_RATIOS["val"])

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Save split statistics
    split_stats = {
        "overall": {"row_count": df.shape[0], "fraud_count": int(df[IngestConfig.TARGET_COL].sum()), "fraud_percentage": float(df[IngestConfig.TARGET_COL].mean() * 100)},
        "train": {"row_count": train_df.shape[0], "fraud_count": int(train_df[IngestConfig.TARGET_COL].sum()), "fraud_percentage": float(train_df[IngestConfig.TARGET_COL].mean() * 100)},
        "val": {"row_count": val_df.shape[0], "fraud_count": int(val_df[IngestConfig.TARGET_COL].sum()), "fraud_percentage": float(val_df[IngestConfig.TARGET_COL].mean() * 100)},
        "test": {"row_count": test_df.shape[0], "fraud_count": int(test_df[IngestConfig.TARGET_COL].sum()), "fraud_percentage": float(test_df[IngestConfig.TARGET_COL].mean() * 100)}
    }

    with open(SplitConfig.SPLIT_STATS_PATH, 'w') as f:
        json.dump(split_stats, f, indent=4)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }