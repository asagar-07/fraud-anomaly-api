# To Transform the Data + Scaler fit/transform + save/load artifacts

# 1. Load df from ingest.py
# 2. Drop exact duplicates, log before/after row counts
# 3. Sort by Time ascending
# 4. Create X using canonical feature columns from ingest.py
# 5. Transform 'Amount' using log1p(Amount)
# 6. Split into train/val/test using chronological split 70/15/15
# 7. Fit StandardScaler on TRAIN X  only, transform train/val/test
# 8. Persist artifacts ( shared for both models + API) at "artifacts/shared/"
# 9. Scaler.joblib
# 10. preprocessor_config.json (feature_order list, amount_transform: "log1p", time_transform: "none", scaler: "StandardScaler", Split ratios: train: 0.7, val: 0.15, test: 0.15, duplicate_handling: "dropped")
# 11. train_stats.json with TRAIN only (row_count, fraud_count, fraud_percentage, feature_stats {feature_name: {mean, std, min, max} } pre-scaling)
# 12. split_stats.json with per-split (row_count, fraud_count, fraud_percentage)
# 13. Ensure No NaNs/ inf after log1p and scaling

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from fraud.data.ingest import IngestConfig, ingest_data
from fraud.data.split import SplitConfig, split_data_chronologically
import joblib
import json 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the path to the source CSV file from root directory
RAW_REL_PATH = Path("data/raw/creditcard.csv")

# Find the Project Root Directory
project_root = Path(__file__).resolve().parents[3]
raw_data_path = project_root / RAW_REL_PATH



# Config for preprocessing
@dataclass(frozen=True)
class PreprocessConfig:
    AMOUNT_TRANSFORM = "log1p"
    TIME_TRANSFORM = "none"
    DUPLICATE_HANDLING = "dropped"
    ARTIFACTS_DIR = project_root / "artifacts/shared/"
    SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
    PREPROCESSOR_CONFIG_PATH = ARTIFACTS_DIR / "preprocessor_config.json"
    TRAIN_STATS_PATH = ARTIFACTS_DIR / "train_stats.json"

def preprocess_data(df: pd.DataFrame) -> dict:
    """
    Preprocess the input DataFrame.
    """
    print ("\n Starting preprocessing data... \n")
    #1. Ensure artifacts directory exists
    PreprocessConfig.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Drop exact duplicates
    initial_row_count = df.shape[0]
    df = df.drop_duplicates()
    final_row_count = df.shape[0]
    print(f"Dropped {initial_row_count - final_row_count} duplicate rows.")

    # 3. Transform 'Amount' using log1p(Amount)
    df["Amount"] = np.log1p(df["Amount"])
    
    # 4. Split data from split.py to get split statistics
    split_data = split_data_chronologically(df)

    train_df = split_data["train"]
    val_df = split_data["val"]
    test_df = split_data["test"]

    # 5. Create X using canonical feature columns
    X_train = train_df[IngestConfig.FEATURE_COLS].copy()
    y_train = train_df[IngestConfig.TARGET_COL].copy()

    X_val = val_df[IngestConfig.FEATURE_COLS].copy()
    y_val = val_df[IngestConfig.TARGET_COL].copy()

    X_test = test_df[IngestConfig.FEATURE_COLS].copy()
    y_test = test_df[IngestConfig.TARGET_COL].copy()
    
    # 6. Fit StandardScaler on TRAIN X only, transform train/val/test
    scaler = StandardScaler()

    #---Guardrail: Ensure scaler is only fit on train data
    scaler._fitted_on_train = False

    def fit_scaler_on_train(X_train):
        if scaler._fitted_on_train:
            raise RuntimeError("StandardScaler attempted to fit more than once.")

        logger.info("Fitting StandardScaler on training data only. Shape: %s", X_train.shape)

        X_train_scaled = scaler.fit_transform(X_train)
        scaler._fitted_on_train = True
        return X_train_scaled

    def transform_with_scaler(X):
        if not scaler._fitted_on_train:
            raise RuntimeError("StandardScaler used before fitting on TRAIN.")
        return scaler.transform(X)

    X_train_scaled_arr = fit_scaler_on_train(X_train)
    X_val_scaled_arr   = transform_with_scaler(X_val)
    X_test_scaled_arr  = transform_with_scaler(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=IngestConfig.FEATURE_COLS)
    X_val_scaled   = pd.DataFrame(X_val_scaled_arr, columns=IngestConfig.FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(X_test_scaled_arr, columns=IngestConfig.FEATURE_COLS)

    # 7. Ensure output arrays are always shape: (n_samples, 30)
    for df in [X_train_scaled, X_val_scaled, X_test_scaled]:
        if df.shape[1] != 30:
            raise ValueError(f"Expected 30 features, but got {df.shape[1]}")
        
    # 8. Ensure No NaNs/ inf after log1p and scaling
    for name, df in zip(["X_train_scaled", "X_val_scaled", "X_test_scaled"], [X_train_scaled, X_val_scaled, X_test_scaled]):
        if df.isnull().values.any():
            raise ValueError(f"{name} contains NaN values after preprocessing.")
        if np.isinf(df.values).any():
            raise ValueError(f"{name} contains infinite values after preprocessing.")

    # 9. Persist artifacts - Scaler
    joblib.dump(scaler, PreprocessConfig.SCALER_PATH)

    #10. Persist artifacts - Preprocessor config
    preprocessor_config = {
        "feature_order": IngestConfig.FEATURE_COLS,
        "amount_transform": PreprocessConfig.AMOUNT_TRANSFORM,
        "time_transform": PreprocessConfig.TIME_TRANSFORM,
        "scaler": "StandardScaler",
        "split_ratios": SplitConfig.SPLIT_RATIOS,
        "duplicate_handling": PreprocessConfig.DUPLICATE_HANDLING
    } 

    with open(PreprocessConfig.PREPROCESSOR_CONFIG_PATH, 'w') as f:
        json.dump(preprocessor_config, f, indent=4)


    # 11. Persist artifacts - Train stats 
    # Feature stats for TRAIN only, MEAN, STD, MIN, MAX pre-scaling, fraud counts and percentages
    train_stats = {
        "row_count": X_train.shape[0],
        "fraud_count": int(y_train.sum()),
        "fraud_percentage": float(y_train.mean() * 100),
        "feature_stats": {}
    }

    for col in IngestConfig.FEATURE_COLS:
        train_stats["feature_stats"][col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
        }

    with open(PreprocessConfig.TRAIN_STATS_PATH, 'w') as f:
        json.dump(train_stats, f, indent=4)


    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }    

if __name__ == "__main__":
    df = ingest_data(raw_data_path)

    if not df.empty:
        # Preprocess data
        processed_data = preprocess_data(df)

        print("Preprocessing complete.")
        print(f"Processed train shape: {processed_data['X_train_scaled'].shape}")
        print(f"Processed val shape: {processed_data['X_val_scaled'].shape}")
        print(f"Processed test shape: {processed_data['X_test_scaled'].shape}")