from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fraud.models.autoencoder import FraudAutoencoder
from fraud.data.preprocess import preprocess_data
from fraud.data.ingest import ingest_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrainConfig:
    raw_csv_path: str = "data/raw/creditcard.csv"
    artifacts_dir: str = "artifacts/autoencoder"
    input_dim: int = 30
    hidden_dims: Tuple[int, int] = (16, 8)
    loss: str = "mae"  # Mean Absolute Error

    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    seed: int = 42
    
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)

def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalars → python scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    # numpy arrays → lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # torch tensors → python/list
    try:
        if isinstance(obj, torch.Tensor):   
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    # fallback
    return obj

def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        payload = {k: _json_safe(v) for k, v in payload.items()}
    else:
        payload = _json_safe(payload)
    path.write_text(json.dumps(payload, indent=2))


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, return_scores: bool = False,) -> Dict[str, Any]:
    model.eval()
    total_mae_sum = 0.0
    total_n = 0

    scores_tensors = []

    for batch in loader:
        input_data = batch[0] if isinstance(batch, (tuple, list)) else batch
        input_data = input_data.to(device)
        reconstruction = model(input_data)

        per_sample_mae = torch.mean(torch.abs(input_data - reconstruction), dim=1)

        total_mae_sum += float(per_sample_mae.sum().item())
        total_n += int(per_sample_mae.numel())

        if return_scores:
            scores_tensors.append(per_sample_mae.detach().cpu())

    recon_mean_mae = total_mae_sum / max(total_n, 1)

    out: Dict[str, Any] = {
        "recon_mae_mean": float(recon_mean_mae),
        "total_samples": int(total_n),
    }

    if return_scores:
        if scores_tensors:
            scores = torch.cat(scores_tensors, dim=0).numpy().astype(np.float64)
        else:
            scores = np.array([], dtype=np.float64)
        out["scores"] = scores

    return out


def compute_recon_stats(scores: np.ndarray) -> Dict[str, Any]:
    '''
    Compute summary statistics from anomaly scoreing thresholds.
    Input: scores: per-sample reconstruction MAE, shape (N,)
    Returns: mean, std, percentiles
    '''
    if scores is None:
        raise ValueError("Scores array is None.")
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Scores array is empty.")
    
    stats = {
        "n_samples": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=0)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p95": float(np.percentile(scores, 95.0)),
        "p97": float(np.percentile(scores, 97.0)),
        "p99": float(np.percentile(scores, 99.0)),
        "p995": float(np.percentile(scores, 99.5)),
    }
    return stats

#----------TRAINING------------
def train_autoencoder() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    repo_root = Path(__file__).resolve().parents[3]
    raw_data_path = repo_root / cfg.raw_csv_path
    artifacts_dir = repo_root / cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Ingest and preprocess data
    logger.info("Ingesting data...")
    df = ingest_data(str(raw_data_path))

    logger.info("Preprocessing data for Autoencoder...")
    out = preprocess_data(df)
    X_train = out["X_train_scaled"]
    y_train = out["y_train"]
    X_val = out["X_val_scaled"]
    y_val = out["y_val"]

        # Convert DataFrames/Series to numpy
    if hasattr(X_train, "to_numpy"):
        X_train = X_train.to_numpy()
    if hasattr(y_train, "to_numpy"):
        y_train = y_train.to_numpy()    
    if hasattr(X_val, "to_numpy"):
        X_val = X_val.to_numpy()
    if hasattr(y_val, "to_numpy"):
        y_val = y_val.to_numpy()

    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)

    # Use only non-fraudulent data for training the autoencoder
    train_mask = (y_train == 0)
    val_mask = (y_val == 0)

    X_train_norm = X_train[train_mask]
    X_val_norm = X_val[val_mask]

    logger.info(
        "TRAIN: total=%d normal=%d fraud=%d",
        int(len(y_train)), int(train_mask.sum()), int((y_train == 1).sum())
    )
    logger.info(
        "VAL:   total=%d normal=%d fraud=%d",
        int(len(y_val)), int(val_mask.sum()), int((y_val == 1).sum())
    )

    if X_train_norm.shape[0] == 0:
        raise ValueError("No normal samples found in training set after filtering (y_train==0).")
    if X_val_norm.shape[0] == 0:
        raise ValueError("No normal samples found in validation set after filtering (y_val==0).")


    train_dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model, optimizer, Criterion
    model = FraudAutoencoder(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims).to(device)

    if cfg.loss.lower() != "mae":
        raise ValueError(f"This training script only supports MAE loss, got: {cfg.loss}")
    
    loss_fn = torch.nn.L1Loss(reduction="mean") # Criterion: Mean Absolute Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
# Early Stopping Setup on  Validation Reconstruction Loss
    best_val_loss = float("inf")
    best_state = None
    patience_left = cfg.early_stopping_patience

    train_curve = {"epoch": [], "train_recon_mae": [], "val_recon_mae": []}

    logger.info("Starting training for Autoencoder...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for (features, ) in train_loader:
            features = features.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, features)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * features.size(0)
            n_samples += int(features.size(0))

        train_loss = running_loss / max(n_samples, 1)

        # Validation Reconstruction Loss Calculation
        val_metrics = evaluate_model(model=model, loader=val_loader, device=device, return_scores=False)
        val_loss = float(val_metrics["recon_mae_mean"])

        logger.info("Epoch %d/%d | train_recon_mae=%.6f | val_recon_mae=%.6f", epoch, cfg.epochs, train_loss, val_loss)

        train_curve["epoch"].append(epoch)
        train_curve["train_recon_mae"].append(float(train_loss))
        train_curve["val_recon_mae"].append(float(val_loss))

        # Early Stopping Check (using training loss as proxy for validation loss here)
        if val_loss < best_val_loss - 1e-6: #small tolerance to avoid tiny improvements
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stopping_patience

            # Save the best model
            torch.save(best_state, artifacts_dir / "model.pt")
            logger.info("New best val_recon_mae=%.6f -> saved %s", best_val_loss, artifacts_dir / "model.pt")
        else:
            patience_left -= 1
            logger.info(f"No improvement in validation reconstruction loss. Patience left: {patience_left}")    
            if patience_left <= 0:
                logger.info("Early stopping triggered. Best val_recon_mae=%.6f", best_val_loss)
                break

    #Load best model state for Scoring
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Loaded best model state for final evaluation.")
    else:
        logger.warning("No best model state found; using last epoch model for final evaluation.")
    model.eval()


    # --- Compute TRAIN_NORMAL reconstruction scores and distribution stats for thresholds ---
    # Use a non-shuffled DataLoader for stable scoring
    train_score_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    train_eval = evaluate_model(model=model, loader=train_score_loader, device=device, return_scores=True)
    scores = train_eval["scores"]
    recon_stats = compute_recon_stats(scores)
    
    logger.info("Reconstruction Statistics Computed from Training Scores:")
    for k, v in recon_stats.items():
        logger.info(f"{k}: {v:.6f}")
    
    # save JSON artifacts 
    model_config = {
        "model_type": "Autoencoder",
        "input_dim": cfg.input_dim,
        "hidden_dim": cfg.hidden_dims,
        "loss_function": cfg.loss,
        "score_definition": "Mean Absolute Error between input and reconstruction over 30 features",
        "train_on": "Normal samples only (Class 0)",
    }

    metrics = {
        "best_val_recon_mae": float(best_val_loss),
        "epochs_ran": int(len(train_curve["epoch"])),
        "final_train_recon_mae": float(train_curve["train_recon_mae"][-1]) if train_curve["train_recon_mae"] else None,
        "final_val_recon_mae": float(train_curve["val_recon_mae"][-1]) if train_curve["val_recon_mae"] else None,
    }

    save_json(artifacts_dir / "model_config.json", model_config)
    save_json(artifacts_dir / "metrics.json", metrics)
    save_json(artifacts_dir / "recon_stats.json", recon_stats)
    save_json(artifacts_dir / "train_curve.json", train_curve)

    logger.info("Saved autoencoder JSON artifacts to: %s", artifacts_dir.resolve())
    logger.info("Recon stats (train normal): p95=%.6f p99=%.6f p995=%.6f",
                recon_stats["p95"], recon_stats["p99"], recon_stats["p995"])


if __name__ == "__main__":
    train_autoencoder()