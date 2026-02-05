from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from logging import config
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from fraud.models.classifier_mlp import FraudMLP
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
    artifacts_dir: str = "artifacts/classifier"
    input_dim: int = 30
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    seed: int = 42
    threshold: float = 0.5  # Initial threshold for classification

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

def to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    x = x.detach()          # Remove from graph
    x = x.cpu()             # Move to CPU
    x = x.numpy()           # Convert to NumPy
    x = x.reshape(-1)       # Flatten to 1D
    return x

def compute_pos_weight(y_train: np.ndarray) -> float:
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos == 0:
        raise ValueError("No positive samples in training set; cannot compute pos_weight.")
    return neg / pos

@torch.no_grad()
def evaluate_model( model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> Dict[str, float]:
    model.eval()
    all_logits  = []
    all_y = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        all_logits.append(logits.cpu())
        all_y.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0).reshape(-1)
    y_true = torch.cat(all_y, dim=0).reshape(-1).numpy().astype(int)

    probs = torch.sigmoid(logits).numpy()

    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")

    y_pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "f1@0.5": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "specificity@0.5": float(specificity),
    }

    return metrics

def _json_safe(obj):
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
        import torch
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

def train_classifier() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    logger.info("Ingesting data...")
    df = ingest_data(cfg.raw_csv_path)

    logger.info("Preprocessing data for Classifier...")
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

    #----Wrap datasets----
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1,1), dtype=torch.float32), #reshape to (N,1) for BCEWithLogitsLoss
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.reshape(-1,1), dtype=torch.float32), #reshape to (N,1) for BCEWithLogitsLoss
    )

    #----DataLoader setup----
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False) #shuffle=True for training to help generalization (reduce overfitting)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False) #shuffle=False for validation
    
    #----Device setup----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    #----Compute pos_weight for BCEWithLogitsLoss----
    pos_weight_value = compute_pos_weight(y_train)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    logger.info(f"Positive samples in training set: {np.sum(y_train == 1)}")
    logger.info(f"Negative samples in training set: {np.sum(y_train == 0)}")
    logger.info(f"Computed pos_weight: {pos_weight_value:.4f}")

    #----Model, Loss, Optimizer setup----
    model = FraudMLP(input_dim=cfg.input_dim).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    # -- Early Stopping Setup on validation PR-AUC --
    best_pr_auc = -1.0
    best_model_state = None
    patience_left = cfg.early_stopping_patience

    train_curve = {"epoch": [], "train_loss": [], "val_pr_auc": []}

    logger.info("Starting training for Classifier...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).reshape(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * features.size(0)
            n_samples += features.size(0)

        train_loss = running / max(n_samples, 1)

        #--- Validation PR-AUC ---
        val_metrics = evaluate_model(model=model, loader=val_loader, device=device, threshold=cfg.threshold)
        val_pr_auc = val_metrics["pr_auc"]

        logger.info(f"Epoch {epoch}: | Train Loss: {train_loss:.4f} | Val PR-AUC: {val_pr_auc:.6f}")

        train_curve["epoch"].append(epoch)
        train_curve["train_loss"].append(train_loss)
        train_curve["val_pr_auc"].append(val_pr_auc)

        #--- Early Stopping Check ---
        if val_pr_auc > best_pr_auc + 1e-6:  # small tolerance to avoid tiny improvements
            best_pr_auc = val_pr_auc
            best_model_state = model.state_dict()
            patience_left = cfg.early_stopping_patience
            logger.info(f"New best model found at epoch {epoch} with Val PR-AUC: {val_pr_auc:.6f}")
        else:
            patience_left -= 1
            logger.info(f"No improvement in Val PR-AUC. Patience left: {patience_left}")
            if patience_left <= 0:
                logger.info("Early stopping triggered.")
                break

    #--- Load best model state ---
    # If no improvement was found during training, best_model_state may be None
    # In that case, we keep the last model state
    # This ensures we always have a model to evaluate/save 
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.info("No improvement during training; using last model state.")
    
    logger.info("Training complete.")

    #--- Final Validation metrics saved (threshold=0.5 placeholder)---
    final_val_metrics = evaluate_model(model=model, loader=val_loader, device=device, threshold=cfg.threshold)
    logger.info("Final Validation Metrics:")
    for k, v in final_val_metrics.items():
        logger.info(f"{k}: {v:.6f}")

    final_val_metrics["best_val_pr_auc"] = float(best_pr_auc)

    #--- Save artifacts ---
    artifacts_path = Path(cfg.artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_save_path = artifacts_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    #Save model config
    model_config_path = artifacts_path / "model_config.json"
    model_config = {
        "model_type": "FraudMLP",
        "input_dim": cfg.input_dim,
        "hidden_dim": [64, 32],
        "dropout": [0.3, 0.2],
        "batchnorm": True
    }
    save_json(model_config_path, model_config)
    logger.info(f"Model config saved to {model_config_path}")

    # Save metrics and training curve
    metrics_path = artifacts_path / "metrics.json"
    save_json(metrics_path, final_val_metrics)
    logger.info(f"Validation metrics saved to {metrics_path}")

    train_curve_path = artifacts_path / "train_curve.json"
    save_json(train_curve_path, train_curve)
    logger.info(f"Training curve saved to {train_curve_path}")

    # Save threshold placeholder
    threshold_path = artifacts_path / "threshold.json"
    save_json(threshold_path, {"threshold": cfg.threshold})
    logger.info(f"Threshold saved to {threshold_path}")

if __name__ == "__main__":
    train_classifier()