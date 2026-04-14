"""Experiment runner: trains models, generates predictions, handles time-series CV."""
from __future__ import annotations

import json
import logging
import uuid
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


# ── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "logistic_regression": {
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
        "description": "Simple, interpretable linear model. Good baseline.",
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "default_params": {
            "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "min_samples_leaf": 10,
        },
        "description": "Gradient boosted trees. Captures nonlinear interactions.",
    },
    "lightgbm": {
        "class": None,  # handled separately
        "default_params": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 20, "verbosity": -1,
        },
        "description": "LightGBM. Fast, handles many features well.",
    },
    "xgboost": {
        "class": None,  # handled separately
        "default_params": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 10, "verbosity": 0,
        },
        "description": "XGBoost. Strong tree ensemble method.",
    },
    "neural_network": {
        "class": MLPClassifier,
        "default_params": {
            "hidden_layer_sizes": (64, 32), "max_iter": 500,
            "learning_rate_init": 0.001, "early_stopping": True,
            "validation_fraction": 0.15, "random_state": 42,
        },
        "description": "Multi-layer perceptron. Captures complex patterns.",
    },
}


def get_available_methods() -> list[dict]:
    """Return list of available model methods with descriptions."""
    return [
        {"method": name, "description": info["description"]}
        for name, info in MODEL_REGISTRY.items()
    ]


def _create_model(method: str, params: Optional[dict] = None):
    """Create a model instance from the registry."""
    if method not in MODEL_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[method]
    model_params = {**info["default_params"], **(params or {})}

    if method == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**model_params)
    elif method == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**model_params, use_label_encoder=False, eval_metric="logloss")
    else:
        return info["class"](**model_params)


def run_experiment(
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "HOME_WIN",
    method: str = "logistic_regression",
    model_params: Optional[dict] = None,
    n_splits: int = 5,
    calibrate: bool = True,
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Run a complete experiment with time-series cross-validation.

    Args:
        feature_matrix: DataFrame with features and target
        feature_cols: List of feature column names to use
        target_col: Target column name
        method: Model method from MODEL_REGISTRY
        model_params: Override default model parameters
        n_splits: Number of time-series CV splits
        calibrate: Whether to apply Platt scaling calibration
        experiment_name: Human-readable name
        run_id: Parent run ID

    Returns:
        Dict with experiment results including metrics, predictions, and model.
    """
    experiment_id = str(uuid.uuid4())[:8]
    if not experiment_name:
        experiment_name = f"{method}_{experiment_id}"

    logger.info(f"Running experiment: {experiment_name} (method={method})")

    # Ensure data is sorted by date
    df = feature_matrix.sort_values("GAME_DATE").reset_index(drop=True)

    # Validate feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "method": method,
            "status": "failed",
            "error": f"Missing feature columns: {missing}",
        }

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    all_test_indices = []
    all_test_preds = []
    all_test_true = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = _create_model(method, model_params)

        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            logger.warning(f"Fold {fold} failed: {e}")
            continue

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.predict(X_test_scaled).astype(float)

        from tools.metrics import compute_metrics
        fold_metrics = compute_metrics(y_test, y_prob)
        fold_metrics["fold"] = fold
        fold_metrics["train_size"] = len(train_idx)
        fold_metrics["test_size"] = len(test_idx)
        fold_results.append(fold_metrics)

        all_test_indices.extend(test_idx.tolist())
        all_test_preds.extend(y_prob.tolist())
        all_test_true.extend(y_test.tolist())

    if not fold_results:
        return {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "method": method,
            "status": "failed",
            "error": "All folds failed",
        }

    # Aggregate metrics across folds
    avg_metrics = {}
    for key in ["log_loss", "brier_score", "accuracy", "sharpness"]:
        values = [f[key] for f in fold_results if key in f]
        if values:
            avg_metrics[key] = round(np.mean(values), 6)
            avg_metrics[f"{key}_std"] = round(np.std(values), 6)

    # Overall metrics on all OOS predictions
    overall = compute_metrics(np.array(all_test_true), np.array(all_test_preds))

    # Train final model on all data
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    final_model = _create_model(method, model_params)
    final_model.fit(X_scaled, y)

    # Save model artifact
    artifact_dir = BASE_DIR / "data" / "experiments" / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "scaler": scaler_final, "feature_cols": feature_cols}, f)

    # Save predictions
    preds_df = pd.DataFrame({
        "index": all_test_indices,
        "y_true": all_test_true,
        "y_prob": all_test_preds,
    })
    preds_df.to_parquet(artifact_dir / "predictions.parquet", index=False)

    # Feature importance (if available)
    feature_importance = {}
    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
        feature_importance = dict(sorted(
            zip(feature_cols, importance.tolist()),
            key=lambda x: x[1], reverse=True
        ))
    elif hasattr(final_model, "coef_"):
        coefs = final_model.coef_[0] if final_model.coef_.ndim > 1 else final_model.coef_
        feature_importance = dict(sorted(
            zip(feature_cols, np.abs(coefs).tolist()),
            key=lambda x: x[1], reverse=True
        ))

    result = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "name": experiment_name,
        "method": method,
        "config": model_params or MODEL_REGISTRY[method]["default_params"],
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "train_samples": int(len(X)),
        "test_samples": int(len(all_test_true)),
        "n_folds": len(fold_results),
        "fold_metrics": fold_results,
        "avg_metrics": avg_metrics,
        "overall_metrics": overall,
        "feature_importance": feature_importance,
        "artifact_dir": str(artifact_dir),
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    # Save experiment result as JSON
    with open(artifact_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        f"  Experiment {experiment_name}: log_loss={overall['log_loss']:.4f}, "
        f"brier={overall['brier_score']:.4f}, accuracy={overall['accuracy']:.4f}"
    )
    return result


def run_ensemble_experiment(
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    methods: list[str] = ["logistic_regression", "gradient_boosting"],
    target_col: str = "HOME_WIN",
    n_splits: int = 5,
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Run an ensemble experiment averaging predictions from multiple models."""
    experiment_id = str(uuid.uuid4())[:8]
    if not experiment_name:
        experiment_name = f"ensemble_{'_'.join(methods)}_{experiment_id}"

    logger.info(f"Running ensemble experiment: {experiment_name}")

    df = feature_matrix.sort_values("GAME_DATE").reset_index(drop=True)
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_test_indices = []
    all_test_preds = []
    all_test_true = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        fold_preds = []
        for method in methods:
            try:
                model = _create_model(method)
                model.fit(X_train_scaled, y_train)
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    pred = model.predict(X_test_scaled).astype(float)
                fold_preds.append(pred)
            except Exception as e:
                logger.warning(f"Ensemble fold {fold}, method {method} failed: {e}")

        if fold_preds:
            avg_pred = np.mean(fold_preds, axis=0)
            all_test_indices.extend(test_idx.tolist())
            all_test_preds.extend(avg_pred.tolist())
            all_test_true.extend(y_test.tolist())

    from tools.metrics import compute_metrics
    overall = compute_metrics(np.array(all_test_true), np.array(all_test_preds))

    artifact_dir = BASE_DIR / "data" / "experiments" / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "name": experiment_name,
        "method": f"ensemble({', '.join(methods)})",
        "config": {"methods": methods},
        "features_used": feature_cols,
        "train_samples": int(len(X)),
        "test_samples": int(len(all_test_true)),
        "overall_metrics": overall,
        "artifact_dir": str(artifact_dir),
        "status": "completed",
        "created_at": datetime.now().isoformat(),
    }

    with open(artifact_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        f"  Ensemble {experiment_name}: log_loss={overall['log_loss']:.4f}, "
        f"brier={overall['brier_score']:.4f}"
    )
    return result


def predict_upcoming(
    model_artifact_dir: str,
    upcoming_features: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for upcoming games using a saved model.

    Args:
        model_artifact_dir: Path to experiment artifact directory
        upcoming_features: DataFrame with same feature columns as training

    Returns:
        DataFrame with model predictions added.
    """
    artifact_path = Path(model_artifact_dir) / "model.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No model artifact at {artifact_path}")

    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    X = upcoming_features[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.predict(X_scaled).astype(float)

    result = upcoming_features.copy()
    result["model_prob"] = probs
    return result
