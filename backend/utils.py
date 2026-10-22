"""
Dataset preparation, statistics, and experiment aggregation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize


EXPERIMENT_LOG_PATH = Path(__file__).with_name("experiment_log.jsonl")


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_wine_dataset(random_state: int = 42) -> DatasetBundle:
    dataset = load_wine()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.15,
        stratify=dataset.target,
        random_state=random_state,
    )
    validation_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        stratify=y_train_val,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DatasetBundle(
        X_train=X_train_scaled.astype(np.float32),
        X_val=X_val_scaled.astype(np.float32),
        X_test=X_test_scaled.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=list(dataset.feature_names),
        target_names=list(dataset.target_names),
    )


def dataset_summary(bundle: DatasetBundle) -> dict:
    return {
        "name": "UCI Wine Dataset",
        "total_samples": int(len(bundle.y_train) + len(bundle.y_val) + len(bundle.y_test)),
        "train_samples": int(len(bundle.y_train)),
        "validation_samples": int(len(bundle.y_val)),
        "test_samples": int(len(bundle.y_test)),
        "features": int(bundle.X_train.shape[1]),
        "classes": int(len(bundle.target_names)),
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "split": "70/15/15 stratified",
        "preprocessing": ["StandardScaler"],
    }


def mean_std_ci(values: list[float]) -> dict:
    samples = np.array(values, dtype=float)
    mean = float(samples.mean()) if samples.size else 0.0
    std = float(samples.std(ddof=1)) if samples.size > 1 else 0.0
    if samples.size > 1:
        critical = float(t.ppf(0.975, df=samples.size - 1))
        margin = critical * std / np.sqrt(samples.size)
    else:
        margin = 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - margin),
        "ci95_high": float(mean + margin),
        "values": [float(value) for value in samples.tolist()],
    }


def build_convergence_summary(run_results: list) -> list[dict]:
    max_epochs = max((len(run.history) for run in run_results), default=0)
    convergence = []
    for epoch in range(max_epochs):
        epoch_train_accuracy = []
        epoch_val_accuracy = []
        epoch_train_loss = []
        epoch_val_loss = []
        for run in run_results:
            point = run.history[min(epoch, len(run.history) - 1)]
            epoch_train_accuracy.append(point["train_accuracy"])
            epoch_val_accuracy.append(point["val_accuracy"])
            epoch_train_loss.append(point["train_loss"])
            epoch_val_loss.append(point["val_loss"])
        convergence.append(
            {
                "epoch": epoch + 1,
                "train_accuracy": float(np.mean(epoch_train_accuracy)),
                "val_accuracy": float(np.mean(epoch_val_accuracy)),
                "train_loss": float(np.mean(epoch_train_loss)),
                "val_loss": float(np.mean(epoch_val_loss)),
            }
        )
    return convergence


def build_roc_curve(targets: np.ndarray, probabilities: np.ndarray, target_names: list[str]) -> list[dict]:
    one_hot = label_binarize(targets, classes=list(range(len(target_names))))
    fpr_grid = np.linspace(0.0, 1.0, 100)
    interpolated = {name: np.zeros_like(fpr_grid) for name in target_names}
    auc_scores = {}

    for index, class_name in enumerate(target_names):
        class_fpr, class_tpr, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        interpolated[class_name] = np.interp(fpr_grid, class_fpr, class_tpr)
        auc_scores[class_name] = float(auc(class_fpr, class_tpr))

    roc_points = []
    for idx, fpr_value in enumerate(fpr_grid):
        row = {"fpr": float(fpr_value)}
        for class_name in target_names:
            row[class_name] = float(interpolated[class_name][idx])
            row[f"{class_name}_auc"] = auc_scores[class_name]
        roc_points.append(row)
    return roc_points


def aggregate_run_results(run_results: list, dataset_bundle: DatasetBundle, hyperparameters: dict, label: str) -> dict:
    probabilities = np.mean(np.stack([run.probabilities for run in run_results]), axis=0)
    predictions = probabilities.argmax(axis=1)
    confusion = confusion_matrix(dataset_bundle.y_test, predictions, labels=list(range(len(dataset_bundle.target_names))))

    aggregated = {
        "label": label,
        "hyperparameters": hyperparameters,
        "runs": [
            {
                "seed": run.seed,
                "train_accuracy": run.train_accuracy,
                "val_accuracy": run.val_accuracy,
                "test_accuracy": run.test_accuracy,
                "train_loss": run.train_loss,
                "val_loss": run.val_loss,
                "test_loss": run.test_loss,
                "precision": run.precision,
                "recall": run.recall,
                "f1_score": run.f1_score,
                "training_time": run.training_time,
                "epochs_ran": run.epochs_ran,
                "best_epoch": run.best_epoch,
            }
            for run in run_results
        ],
        "metrics": {
            "train_accuracy": mean_std_ci([run.train_accuracy for run in run_results]),
            "val_accuracy": mean_std_ci([run.val_accuracy for run in run_results]),
            "test_accuracy": mean_std_ci([run.test_accuracy for run in run_results]),
            "train_loss": mean_std_ci([run.train_loss for run in run_results]),
            "val_loss": mean_std_ci([run.val_loss for run in run_results]),
            "test_loss": mean_std_ci([run.test_loss for run in run_results]),
            "precision": mean_std_ci([run.precision for run in run_results]),
            "recall": mean_std_ci([run.recall for run in run_results]),
            "f1_score": mean_std_ci([run.f1_score for run in run_results]),
            "training_time": mean_std_ci([run.training_time for run in run_results]),
            "epochs_ran": mean_std_ci([run.epochs_ran for run in run_results]),
            "best_epoch": mean_std_ci([run.best_epoch for run in run_results]),
        },
        "convergence": build_convergence_summary(run_results),
        "prediction_distribution": [
            {"name": dataset_bundle.target_names[class_index], "value": int(np.sum(predictions == class_index))}
            for class_index in range(len(dataset_bundle.target_names))
        ],
        "confusion_matrix": {
            "labels": dataset_bundle.target_names,
            "matrix": confusion.tolist(),
        },
        "roc_curve": build_roc_curve(dataset_bundle.y_test, probabilities, dataset_bundle.target_names),
    }
    return aggregated


def compare_experiments(baseline: dict, optimized: dict) -> dict:
    baseline_acc = np.array(baseline["metrics"]["test_accuracy"]["values"], dtype=float)
    optimized_acc = np.array(optimized["metrics"]["test_accuracy"]["values"], dtype=float)
    baseline_loss = np.array(baseline["metrics"]["test_loss"]["values"], dtype=float)
    optimized_loss = np.array(optimized["metrics"]["test_loss"]["values"], dtype=float)
    baseline_epoch = np.array(baseline["metrics"]["best_epoch"]["values"], dtype=float)
    optimized_epoch = np.array(optimized["metrics"]["best_epoch"]["values"], dtype=float)

    acc_test = ttest_rel(optimized_acc, baseline_acc)
    loss_test = ttest_rel(baseline_loss, optimized_loss)
    epoch_test = ttest_rel(baseline_epoch, optimized_epoch)

    baseline_mean_acc = baseline["metrics"]["test_accuracy"]["mean"]
    optimized_mean_acc = optimized["metrics"]["test_accuracy"]["mean"]
    baseline_mean_loss = baseline["metrics"]["test_loss"]["mean"]
    optimized_mean_loss = optimized["metrics"]["test_loss"]["mean"]
    baseline_mean_epoch = baseline["metrics"]["best_epoch"]["mean"]
    optimized_mean_epoch = optimized["metrics"]["best_epoch"]["mean"]

    return {
        "accuracy_gain": float(optimized_mean_acc - baseline_mean_acc),
        "accuracy_gain_percent": float(
            ((optimized_mean_acc - baseline_mean_acc) / baseline_mean_acc) * 100 if baseline_mean_acc else 0.0
        ),
        "loss_reduction": float(baseline_mean_loss - optimized_mean_loss),
        "loss_reduction_percent": float(
            ((baseline_mean_loss - optimized_mean_loss) / baseline_mean_loss) * 100 if baseline_mean_loss else 0.0
        ),
        "convergence_gain_epochs": float(baseline_mean_epoch - optimized_mean_epoch),
        "accuracy_p_value": float(acc_test.pvalue) if np.isfinite(acc_test.pvalue) else 1.0,
        "loss_p_value": float(loss_test.pvalue) if np.isfinite(loss_test.pvalue) else 1.0,
        "convergence_p_value": float(epoch_test.pvalue) if np.isfinite(epoch_test.pvalue) else 1.0,
        "significant_accuracy": bool(optimized_mean_acc > baseline_mean_acc and acc_test.pvalue < 0.05),
        "significant_loss": bool(optimized_mean_loss < baseline_mean_loss and loss_test.pvalue < 0.05),
        "faster_convergence": bool(optimized_mean_epoch < baseline_mean_epoch),
        "improved": bool(optimized_mean_acc > baseline_mean_acc and optimized_mean_loss < baseline_mean_loss),
    }


def build_comparison_chart(baseline: dict, optimized: dict) -> list[dict]:
    return [
        {"name": "Baseline", "accuracy": baseline["metrics"]["test_accuracy"]["mean"] * 100, "loss": baseline["metrics"]["test_loss"]["mean"]},
        {"name": "ABC Optimized", "accuracy": optimized["metrics"]["test_accuracy"]["mean"] * 100, "loss": optimized["metrics"]["test_loss"]["mean"]},
    ]


def export_results_rows(results: dict) -> list[dict]:
    rows = []
    for key in ["baseline", "optimized"]:
        experiment = results.get(key)
        if not experiment:
            continue
        rows.append(
            {
                "experiment": experiment["label"],
                "test_accuracy_mean": experiment["metrics"]["test_accuracy"]["mean"],
                "test_accuracy_std": experiment["metrics"]["test_accuracy"]["std"],
                "test_loss_mean": experiment["metrics"]["test_loss"]["mean"],
                "f1_mean": experiment["metrics"]["f1_score"]["mean"],
                "training_time_mean": experiment["metrics"]["training_time"]["mean"],
                "hyperparameters": json.dumps(experiment["hyperparameters"]),
            }
        )
    if results.get("comparison"):
        rows.append({"experiment": "comparison", **results["comparison"]})
    return rows


def write_experiment_log(entry: dict) -> None:
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def results_to_dataframe(results: dict) -> pd.DataFrame:
    return pd.DataFrame(export_results_rows(results))
