"""
Sklearn MLP-based ANN training for reproducible wine experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
import copy
import random

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class ANNHyperparameters:
    hidden_size: int = 64
    num_layers: int = 1
    activation: str = "relu"
    learning_rate: float = 0.001
    batch_size: int = 16
    dropout: float = 0.0
    weight_init: str = "xavier"
    max_epochs: int = 50
    patience: int = 8

    def hidden_layers(self) -> list[int]:
        return [self.hidden_size for _ in range(self.num_layers)]

    def serialize(self) -> dict:
        return {**asdict(self), "hidden_layers": self.hidden_layers()}


@dataclass
class SingleRunResult:
    seed: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_loss: float
    val_loss: float
    test_loss: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    epochs_ran: int
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    probabilities: np.ndarray | None = None
    predictions: np.ndarray | None = None
    targets: np.ndarray | None = None
    predicted_distribution: list[int] = field(default_factory=list)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _evaluate(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    loss = float(log_loss(targets, probabilities, labels=[0, 1, 2]))
    accuracy = float(accuracy_score(targets, predictions))
    return {
        "accuracy": accuracy,
        "loss": loss,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def train_single_run(
    params: ANNHyperparameters,
    dataset_bundle,
    seed: int,
    log_hook=None,
    log_prefix: str = "",
) -> SingleRunResult:
    set_global_seed(seed)

    model = MLPClassifier(
        hidden_layer_sizes=tuple(params.hidden_layers()),
        activation=params.activation,
        learning_rate_init=params.learning_rate,
        batch_size=params.batch_size,
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=seed,
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    started_at = perf_counter()

    for epoch in range(1, params.max_epochs + 1):
        model.fit(dataset_bundle.X_train, dataset_bundle.y_train)

        train_probs = model.predict_proba(dataset_bundle.X_train)
        train_eval = _evaluate(train_probs, dataset_bundle.y_train)

        val_probs = model.predict_proba(dataset_bundle.X_val)
        val_eval = _evaluate(val_probs, dataset_bundle.y_val)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_eval["loss"]),
                "train_accuracy": float(train_eval["accuracy"]),
                "val_loss": float(val_eval["loss"]),
                "val_accuracy": float(val_eval["accuracy"]),
            }
        )

        if log_hook and (epoch == 1 or epoch % 5 == 0 or epoch == params.max_epochs):
            log_hook(
                f"{log_prefix}epoch {epoch}: train_acc={train_eval['accuracy']:.4f}, "
                f"val_acc={val_eval['accuracy']:.4f}, val_loss={val_eval['loss']:.4f}"
            )

        if val_eval["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params.patience:
            break

    if best_epoch > 0:
        model = best_state

    training_time = perf_counter() - started_at
    train_eval = _evaluate(model.predict_proba(dataset_bundle.X_train), dataset_bundle.y_train)
    val_eval = _evaluate(model.predict_proba(dataset_bundle.X_val), dataset_bundle.y_val)
    test_eval = _evaluate(model.predict_proba(dataset_bundle.X_test), dataset_bundle.y_test)

    true_targets = dataset_bundle.y_test
    predictions = test_eval["predictions"]

    true_positive = np.zeros(len(dataset_bundle.target_names), dtype=float)
    predicted_positive = np.zeros(len(dataset_bundle.target_names), dtype=float)
    actual_positive = np.zeros(len(dataset_bundle.target_names), dtype=float)
    for class_index in range(len(dataset_bundle.target_names)):
        true_positive[class_index] = np.sum((predictions == class_index) & (true_targets == class_index))
        predicted_positive[class_index] = np.sum(predictions == class_index)
        actual_positive[class_index] = np.sum(true_targets == class_index)

    precision_per_class = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive > 0,
    )
    recall_per_class = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(actual_positive),
        where=actual_positive > 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(precision_per_class),
        where=(precision_per_class + recall_per_class) > 0,
    )

    return SingleRunResult(
        seed=seed,
        train_accuracy=float(train_eval["accuracy"]),
        val_accuracy=float(val_eval["accuracy"]),
        test_accuracy=float(test_eval["accuracy"]),
        train_loss=float(train_eval["loss"]),
        val_loss=float(val_eval["loss"]),
        test_loss=float(test_eval["loss"]),
        precision=float(np.mean(precision_per_class)),
        recall=float(np.mean(recall_per_class)),
        f1_score=float(np.mean(f1_per_class)),
        training_time=float(training_time),
        epochs_ran=len(history),
        best_epoch=best_epoch or len(history),
        history=history,
        probabilities=test_eval["probabilities"],
        predictions=predictions,
        targets=true_targets,
        predicted_distribution=[int(np.sum(predictions == class_index)) for class_index in range(len(dataset_bundle.target_names))],
    )
