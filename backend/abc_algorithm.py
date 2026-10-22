"""
Artificial Bee Colony optimizer for ANN hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10

import numpy as np

from ann_model import ANNHyperparameters, train_single_run


@dataclass(frozen=True)
class ABCConfig:
    population_size: int = 12
    iterations: int = 10
    limit: int = 4
    search_runs: int = 1
    fitness_lambda: float = 0.25
    overfit_lambda: float = 0.5
    no_improvement_patience: int = 4
    random_state: int = 42


class ArtificialBeeColonyOptimizer:
    hidden_size_choices = [32, 64, 128, 256]
    batch_size_choices = [8, 16, 32]
    activation_choices = ["relu", "tanh"]
    init_choices = ["xavier", "he"]

    def __init__(self, config: ABCConfig, dataset_bundle, log_hook=None, progress_hook=None) -> None:
        self.config = config
        self.dataset_bundle = dataset_bundle
        self.log_hook = log_hook
        self.progress_hook = progress_hook
        self.rng = np.random.default_rng(config.random_state)
        self.population = []
        self.fitness = []
        self.trials = []
        self.best_solution = None
        self.best_fitness = float("-inf")
        self.convergence = []

    def _log(self, message: str) -> None:
        if self.log_hook:
            self.log_hook(message)

    def _random_solution(self) -> dict:
        return {
            "hidden_idx": int(self.rng.integers(0, len(self.hidden_size_choices))),
            "num_layers": int(self.rng.integers(1, 4)),
            "learning_rate": float(10 ** self.rng.uniform(log10(1e-4), log10(5e-2))),
            "batch_idx": int(self.rng.integers(0, len(self.batch_size_choices))),
            "activation_idx": int(self.rng.integers(0, len(self.activation_choices))),
            "dropout": float(self.rng.uniform(0.0, 0.5)),
            "init_idx": int(self.rng.integers(0, len(self.init_choices))),
        }

    def _clip_solution(self, solution: dict) -> dict:
        return {
            "hidden_idx": int(np.clip(round(solution["hidden_idx"]), 0, len(self.hidden_size_choices) - 1)),
            "num_layers": int(np.clip(round(solution["num_layers"]), 1, 3)),
            "learning_rate": float(np.clip(solution["learning_rate"], 1e-4, 5e-2)),
            "batch_idx": int(np.clip(round(solution["batch_idx"]), 0, len(self.batch_size_choices) - 1)),
            "activation_idx": int(np.clip(round(solution["activation_idx"]), 0, len(self.activation_choices) - 1)),
            "dropout": float(np.clip(solution["dropout"], 0.0, 0.5)),
            "init_idx": int(np.clip(round(solution["init_idx"]), 0, len(self.init_choices) - 1)),
        }

    def _decode(self, solution: dict) -> ANNHyperparameters:
        return ANNHyperparameters(
            hidden_size=self.hidden_size_choices[solution["hidden_idx"]],
            num_layers=solution["num_layers"],
            activation=self.activation_choices[solution["activation_idx"]],
            learning_rate=solution["learning_rate"],
            batch_size=self.batch_size_choices[solution["batch_idx"]],
            dropout=solution["dropout"],
            weight_init=self.init_choices[solution["init_idx"]],
            max_epochs=30,
            patience=5,
        )

    def _neighbor(self, index: int) -> dict:
        partner_index = int(self.rng.choice([idx for idx in range(len(self.population)) if idx != index]))
        current = self.population[index]
        partner = self.population[partner_index]
        phi = self.rng.uniform(-1.0, 1.0, size=7)
        neighbor = {
            "hidden_idx": current["hidden_idx"] + phi[0] * (current["hidden_idx"] - partner["hidden_idx"]),
            "num_layers": current["num_layers"] + phi[1] * (current["num_layers"] - partner["num_layers"]),
            "learning_rate": current["learning_rate"] + phi[2] * (current["learning_rate"] - partner["learning_rate"]),
            "batch_idx": current["batch_idx"] + phi[3] * (current["batch_idx"] - partner["batch_idx"]),
            "activation_idx": current["activation_idx"] + phi[4] * (current["activation_idx"] - partner["activation_idx"]),
            "dropout": current["dropout"] + phi[5] * (current["dropout"] - partner["dropout"]),
            "init_idx": current["init_idx"] + phi[6] * (current["init_idx"] - partner["init_idx"]),
        }
        return self._clip_solution(neighbor)

    def _fitness(self, solution: dict) -> tuple[float, dict]:
        params = self._decode(solution)
        seeds = [self.config.random_state + offset * 13 for offset in range(self.config.search_runs)]
        runs = [
            train_single_run(params, self.dataset_bundle, seed=seed, log_hook=None)
            for seed in seeds
        ]
        mean_train_acc = float(np.mean([run.train_accuracy for run in runs]))
        mean_val_acc = float(np.mean([run.val_accuracy for run in runs]))
        mean_val_loss = float(np.mean([run.val_loss for run in runs]))
        overfit_gap = max(0.0, mean_train_acc - mean_val_acc)
        fitness = mean_val_acc - (self.config.fitness_lambda * mean_val_loss) - (self.config.overfit_lambda * overfit_gap)
        metrics = {
            "fitness": float(fitness),
            "mean_val_accuracy": mean_val_acc,
            "mean_val_loss": mean_val_loss,
            "overfit_gap": float(overfit_gap),
            "params": params.serialize(),
        }
        return float(fitness), metrics

    def optimize(self) -> tuple[ANNHyperparameters, list[dict]]:
        self.population = [self._random_solution() for _ in range(self.config.population_size)]
        evaluated = [self._fitness(solution) for solution in self.population]
        self.fitness = [score for score, _ in evaluated]
        metadata = [meta for _, meta in evaluated]
        self.trials = [0 for _ in range(self.config.population_size)]
        no_improvement = 0

        best_idx = int(np.argmax(self.fitness))
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()
        best_meta = metadata[best_idx]

        for iteration in range(1, self.config.iterations + 1):
            for index in range(self.config.population_size):
                neighbor = self._neighbor(index)
                neighbor_fitness, neighbor_meta = self._fitness(neighbor)
                if neighbor_fitness > self.fitness[index]:
                    self.population[index] = neighbor
                    self.fitness[index] = neighbor_fitness
                    metadata[index] = neighbor_meta
                    self.trials[index] = 0
                else:
                    self.trials[index] += 1

            probabilities = np.array(self.fitness, dtype=float)
            probabilities = probabilities - probabilities.min() + 1e-6
            probabilities = probabilities / probabilities.sum()

            for _ in range(self.config.population_size):
                index = int(self.rng.choice(self.config.population_size, p=probabilities))
                neighbor = self._neighbor(index)
                neighbor_fitness, neighbor_meta = self._fitness(neighbor)
                if neighbor_fitness > self.fitness[index]:
                    self.population[index] = neighbor
                    self.fitness[index] = neighbor_fitness
                    metadata[index] = neighbor_meta
                    self.trials[index] = 0
                else:
                    self.trials[index] += 1

            for index, trial_count in enumerate(self.trials):
                if trial_count >= self.config.limit:
                    self.population[index] = self._random_solution()
                    self.fitness[index], metadata[index] = self._fitness(self.population[index])
                    self.trials[index] = 0

            best_idx = int(np.argmax(self.fitness))
            current_best = self.fitness[best_idx]
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = self.population[best_idx].copy()
                best_meta = metadata[best_idx]
                no_improvement = 0
            else:
                no_improvement += 1

            record = {
                "iteration": iteration,
                "best_fitness": float(self.best_fitness),
                "best_val_accuracy": float(best_meta["mean_val_accuracy"]),
                "best_val_loss": float(best_meta["mean_val_loss"]),
                "mean_population_fitness": float(np.mean(self.fitness)),
            }
            self.convergence.append(record)
            if self.progress_hook:
                self.progress_hook((iteration / self.config.iterations) * 0.8)  # 80% of the progress for ABC iterations
            self._log(
                f"ABC iteration {iteration}/{self.config.iterations}: "
                f"best_fitness={record['best_fitness']:.4f}, "
                f"val_acc={record['best_val_accuracy']:.4f}, "
                f"val_loss={record['best_val_loss']:.4f}"
            )

            if no_improvement >= self.config.no_improvement_patience:
                self._log("ABC early stopping triggered due to no further improvement.")
                break

        if self.best_solution is None:
            raise RuntimeError("ABC optimization failed to produce a solution.")
        return self._decode(self.best_solution), self.convergence
