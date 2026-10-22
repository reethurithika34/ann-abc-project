"""
Background experiment manager with live logs and result snapshots.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import copy
import threading

from abc_algorithm import ABCConfig, ArtificialBeeColonyOptimizer
from ann_model import ANNHyperparameters, train_single_run
from utils import (
    aggregate_run_results,
    build_comparison_chart,
    compare_experiments,
    dataset_summary,
    export_results_rows,
    load_wine_dataset,
    results_to_dataframe,
    write_experiment_log,
)


class ExperimentManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.dataset_bundle = load_wine_dataset()
        self.state = {
            "running": False,
            "active_job": None,
            "progress": 0.0,
            "stage": "idle",
            "logs": [],
            "results": {
                "dataset": dataset_summary(self.dataset_bundle),
                "baseline": None,
                "optimized": None,
                "comparison": None,
                "comparison_chart": [],
                "csv_rows": [],
            },
            "convergence": {"baseline": [], "optimization": [], "abc_search": []},
        }

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.state)

    def _push_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.state["logs"].append(f"[{timestamp}] {message}")
            self.state["logs"] = self.state["logs"][-250:]

    def _set_progress(self, progress: float, stage: str) -> None:
        with self.lock:
            self.state["progress"] = progress
            self.state["stage"] = stage

    def _set_running(self, running: bool, active_job: str | None = None) -> None:
        with self.lock:
            self.state["running"] = running
            self.state["active_job"] = active_job

    def _store_baseline(self, baseline: dict) -> None:
        with self.lock:
            self.state["results"]["baseline"] = baseline
            self.state["convergence"]["baseline"] = baseline["convergence"]

    def _store_optimization(self, optimized: dict, convergence: list[dict], comparison: dict) -> None:
        with self.lock:
            self.state["results"]["optimized"] = optimized
            self.state["results"]["comparison"] = comparison
            self.state["results"]["comparison_chart"] = build_comparison_chart(
                self.state["results"]["baseline"], optimized
            )
            self.state["convergence"]["optimization"] = optimized["convergence"]
            self.state["convergence"]["abc_search"] = convergence
            self.state["results"]["csv_rows"] = export_results_rows(self.state["results"])

    def _evaluate_repeated(self, params: ANNHyperparameters, seeds: list[int], label: str, stage_prefix: str) -> dict:
        runs = []
        for index, seed in enumerate(seeds, start=1):
            self._push_log(f"{label}: run {index}/{len(seeds)} with seed {seed}")
            run = train_single_run(
                params,
                self.dataset_bundle,
                seed=seed,
                log_hook=self._push_log if index == 1 else None,
                log_prefix=f"{stage_prefix} ",
            )
            runs.append(run)
            self._push_log(
                f"{label}: run {index} completed with test_acc={run.test_accuracy:.4f}, "
                f"test_loss={run.test_loss:.4f}, best_epoch={run.best_epoch}"
            )
        return aggregate_run_results(runs, self.dataset_bundle, params.serialize(), label)

    def start_baseline(self, payload: dict) -> bool:
        with self.lock:
            if self.state["running"]:
                return False
        thread = threading.Thread(target=self._run_baseline_job, args=(payload,), daemon=True)
        thread.start()
        return True

    def start_optimization(self, payload: dict) -> bool:
        with self.lock:
            if self.state["running"]:
                return False
        thread = threading.Thread(target=self._run_optimization_job, args=(payload,), daemon=True)
        thread.start()
        return True

    def _run_baseline_job(self, payload: dict) -> None:
        self._set_running(True, "baseline")
        self._set_progress(0.02, "Preparing baseline experiment")
        self._push_log("Baseline experiment started.")
        try:
            params = ANNHyperparameters(
                hidden_size=int(payload.get("hidden_size", 64)),
                num_layers=1,
                activation="relu",
                learning_rate=float(payload.get("learning_rate", 0.001)),
                batch_size=int(payload.get("batch_size", 16)),
                dropout=0.0,
                weight_init="he",
                max_epochs=int(payload.get("max_epochs", 50)),
                patience=int(payload.get("patience", 8)),
            )
            seeds = payload.get("seeds", [42, 52, 62])
            baseline = self._evaluate_repeated(params, seeds, "Baseline ANN", "baseline")
            self._store_baseline(baseline)
            self._set_progress(1.0, "Baseline complete")
            self._push_log("Baseline experiment completed.")
            write_experiment_log({"type": "baseline", "hyperparameters": params.serialize(), "results": baseline["metrics"]})
        except Exception as exc:
            self._push_log(f"Baseline experiment failed: {exc}")
            raise
        finally:
            self._set_running(False, None)

    def _run_optimization_job(self, payload: dict) -> None:
        self._set_running(True, "optimization")
        self._push_log("Optimization experiment started.")
        try:
            baseline = self.snapshot()["results"]["baseline"]
            if baseline is None:
                self._push_log("No baseline found. Running default baseline first.")
                baseline_params = {"hidden_size": 64, "learning_rate": 0.001, "batch_size": 16, "max_epochs": 50, "patience": 8}
                baseline = self._evaluate_repeated(
                    ANNHyperparameters(
                        hidden_size=baseline_params["hidden_size"],
                        num_layers=1,
                        activation="relu",
                        learning_rate=baseline_params["learning_rate"],
                        batch_size=baseline_params["batch_size"],
                        dropout=0.0,
                        weight_init="he",
                        max_epochs=baseline_params["max_epochs"],
                        patience=baseline_params["patience"],
                    ),
                    [42, 52, 62],
                    "Baseline ANN",
                    "baseline",
                )
                self._store_baseline(baseline)

            population_size = int(payload.get("population_size", 12))
            iterations = int(payload.get("iterations", 10))
            limit = int(payload.get("limit", 4))
            search_runs = int(payload.get("search_runs", 1))
            final_seeds = payload.get("final_seeds", [42, 52, 62])
            max_retries = int(payload.get("auto_retry_max", 2))
            best_attempt = None

            for attempt in range(1, max_retries + 1):
                self._set_progress(0.1 + ((attempt - 1) / max_retries) * 0.4, f"ABC search attempt {attempt}")
                self._push_log(f"ABC attempt {attempt}/{max_retries} started.")
                optimizer = ArtificialBeeColonyOptimizer(
                    ABCConfig(
                        population_size=population_size,
                        iterations=iterations,
                        limit=limit,
                        search_runs=search_runs,
                        random_state=41 + attempt * 11,
                    ),
                    self.dataset_bundle,
                    log_hook=self._push_log,
                    progress_hook=lambda p: self._set_progress(0.1 + ((attempt - 1) / max_retries) * 0.4 + p * 0.4, f"ABC search attempt {attempt}"),
                )
                best_params, abc_convergence = optimizer.optimize()
                self._push_log(f"ABC attempt {attempt}: best parameters = {best_params.serialize()}")
                self._set_progress(0.55 + (attempt / max_retries) * 0.25, f"Final evaluation attempt {attempt}")
                optimized = self._evaluate_repeated(best_params, final_seeds, f"ABC Optimized ANN attempt {attempt}", "optimized")
                comparison = compare_experiments(baseline, optimized)
                candidate = {"optimized": optimized, "comparison": comparison, "abc_convergence": abc_convergence, "params": best_params.serialize()}

                if best_attempt is None or comparison["accuracy_gain"] > best_attempt["comparison"]["accuracy_gain"]:
                    best_attempt = candidate

                self._push_log(
                    f"Attempt {attempt} summary: accuracy_gain={comparison['accuracy_gain_percent']:.2f}%, "
                    f"loss_reduction={comparison['loss_reduction_percent']:.2f}%, "
                    f"p_acc={comparison['accuracy_p_value']:.4f}, p_loss={comparison['loss_p_value']:.4f}"
                )

                if comparison["improved"]:
                    self._push_log("Improved configuration found. Stopping auto-retry loop.")
                    break

                self._push_log("Improvement criteria not met yet. Re-running ABC with a new search seed.")

            if best_attempt is None:
                raise RuntimeError("Optimization did not produce any valid candidate.")

            self._store_optimization(best_attempt["optimized"], best_attempt["abc_convergence"], best_attempt["comparison"])
            self._set_progress(1.0, "Optimization complete")
            self._push_log("Optimization experiment completed.")
            write_experiment_log(
                {
                    "type": "optimization",
                    "hyperparameters": best_attempt["params"],
                    "comparison": best_attempt["comparison"],
                    "optimized_metrics": best_attempt["optimized"]["metrics"],
                }
            )
            dataframe = results_to_dataframe(self.snapshot()["results"])
            with self.lock:
                self.state["results"]["csv_rows"] = dataframe.to_dict(orient="records")
        except Exception as exc:
            self._push_log(f"Optimization experiment failed: {exc}")
            raise
        finally:
            self._set_running(False, None)
