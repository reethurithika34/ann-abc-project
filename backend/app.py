"""
Flask API for research-grade ANN + ABC experimentation.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

from experiment_manager import ExperimentManager


app = Flask(__name__)
CORS(app)
manager = ExperimentManager()


@app.get("/api/health")
def health():
    snapshot = manager.snapshot()
    return jsonify(
        {
            "status": "healthy",
            "message": "Research ANN-ABC backend ready",
            "running": snapshot["running"],
            "stage": snapshot["stage"],
        }
    ), 200


@app.get("/api/dataset-info")
def dataset_info():
    return jsonify({"success": True, "dataset": manager.snapshot()["results"]["dataset"]}), 200


@app.post("/run-baseline")
@app.post("/api/run-baseline")
def run_baseline():
    payload = request.get_json(silent=True) or {}
    started = manager.start_baseline(payload)
    if not started:
        return jsonify({"success": False, "message": "Another experiment is already running."}), 409
    return jsonify({"success": True, "message": "Baseline experiment started."}), 202


@app.post("/run-abc-optimization")
@app.post("/api/run-abc-optimization")
def run_abc_optimization():
    payload = request.get_json(silent=True) or {}
    started = manager.start_optimization(payload)
    if not started:
        return jsonify({"success": False, "message": "Another experiment is already running."}), 409
    return jsonify({"success": True, "message": "ABC optimization experiment started."}), 202


@app.get("/get-results")
@app.get("/api/get-results")
def get_results():
    snapshot = manager.snapshot()
    return jsonify(
        {
            "success": True,
            "running": snapshot["running"],
            "active_job": snapshot["active_job"],
            "progress": snapshot["progress"],
            "stage": snapshot["stage"],
            "logs": snapshot["logs"],
            "results": snapshot["results"],
        }
    ), 200


@app.get("/get-convergence-data")
@app.get("/api/get-convergence-data")
def get_convergence_data():
    snapshot = manager.snapshot()
    return jsonify({"success": True, "convergence": snapshot["convergence"]}), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
