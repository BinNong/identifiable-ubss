import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import (
    align_sources_to_truth,
    estimate_observation_noise_std,
    recover_sources,
    score_directions,
    score_source_recovery,
)
from pnl import (
    calibrate_two_channel_pnl,
    estimate_columns_from_linearized,
    forward_nonlinearity,
    generate_two_regime_pnl,
    inverse_nonlinearity,
)


def evaluate_regimes(mixing, regimes, include_recovery=True):
    calibrated = []
    calibrated_oracle_labels = []
    calibration_times = []
    for regime in regimes:
        start = time.perf_counter()
        result = calibrate_two_channel_pnl(
            regime["observations"],
            2,
            zone_length=regime["zone_length"],
        )
        calibration_times.append(time.perf_counter() - start)
        calibrated.append(result)
        oracle_label_result = calibrate_two_channel_pnl(
            regime["observations"],
            2,
            zone_length=regime["zone_length"],
            labels=regime["labels"],
        )
        calibrated_oracle_labels.append(oracle_label_result)

    true_sources = np.concatenate([
        regime["sources"][result["mask"]]
        for regime, result in zip(regimes, calibrated)
    ], axis=0)
    raw_values = np.concatenate([
        regime["observations"][result["mask"]]
        for regime, result in zip(regimes, calibrated)
    ], axis=0)
    oracle_values = np.concatenate([
        inverse_nonlinearity(
            regime["observations"],
            regime["forward_coefficients"],
        )[result["mask"]]
        for regime, result in zip(regimes, calibrated)
    ], axis=0)
    blind_values = np.concatenate([result["linearized"] for result in calibrated], axis=0)
    blind_oracle_label_values = np.concatenate([
        result["linearized"] for result in calibrated_oracle_labels
    ], axis=0)
    window_length = regimes[0]["zone_length"]

    output = {}
    for method, values in [
        ("joint_blind", blind_values),
        ("joint_blind_oracle_labels", blind_oracle_label_values),
        ("joint_oracle", oracle_values),
        ("pooled_no_calibration", raw_values),
    ]:
        start = time.perf_counter()
        columns, _ = estimate_columns_from_linearized(values, 3)
        record = score_directions(mixing, columns)
        if include_recovery:
            estimated_noise = estimate_observation_noise_std(values, window_length, 1)
            estimates, _, _ = recover_sources(values, columns, 1, noise_std=estimated_noise)
            estimates = align_sources_to_truth(mixing, columns, estimates)
            record.update(score_source_recovery(estimates, true_sources))
            record["estimated_recovery_noise_std"] = estimated_noise
        record["runtime_seconds"] = time.perf_counter() - start
        output[method] = record

    coefficient_errors = []
    cluster_scores = []
    calibration_records = []
    for regime, result in zip(regimes, calibrated):
        coefficient_errors.append(np.sqrt(np.mean(
            (result["inverse_coefficients"] - regime["forward_coefficients"]) ** 2
        )))
        cluster_scores.append(adjusted_rand_score(
            regime["labels"][result["mask"]],
            result["labels"],
        ))
        calibration_records.append({
            "regime": int(regime["regime"]),
            "estimated_coefficients": result["inverse_coefficients"].tolist(),
            "objective_cost": result["objective_cost"],
            "zero_calibration_cost": result["zero_calibration_cost"],
            "coefficient_bound_hit": result["coefficient_bound_hit"],
            "coefficient_bound_hit_channels": result["coefficient_bound_hit_channels"],
            "line_errors": result["line_errors"].tolist(),
        })
        columns, _ = estimate_columns_from_linearized(result["linearized"], 2)
        output[f"separate_regime_{regime['regime']}"] = score_directions(mixing, columns)

    output["joint_blind"]["coefficient_rmse"] = float(np.mean(coefficient_errors))
    output["joint_blind"]["cluster_ari"] = float(np.mean(cluster_scores))
    output["joint_blind"]["coefficient_bound_hit_rate"] = float(np.mean([
        record["coefficient_bound_hit"] for record in calibration_records
    ]))
    output["joint_blind"]["calibration_objective_relative_reduction"] = float(np.mean([
        (record["zero_calibration_cost"] - record["objective_cost"])
        / (record["zero_calibration_cost"] + 1e-12)
        for record in calibration_records
    ]))
    output["joint_blind"]["calibration_records"] = calibration_records
    output["joint_blind"]["calibration_runtime_seconds"] = float(np.sum(calibration_times))
    output["joint_blind"]["runtime_seconds"] += float(np.sum(calibration_times))
    oracle_label_errors = []
    oracle_label_bound_hits = []
    for regime, result in zip(regimes, calibrated_oracle_labels):
        oracle_label_errors.append(float(np.sqrt(np.mean(
            (result["inverse_coefficients"] - regime["forward_coefficients"]) ** 2
        ))))
        oracle_label_bound_hits.append(bool(result["coefficient_bound_hit"]))
    output["joint_blind_oracle_labels"]["coefficient_rmse"] = float(np.mean(oracle_label_errors))
    output["joint_blind_oracle_labels"]["coefficient_bound_hit_rate"] = float(np.mean(oracle_label_bound_hits))
    return output


def summarize_runs(runs):
    groups = {}
    for run in runs:
        key = (run["method"], run["noise_std"], run["nonlinearity_strength"])
        groups.setdefault(key, []).append(run)
    metrics = [
        "mean_direction_cosine",
        "mean_matched_cosine",
        "direction_recovery_0.99",
        "direction_recovery_0.95",
        "coverage_0.99",
        "coverage_0.95",
        "support_accuracy",
        "support_f1",
        "mean_active_correlation",
        "mean_scaled_error",
        "coefficient_rmse",
        "cluster_ari",
        "coefficient_bound_hit_rate",
        "calibration_objective_relative_reduction",
        "runtime_seconds",
    ]
    summary = []
    for (method, noise, strength), records in sorted(groups.items()):
        row = {
            "method": method,
            "noise_std": noise,
            "nonlinearity_strength": strength,
            "n_runs": len(records),
        }
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
                row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def pnl_diagnostics(runs):
    """Bound-hit mechanism analysis for the two-channel calibration failure."""

    def safe_corr(first, second):
        if len(first) < 3 or np.std(first) == 0 or np.std(second) == 0:
            return None
        return float(np.corrcoef(first, second)[0, 1])

    blind = [run for run in runs if run["method"] == "joint_blind"]
    cells = {}
    for run in blind:
        key = (run["noise_std"], run["nonlinearity_strength"])
        cells.setdefault(key, []).append(run)
    diagnostics = {"cells": []}
    for (noise, strength), records in sorted(cells.items()):
        bound_hit = np.array([bool(record["coefficient_bound_hit_rate"]) for record in records])
        objective_reduction = np.array([record["calibration_objective_relative_reduction"] for record in records])
        cosine = np.array([record["mean_direction_cosine"] for record in records])
        ari = np.array([record["cluster_ari"] for record in records])
        rmse = np.array([record["coefficient_rmse"] for record in records])
        cell = {
            "noise_std": noise,
            "nonlinearity_strength": strength,
            "n_cases": len(records),
            "bound_hit_rate": float(np.mean(bound_hit)),
            "correlation_objective_reduction_direction_cosine": safe_corr(objective_reduction, cosine),
            "correlation_cluster_ari_direction_cosine": safe_corr(ari, cosine),
            "correlation_cluster_ari_coefficient_rmse": safe_corr(ari, rmse),
        }
        for flag in [True, False]:
            subset = np.flatnonzero(bound_hit == flag)
            if len(subset):
                cell[f"bound_hit_{flag}"] = {
                    "n_cases": int(len(subset)),
                    "mean_direction_cosine": float(np.mean(cosine[subset])),
                    "mean_cluster_ari": float(np.mean(ari[subset])),
                    "mean_coefficient_rmse": float(np.mean(rmse[subset])),
                    "mean_objective_relative_reduction": float(np.mean(objective_reduction[subset])),
                }
        diagnostics["cells"].append(cell)
    oracle_label = [run for run in runs if run["method"] == "joint_blind_oracle_labels"]
    diagnostics["oracle_label_calibration"] = []
    cells = {}
    for run in oracle_label:
        key = (run["noise_std"], run["nonlinearity_strength"])
        cells.setdefault(key, []).append(run)
    for (noise, strength), records in sorted(cells.items()):
        diagnostics["oracle_label_calibration"].append({
            "noise_std": noise,
            "nonlinearity_strength": strength,
            "n_cases": len(records),
            "mean_direction_cosine": float(np.mean([record["mean_direction_cosine"] for record in records])),
            "mean_coefficient_rmse": float(np.mean([record["coefficient_rmse"] for record in records])),
            "bound_hit_rate": float(np.mean([record["coefficient_bound_hit_rate"] for record in records])),
        })
    return diagnostics


def evaluate_configuration(configuration):
    strength = configuration["nonlinearity_strength"]
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    mixing, regimes = generate_two_regime_pnl(
        n_samples=2500,
        noise_std=0.0,
        seed=seed,
    )
    noise_rng = np.random.default_rng(seed + 991)
    for regime in regimes:
        regime["forward_coefficients"] = regime["forward_coefficients"] * strength
        regime["observations"] = forward_nonlinearity(
            regime["linear"],
            regime["forward_coefficients"],
        )
        regime["observations"] += noise * noise_rng.normal(size=regime["observations"].shape)
    results = evaluate_regimes(mixing, regimes)
    return [{
        "method": method,
        "noise_std": noise,
        "nonlinearity_strength": strength,
        "seed": seed,
        **metrics,
    } for method, metrics in results.items()]


def run_suite(mode, workers):
    if mode == "smoke":
        noises = [0.002, 0.01]
        strengths = [1.0]
        seeds = [20261016]
    elif mode == "development":
        noises = [0.0, 0.002, 0.01, 0.03]
        strengths = [0.5, 1.0, 1.5]
        seeds = list(range(1001, 1006))
    else:
        noises = [0.0, 0.002, 0.01, 0.03]
        strengths = [0.5, 1.0, 1.5]
        seeds = list(range(101001, 101021))
    configurations = [{
        "noise_std": noise,
        "nonlinearity_strength": strength,
        "seed": seed,
    } for strength in strengths for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_configuration(configuration))
            print(f"PNL {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_configuration, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"PNL {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (
        run["nonlinearity_strength"], run["noise_std"], run["seed"], run["method"]
    ))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else f"_{mode}"
    with (results_dir / f"pnl{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"pnl{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"pnl{suffix}_diagnostics.json").open("w") as handle:
        json.dump(pnl_diagnostics(runs), handle, indent=2)
    with (results_dir / f"pnl{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "n_samples_per_regime": 2500,
            "zone_length": 100,
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "development", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
