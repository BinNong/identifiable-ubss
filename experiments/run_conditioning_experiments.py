import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import support_basis
from pnl import forward_nonlinearity, inverse_nonlinearity
from run_coverage_experiments import DIRECTION_CONFIG, calibrate_regimes, run_pipeline
from subspace import _subspace_distance, generate_running_example


def measure_conditioning(mixing):
    from scipy.linalg import orth
    from coverage import intersect_subspaces

    left = support_basis(mixing, (0, 4))
    right = support_basis(mixing, (1, 4))
    intersection = intersect_subspaces(left, right)
    left_perp = orth((np.eye(mixing.shape[0]) - intersection @ intersection.T) @ left)
    residual_map = (np.eye(mixing.shape[0]) - right @ right.T) @ left_perp
    gamma = float(np.min(np.linalg.svd(residual_map, compute_uv=False)))
    return {
        "accessible_condition_number": float(np.linalg.cond(mixing[:, :4])),
        "intersection_gamma": gamma,
    }


def evaluate_case(configuration):
    threshold = configuration["min_calibration_entry"]
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    mixing, regimes = generate_running_example(
        blocks_per_support=8,
        block_length=80,
        noise_std=0.0,
        seed=seed,
        min_calibration_entry=threshold,
    )
    coefficients = [
        np.array([0.45, -0.25, 0.55, -0.35]),
        np.array([-0.30, 0.50, -0.20, 0.60]),
    ]
    noise_rng = np.random.default_rng(seed + 313)
    for regime in regimes:
        regime["linear_observations"] = regime["observations"].copy()
        regime["forward_coefficients"] = coefficients[regime["regime"]]
        regime["observations"] = forward_nonlinearity(
            regime["linear_observations"], regime["forward_coefficients"]
        )
        regime["observations"] += noise * noise_rng.normal(size=regime["observations"].shape)
    blind_values, calibration_records = calibrate_regimes(regimes, 50, seed)
    oracle_values = [
        inverse_nonlinearity(regime["observations"], regime["forward_coefficients"])
        for regime in regimes
    ]
    true_sources = np.concatenate([regime["sources"] for regime in regimes], axis=0)
    cross_distance = _subspace_distance(
        support_basis(mixing, (0, 4)), support_basis(mixing, (1, 4))
    )
    geometry = measure_conditioning(mixing)
    output = []
    score_rules = configuration.get("score_rules", [DIRECTION_CONFIG["score_rule"]])
    for method, values in [("joint_blind_lattice", blind_values), ("joint_oracle_lattice", oracle_values)]:
        for score_rule in score_rules:
            record = run_pipeline(
                mixing,
                np.concatenate(values, axis=0),
                true_sources,
                80,
                "lattice",
                recover=configuration.get("recover", True),
                score_rule=score_rule,
            )
            record.update({
                "method": method,
                "direction_score_rule": score_rule,
                "min_calibration_entry": threshold,
                "actual_minimum_calibration_entry": float(np.min(np.abs(mixing[:, :4]))),
                "cross_subspace_distance": cross_distance,
                **geometry,
                "noise_std": noise,
                "seed": seed,
                "subspace_prototype_match": bool(
                    record["n_prototypes"] == 6 and record["n_rank_one_prototypes"] == 4
                ),
                "complete_direction_recovery": bool(record["coverage_0.99"] == 5),
            })
            if method == "joint_blind_lattice":
                record["calibration_coefficient_rmse"] = float(np.mean([
                    item["coefficient_rmse"] for item in calibration_records
                ]))
                record["calibration_relative_error"] = float(np.mean([
                    item["relative_linearization_error"] for item in calibration_records
                ]))
            output.append(record)
    return output


def summarize_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["method"], run["min_calibration_entry"], run["noise_std"]), []).append(run)
    metrics = [
        "mean_direction_cosine",
        "direction_recovery_0.99",
        "coverage_0.99",
        "support_f1",
        "mean_active_correlation",
        "calibration_coefficient_rmse",
        "calibration_relative_error",
        "actual_minimum_calibration_entry",
        "cross_subspace_distance",
        "accessible_condition_number",
        "intersection_gamma",
        "runtime_seconds",
    ]
    summary = []
    for (method, threshold, noise), records in sorted(groups.items()):
        row = {
            "method": method,
            "min_calibration_entry": threshold,
            "noise_std": noise,
            "n_runs": len(records),
        }
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
                row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        values = [
            record["cosines_by_source"][4] if record["cosines_by_source"][4] is not None else 0.0
            for record in records
        ]
        row["source_5_cosine_mean"] = float(np.mean(values))
        row["source_5_recovery_0.99"] = float(np.mean(np.asarray(values) >= 0.99))
        complete = np.asarray([record["complete_direction_recovery"] for record in records], dtype=float)
        row["complete_direction_recovery_rate"] = float(np.mean(complete))
        row["complete_direction_recovery_ci95"] = float(
            t.ppf(0.975, len(records) - 1) * np.std(complete, ddof=1) / np.sqrt(len(records))
        ) if len(records) > 1 else 0.0
        row["subspace_prototype_match_rate"] = float(np.mean([
            record["subspace_prototype_match"] for record in records
        ]))
        summary.append(row)
    return summary


def conditioning_diagnostics(runs):
    diagnostics = {
        "calibration_error_threshold": 0.05,
        "expected_prototypes": 6,
        "expected_rank_one_prototypes": 4,
        "distance_bins": [],
        "failure_attribution": [],
    }
    for noise in sorted({run["noise_std"] for run in runs}):
        blind = [run for run in runs if run["method"] == "joint_blind_lattice" and run["noise_std"] == noise]
        oracle_lookup = {
            (run["min_calibration_entry"], run["seed"]): run
            for run in runs
            if run["method"] == "joint_oracle_lattice" and run["noise_std"] == noise
        }
        distances = np.asarray([run["cross_subspace_distance"] for run in blind])
        edges = np.quantile(distances, [0.0, 0.25, 0.5, 0.75, 1.0])
        assignments = np.digitize(distances, edges[1:-1], right=True)
        for bin_index in range(4):
            records = [record for record, assignment in zip(blind, assignments) if assignment == bin_index]
            if not records:
                continue
            paired_oracle = [oracle_lookup[(record["min_calibration_entry"], record["seed"])] for record in records]
            diagnostics["distance_bins"].append({
                "noise_std": noise,
                "bin": bin_index + 1,
                "distance_min": float(min(record["cross_subspace_distance"] for record in records)),
                "distance_max": float(max(record["cross_subspace_distance"] for record in records)),
                "median_intersection_gamma": float(np.median([record["intersection_gamma"] for record in records])),
                "median_accessible_condition_number": float(np.median([record["accessible_condition_number"] for record in records])),
                "n_cases": len(records),
                "blind_complete_success_rate": float(np.mean([record["complete_direction_recovery"] for record in records])),
                "oracle_complete_success_rate": float(np.mean([record["complete_direction_recovery"] for record in paired_oracle])),
                "median_blind_calibration_relative_error": float(np.median([record["calibration_relative_error"] for record in records])),
            })
        attribution = {
            "noise_std": noise,
            "n_cases": len(blind),
            "both_successes": 0,
            "blind_only_failures": 0,
            "oracle_only_failures": 0,
            "both_failures": 0,
            "blind_failures": 0,
            "shared_direction_stage_failures": 0,
            "calibration_associated_failures": 0,
            "subspace_clustering_failures": 0,
            "intersection_selection_failures": 0,
            "oracle_failures": 0,
            "oracle_prototype_mismatch_failures": 0,
        }
        for record in blind:
            oracle = oracle_lookup[(record["min_calibration_entry"], record["seed"])]
            blind_failed = not record["complete_direction_recovery"]
            oracle_failed = not oracle["complete_direction_recovery"]
            if not blind_failed and not oracle_failed:
                attribution["both_successes"] += 1
            elif blind_failed and not oracle_failed:
                attribution["blind_only_failures"] += 1
            elif not blind_failed and oracle_failed:
                attribution["oracle_only_failures"] += 1
            else:
                attribution["both_failures"] += 1
            if oracle_failed:
                attribution["oracle_failures"] += 1
                if not oracle["subspace_prototype_match"]:
                    attribution["oracle_prototype_mismatch_failures"] += 1
            if not blind_failed:
                continue
            attribution["blind_failures"] += 1
            if oracle_failed:
                attribution["shared_direction_stage_failures"] += 1
            elif record["calibration_relative_error"] > diagnostics["calibration_error_threshold"]:
                attribution["calibration_associated_failures"] += 1
            elif not record["subspace_prototype_match"]:
                attribution["subspace_clustering_failures"] += 1
            else:
                attribution["intersection_selection_failures"] += 1
        diagnostics["failure_attribution"].append(attribution)
    return diagnostics


def run_suite(mode, workers):
    if mode == "smoke":
        thresholds = [0.0, 0.2]
        noises = [0.005]
        seeds = [20261016]
    else:
        thresholds = [0.0, 0.05, 0.1, 0.2, 0.4]
        noises = [0.005, 0.02]
        seeds = list(range(101001, 101011))
    configurations = [{
        "min_calibration_entry": threshold,
        "noise_std": noise,
        "seed": seed,
    } for threshold in thresholds for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Conditioning {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_case, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Conditioning {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (
        run["min_calibration_entry"], run["noise_std"], run["seed"], run["method"]
    ))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"conditioning{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"conditioning{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"conditioning{suffix}_diagnostics.json").open("w") as handle:
        json.dump(conditioning_diagnostics(runs), handle, indent=2)
    with (results_dir / f"conditioning{suffix}_config.json").open("w") as handle:
        json.dump({"mode": mode, "workers": workers, "configurations": configurations}, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
