import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines import DICTIONARY_LEARNING_CONFIG, dictionary_learning_directions
from evaluation import (
    align_sources_to_truth,
    estimate_observation_noise_std,
    recover_sources,
    score_directions,
    score_source_recovery,
)
from pnl import calibrate_multichannel_pnl, forward_nonlinearity, inverse_nonlinearity
from subspace import (
    estimate_window_bases,
    generate_running_example,
    recover_directions_intersection_lattice,
    recover_rank_one_directions,
)


CALIBRATION_CONFIG = {
    "keep_fraction": 0.75,
    "upper_bound": 1.2,
    "selection_standard_errors": 2.0,
    "popsize": 7,
    "local_max_iter": 250,
}
DIRECTION_CONFIG = {
    "relative_threshold": 1e-3,
    "minimum_subspace_threshold": 0.03,
    "maximum_subspace_threshold": 0.15,
    "noise_threshold_factor": 5.0,
    "intersection_threshold": 0.995,
    "cluster_threshold": 0.995,
    "score_rule": "max",
    "selection_rule": "family_consistency",
}
DICTIONARY_LEARNING_N_INIT = 4


def truncate_regime(regime, n_blocks):
    block_length = regime["block_length"]
    truncated = dict(regime)
    for key in ["observations", "sources", "linear_observations"]:
        if key in truncated:
            truncated[key] = truncated[key][:n_blocks * block_length]
    if "block_supports" in truncated:
        truncated["block_supports"] = truncated["block_supports"][:n_blocks]
    return truncated


def build_window_preserving_missing_cross(regime, mixing, seed, noise, cross_support=(1, 4)):
    """Replace the cross-support blocks of ``regime`` with same-count partner-only blocks.

    Unlike ``truncate_regime`` this keeps the total window count fixed, so the
    control isolates the support-family ablation from any sample-size effect.
    """
    block_length = regime["block_length"]
    replacement_rng = np.random.default_rng(seed + 777)
    noise_rng = np.random.default_rng(seed + 778)
    n_sources = mixing.shape[1]
    observation_blocks = []
    source_blocks = []
    linear_blocks = []
    block_supports = []
    for block_index, support in enumerate(regime["block_supports"]):
        rows = slice(block_index * block_length, (block_index + 1) * block_length)
        if tuple(sorted(support)) == tuple(sorted(cross_support)):
            sources = np.zeros((block_length, n_sources))
            sources[:, cross_support[0]] = replacement_rng.normal(size=block_length)
            linear = sources @ mixing.T
            observations = forward_nonlinearity(linear, regime["forward_coefficients"])
            observations += noise * noise_rng.normal(size=observations.shape)
            observation_blocks.append(observations)
            source_blocks.append(sources)
            linear_blocks.append(linear)
            block_supports.append((cross_support[0],))
        else:
            observation_blocks.append(regime["observations"][rows])
            source_blocks.append(regime["sources"][rows])
            linear_blocks.append(regime["linear_observations"][rows])
            block_supports.append(support)
    replaced = {
        "observations": np.concatenate(observation_blocks, axis=0),
        "sources": np.concatenate(source_blocks, axis=0),
        "linear_observations": np.concatenate(linear_blocks, axis=0),
        "block_supports": block_supports,
        "block_length": block_length,
        "regime": regime["regime"],
        "forward_coefficients": regime["forward_coefficients"],
        "window_preserving_ablation": True,
    }
    return replaced


def calibrate_regimes(regimes, calibration_maxiter, seed):
    values = []
    records = []
    for regime in regimes:
        start = time.perf_counter()
        result = calibrate_multichannel_pnl(
            regime["observations"],
            regime["block_length"],
            max_iter=calibration_maxiter,
            seed=seed + regime["regime"],
            **CALIBRATION_CONFIG,
        )
        if "linear_observations" in regime:
            reference = regime["linear_observations"]
            relative_error = float(
                np.linalg.norm(result["linearized"] - reference)
                / (np.linalg.norm(reference) + 1e-12)
            )
        else:
            relative_error = None
        records.append({
            "regime": int(regime["regime"]),
            "runtime_seconds": time.perf_counter() - start,
            "coefficient_rmse": float(np.sqrt(np.mean(
                (result["inverse_coefficients"] - regime["forward_coefficients"]) ** 2
            ))),
            "relative_linearization_error": relative_error,
            "estimated_coefficients": result["inverse_coefficients"].tolist(),
            "objective": result["objective"],
            "selected_candidate": result["selected_candidate"],
            "candidate_objectives": result["candidate_objectives"],
            "n_evaluations": result["n_evaluations"],
        })
        values.append(result["linearized"])
    return values, records


def run_pipeline(mixing, observations, true_sources, block_length, method, recover=True, score_rule=None):
    start = time.perf_counter()
    score_rule = score_rule or DIRECTION_CONFIG["score_rule"]
    estimated_noise_std = estimate_observation_noise_std(observations, block_length, 2)
    estimated_relative_noise = estimate_observation_noise_std(
        observations, block_length, 2, relative=True
    )
    subspace_threshold = float(np.clip(
        DIRECTION_CONFIG["noise_threshold_factor"] * estimated_relative_noise,
        DIRECTION_CONFIG["minimum_subspace_threshold"],
        DIRECTION_CONFIG["maximum_subspace_threshold"],
    ))
    if method not in {"dictionary_learning", "dictionary_learning_warm", "lattice", "lattice_score", "rank_one"}:
        raise ValueError(method)
    bases, windows, _ = estimate_window_bases(
        observations,
        block_length,
        max_rank=2,
        relative_threshold=DIRECTION_CONFIG["relative_threshold"],
    )
    if method == "dictionary_learning":
        columns, dictionary_diagnostics = dictionary_learning_directions(
            observations, 5, 2, config=DICTIONARY_LEARNING_CONFIG, n_init=DICTIONARY_LEARNING_N_INIT
        )
        diagnostics = {
            "n_prototypes": 0,
            "n_rank_one_prototypes": 0,
            "n_candidates": 0,
            **dictionary_diagnostics,
        }
    elif method == "dictionary_learning_warm":
        warm_columns, _, _, _ = recover_directions_intersection_lattice(
            bases,
            5,
            2,
            windows,
            subspace_threshold=subspace_threshold,
            intersection_threshold=DIRECTION_CONFIG["intersection_threshold"],
            cluster_threshold=DIRECTION_CONFIG["cluster_threshold"],
            score_rule=score_rule,
        )
        columns, dictionary_diagnostics = dictionary_learning_directions(
            observations, 5, 2, config=DICTIONARY_LEARNING_CONFIG, n_init=1, dict_init=warm_columns
        )
        diagnostics = {
            "n_prototypes": 0,
            "n_rank_one_prototypes": 0,
            "n_candidates": 0,
            **dictionary_diagnostics,
        }
    else:
        if method == "lattice":
            columns, _, _, diagnostics = recover_directions_intersection_lattice(
                bases,
                5,
                2,
                windows,
                subspace_threshold=subspace_threshold,
                intersection_threshold=DIRECTION_CONFIG["intersection_threshold"],
                cluster_threshold=DIRECTION_CONFIG["cluster_threshold"],
                score_rule=score_rule,
            )
        elif method == "lattice_score":
            columns, _, _, diagnostics = recover_directions_intersection_lattice(
                bases,
                5,
                2,
                windows,
                subspace_threshold=subspace_threshold,
                intersection_threshold=DIRECTION_CONFIG["intersection_threshold"],
                cluster_threshold=DIRECTION_CONFIG["cluster_threshold"],
                score_rule=score_rule,
                selection_rule="score",
            )
        elif method == "rank_one":
            columns, _, diagnostics = recover_rank_one_directions(
                bases,
                5,
                windows,
                subspace_threshold=subspace_threshold,
            )
        else:
            raise ValueError(method)
    record = score_directions(mixing, columns)
    record["n_prototypes"] = diagnostics["n_prototypes"]
    record["n_rank_one_prototypes"] = diagnostics["n_rank_one_prototypes"]
    record["n_direction_candidates"] = diagnostics.get("n_candidates", 0)
    record["n_direction_clusters"] = diagnostics.get("n_clusters", 0)
    record["cluster_count_within_source_count"] = diagnostics.get("cluster_count_within_source_count")
    record["direction_clusters"] = diagnostics.get("direction_clusters", [])
    record["selection_diagnostics"] = {
        key: diagnostics["selection"][key]
        for key in ("rule", "search", "explained_windows", "n_windows")
        if key in diagnostics.get("selection", {})
    }
    record["dictionary_iterations"] = diagnostics.get("n_iter")
    record["dictionary_fit_records"] = diagnostics.get("fit_records")
    record["dictionary_selected_start"] = diagnostics.get("selected_start_label")
    record["estimated_recovery_noise_std"] = estimated_noise_std
    record["estimated_relative_noise"] = estimated_relative_noise
    record["subspace_threshold"] = subspace_threshold
    record["direction_runtime_seconds"] = time.perf_counter() - start
    if recover:
        recovery_start = time.perf_counter()
        estimates, _, _ = recover_sources(observations, columns, 2, noise_std=estimated_noise_std)
        estimates = align_sources_to_truth(mixing, columns, estimates)
        record.update(score_source_recovery(estimates, true_sources))
        record["recovery_runtime_seconds"] = time.perf_counter() - recovery_start
    record["runtime_seconds"] = time.perf_counter() - start
    return record


def evaluate_case(configuration):
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    blocks = configuration["blocks_per_support"]
    block_length = configuration["block_length"]
    calibration_maxiter = configuration["calibration_maxiter"]
    mixing, regimes = generate_running_example(
        blocks_per_support=blocks,
        block_length=block_length,
        noise_std=0.0,
        seed=seed,
    )
    noise_rng = np.random.default_rng(seed + 313)
    coefficients = [
        np.array([0.45, -0.25, 0.55, -0.35]),
        np.array([-0.30, 0.50, -0.20, 0.60]),
    ]
    for regime in regimes:
        regime["linear_observations"] = regime["observations"].copy()
        regime["forward_coefficients"] = coefficients[regime["regime"]]
        regime["observations"] = forward_nonlinearity(
            regime["linear_observations"],
            regime["forward_coefficients"],
        )
        regime["observations"] += noise * noise_rng.normal(size=regime["observations"].shape)

    blind_values, calibration_records = calibrate_regimes(regimes, calibration_maxiter, seed)
    oracle_values = [
        inverse_nonlinearity(regime["observations"], regime["forward_coefficients"])
        for regime in regimes
    ]
    raw_values = [regime["observations"] for regime in regimes]
    joint_sources = np.concatenate([regime["sources"] for regime in regimes], axis=0)
    output = {}

    is_main = configuration["experiment"] == "main"
    method_plan = [
        ("joint_blind_lattice", blind_values, "lattice"),
        ("joint_blind_rank_one", blind_values, "rank_one"),
        ("joint_oracle_lattice", oracle_values, "lattice"),
    ]
    if is_main:
        method_plan = [
            ("joint_blind_lattice", blind_values, "lattice"),
            ("joint_blind_lattice_score", blind_values, "lattice_score"),
            ("joint_blind_rank_one", blind_values, "rank_one"),
            ("joint_blind_dictionary_learning", blind_values, "dictionary_learning"),
            ("joint_blind_dictionary_learning_warm", blind_values, "dictionary_learning_warm"),
            ("joint_oracle_lattice", oracle_values, "lattice"),
            ("joint_oracle_rank_one", oracle_values, "rank_one"),
            ("joint_no_calibration_lattice", raw_values, "lattice"),
        ]

    for name, values, method in method_plan:
        output[name] = run_pipeline(
            mixing,
            np.concatenate(values, axis=0),
            joint_sources,
            block_length,
            method,
            recover=True,
        )

    for regime, values in zip(regimes, blind_values):
        if not is_main:
            break
        output[f"separate_regime_{regime['regime']}_blind_lattice"] = run_pipeline(
            mixing,
            values,
            regime["sources"],
            block_length,
            "lattice",
            recover=False,
        )

    if is_main:
        missing_regime = build_window_preserving_missing_cross(regimes[1], mixing, seed, noise)
        missing_values, missing_calibration = calibrate_regimes([missing_regime], calibration_maxiter, seed + 1000)
        missing_observations = np.concatenate([blind_values[0], missing_values[0]], axis=0)
        missing_sources = np.concatenate([regimes[0]["sources"], missing_regime["sources"]], axis=0)
        output["missing_cross_blind_lattice"] = run_pipeline(
            mixing,
            missing_observations,
            missing_sources,
            block_length,
            "lattice",
            recover=True,
        )

    output["joint_blind_lattice"]["calibration_coefficient_rmse"] = float(np.mean([
        record["coefficient_rmse"] for record in calibration_records
    ]))
    output["joint_blind_lattice"]["calibration_relative_error"] = float(np.mean([
        record["relative_linearization_error"] for record in calibration_records
    ]))
    output["joint_blind_lattice"]["calibration_runtime_seconds"] = float(np.sum([
        record["runtime_seconds"] for record in calibration_records
    ]))
    output["joint_blind_lattice"]["calibration_records"] = calibration_records
    if is_main:
        output["missing_cross_blind_lattice"]["calibration_coefficient_rmse"] = float(
            missing_calibration[0]["coefficient_rmse"]
        )
        output["missing_cross_blind_lattice"]["calibration_records"] = missing_calibration
        output["missing_cross_blind_lattice"]["window_preserving_ablation"] = True

    runs = []
    for method, metrics in output.items():
        runs.append({
            "experiment": configuration["experiment"],
            "method": method,
            "noise_std": noise,
            "seed": seed,
            "blocks_per_support": blocks,
            "block_length": block_length,
            **metrics,
        })
    return runs


def summarize_runs(runs):
    groups = {}
    for run in runs:
        key = (
            run["experiment"],
            run["method"],
            run["noise_std"],
            run["blocks_per_support"],
            run["block_length"],
        )
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
        "n_direction_clusters",
        "mean_active_correlation",
        "mean_scaled_error",
        "calibration_coefficient_rmse",
        "calibration_relative_error",
        "runtime_seconds",
        "direction_runtime_seconds",
        "recovery_runtime_seconds",
    ]
    summary = []
    for key, records in sorted(groups.items()):
        experiment, method, noise, blocks, block_length = key
        row = {
            "experiment": experiment,
            "method": method,
            "noise_std": noise,
            "blocks_per_support": blocks,
            "block_length": block_length,
            "n_runs": len(records),
        }
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                if len(values) > 1:
                    critical = float(t.ppf(0.975, len(values) - 1))
                    row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values))
                else:
                    row[f"{metric}_ci95"] = 0.0
        for source in range(5):
            values = [
                record["cosines_by_source"][source]
                if record["cosines_by_source"][source] is not None else 0.0
                for record in records
            ]
            row[f"source_{source + 1}_cosine_mean"] = float(np.mean(values))
            row[f"source_{source + 1}_cosine_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"source_{source + 1}_recovery_0.99"] = float(np.mean(np.asarray(values) >= 0.99))
        summary.append(row)
    return summary


def configurations(mode):
    if mode == "smoke":
        noises = [0.0, 0.01]
        seeds = [20261016]
        calibration_maxiter = 10
    elif mode == "development":
        noises = [0.0, 0.005, 0.01, 0.02]
        seeds = list(range(1001, 1006))
        calibration_maxiter = 50
    else:
        noises = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
        seeds = list(range(101001, 101101))
        sweep_seeds = list(range(101001, 101021))
        calibration_maxiter = 50
    configurations = []
    for noise in noises:
        for seed in seeds:
            configurations.append({
                "experiment": "main",
                "noise_std": noise,
                "seed": seed,
                "blocks_per_support": 8,
                "block_length": 80,
                "calibration_maxiter": calibration_maxiter,
            })
    if mode == "full":
        for blocks in [2, 4, 16]:
            for seed in sweep_seeds:
                configurations.append({
                    "experiment": "sample_complexity",
                    "noise_std": 0.005,
                    "seed": seed,
                    "blocks_per_support": blocks,
                    "block_length": 80,
                    "calibration_maxiter": calibration_maxiter,
                })
        for block_length in [40, 160]:
            for seed in sweep_seeds:
                configurations.append({
                    "experiment": "window_length",
                    "noise_std": 0.005,
                    "seed": seed,
                    "blocks_per_support": 8,
                    "block_length": block_length,
                    "calibration_maxiter": calibration_maxiter,
                })
    return configurations


def run_suite(mode, workers):
    configs = configurations(mode)
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configs, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Coverage {index}/{len(configs)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_case, configuration): configuration for configuration in configs}
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                configuration = futures[future]
                print(
                    f"Coverage {index}/{len(configs)} completed: "
                    f"{configuration['experiment']} noise={configuration['noise_std']} seed={configuration['seed']}",
                    flush=True,
                )
    runs.sort(key=lambda run: (
        run["experiment"], run["noise_std"], run["blocks_per_support"],
        run["block_length"], run["seed"], run["method"],
    ))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else f"_{mode}"
    with (results_dir / f"coverage{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"coverage{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"coverage{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configs,
            "calibration": CALIBRATION_CONFIG,
            "direction": DIRECTION_CONFIG,
            "dictionary_learning": DICTIONARY_LEARNING_CONFIG,
            "dictionary_learning_n_init": DICTIONARY_LEARNING_N_INIT,
            "baseline_description": "MiniBatchDictionaryLearning with OMP codes at the true n and k, using the same blind calibrated observations; best of n_init random starts, plus a warm start from the lattice estimate",
            "n_cases": len(configs),
            "n_runs": len(runs),
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "development", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
