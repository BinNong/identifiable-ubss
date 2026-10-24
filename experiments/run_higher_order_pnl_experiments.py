import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import generate_full_spark
from evaluation import estimate_observation_noise_std, score_directions
from pnl import forward_nonlinearity, inverse_nonlinearity
from run_coverage_experiments import calibrate_regimes
from subspace import (
    estimate_window_bases,
    recover_directions_intersection_lattice,
    recover_rank_one_directions,
)


DIRECTION_CONFIG = {
    "relative_threshold": 1e-3,
    "minimum_subspace_threshold": 0.03,
    "maximum_subspace_threshold": 0.15,
    "noise_threshold_factor": 5.0,
    "intersection_threshold": 0.995,
    "cluster_threshold": 0.995,
}

PNL_COEFFICIENTS = [
    np.array([0.45, -0.25, 0.55, -0.35, 0.30, -0.50]),
    np.array([-0.30, 0.50, -0.20, 0.60, -0.40, 0.35]),
]

ISOLATED_SOURCES = [0, 1, 2, 4, 5, 6]
TARGET_SOURCE = 3
TRIPLE_SUPPORTS = [(0, 1, 3), (0, 2, 3), (1, 2, 3)]


def generate_higher_order_pnl_example(seed, blocks_per_support=4, block_length=100):
    rng = np.random.default_rng(seed)
    isolated = sorted(ISOLATED_SOURCES)
    accessible = None
    for _ in range(10000):
        candidate = rng.normal(size=(6, 6))
        if np.min(np.abs(candidate)) >= 0.2 and abs(np.linalg.det(candidate)) >= 0.005:
            accessible = candidate
            break
    if accessible is None:
        raise RuntimeError("Failed to generate a calibrable accessible block")
    column_order = [0, 1, 2, 4, 5, 6]
    mixing = np.zeros((6, 7))
    for position, source in enumerate(column_order):
        mixing[:, source] = accessible[:, position]
    from coverage import is_full_spark

    for _ in range(10000):
        target_column = rng.normal(size=6)
        mixing[:, TARGET_SOURCE] = target_column
        if is_full_spark(mixing, tolerance=1e-9):
            break
    else:
        raise RuntimeError("Failed to generate a full-spark higher-order mixing matrix")
    support_families = [
        [(index,) for index in isolated] + [TRIPLE_SUPPORTS[0], TRIPLE_SUPPORTS[1]],
        [(index,) for index in isolated] + [TRIPLE_SUPPORTS[2]],
    ]
    regimes = []
    for regime, supports in enumerate(support_families):
        observation_blocks = []
        source_blocks = []
        block_supports = []
        for support in supports:
            for _ in range(blocks_per_support):
                sources = np.zeros((block_length, 7))
                for index in support:
                    sources[:, index] = rng.normal(size=block_length)
                observation_blocks.append(sources @ mixing.T)
                source_blocks.append(sources)
                block_supports.append(support)
        regimes.append({
            "observations": np.concatenate(observation_blocks, axis=0),
            "sources": np.concatenate(source_blocks, axis=0),
            "block_supports": block_supports,
            "block_length": block_length,
            "regime": regime,
        })
    return mixing, regimes


def run_direction_stage(observations, mixing, block_length, method):
    estimated_relative_noise = estimate_observation_noise_std(
        observations, block_length, 3, relative=True
    )
    estimated_noise_std = estimate_observation_noise_std(observations, block_length, 3)
    subspace_threshold = float(np.clip(
        DIRECTION_CONFIG["noise_threshold_factor"] * estimated_relative_noise,
        DIRECTION_CONFIG["minimum_subspace_threshold"],
        DIRECTION_CONFIG["maximum_subspace_threshold"],
    ))
    bases, windows, _ = estimate_window_bases(
        observations,
        block_length,
        max_rank=3,
        relative_threshold=DIRECTION_CONFIG["relative_threshold"],
    )
    start = time.perf_counter()
    if method == "rank_one":
        directions, _, diagnostics = recover_rank_one_directions(
            bases, 7, windows, subspace_threshold=subspace_threshold,
        )
    else:
        max_depth = 2 if method == "pairwise_lattice" else None
        directions, _, _, diagnostics = recover_directions_intersection_lattice(
            bases,
            7,
            3,
            windows,
            subspace_threshold=subspace_threshold,
            intersection_threshold=DIRECTION_CONFIG["intersection_threshold"],
            cluster_threshold=DIRECTION_CONFIG["cluster_threshold"],
            max_depth=max_depth,
        )
    record = score_directions(mixing, directions)
    record["n_prototypes"] = diagnostics["n_prototypes"]
    record["n_rank_one_prototypes"] = diagnostics["n_rank_one_prototypes"]
    record["n_direction_candidates"] = diagnostics.get("n_candidates", 0)
    record["n_direction_clusters"] = diagnostics.get("n_clusters", 0)
    record["estimated_noise_std"] = estimated_noise_std
    record["subspace_threshold"] = subspace_threshold
    record["runtime_seconds"] = time.perf_counter() - start
    return record


def evaluate_case(configuration):
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    mixing, regimes = generate_higher_order_pnl_example(
        seed,
        blocks_per_support=configuration["blocks_per_support"],
        block_length=configuration["block_length"],
    )
    noise_rng = np.random.default_rng(seed + 313)
    for regime in regimes:
        regime["linear_observations"] = regime["observations"].copy()
        regime["forward_coefficients"] = PNL_COEFFICIENTS[regime["regime"]]
        regime["observations"] = forward_nonlinearity(
            regime["linear_observations"], regime["forward_coefficients"]
        )
        regime["observations"] += noise * noise_rng.normal(size=regime["observations"].shape)
    blind_values, calibration_records = calibrate_regimes(
        regimes, configuration["calibration_maxiter"], seed
    )
    oracle_values = [
        inverse_nonlinearity(regime["observations"], regime["forward_coefficients"])
        for regime in regimes
    ]
    block_length = configuration["block_length"]
    output = {}
    for name, values in [("blind", blind_values), ("oracle", oracle_values)]:
        pooled = np.concatenate(values, axis=0)
        for method in ["lattice", "pairwise_lattice", "rank_one"]:
            if name == "oracle" and method == "rank_one":
                continue
            output[f"{name}_{method}"] = run_direction_stage(pooled, mixing, block_length, method)
    output["blind_lattice"]["calibration_coefficient_rmse"] = float(np.mean([
        record["coefficient_rmse"] for record in calibration_records
    ]))
    output["blind_lattice"]["calibration_relative_error"] = float(np.mean([
        record["relative_linearization_error"] for record in calibration_records
    ]))
    runs = []
    for method, metrics in output.items():
        runs.append({
            "experiment": "higher_order_pnl",
            "method": method,
            "noise_std": noise,
            "seed": seed,
            "blocks_per_support": configuration["blocks_per_support"],
            "block_length": block_length,
            **metrics,
        })
    return runs


def summarize_runs(runs):
    from scipy.stats import t

    groups = {}
    for run in runs:
        groups.setdefault((run["method"], run["noise_std"]), []).append(run)
    metrics = [
        "mean_direction_cosine",
        "direction_recovery_0.99",
        "coverage_0.99",
        "n_prototypes",
        "n_direction_candidates",
        "n_direction_clusters",
        "runtime_seconds",
        "calibration_coefficient_rmse",
    ]
    summary = []
    for (method, noise), records in sorted(groups.items()):
        row = {"method": method, "noise_std": noise, "n_runs": len(records)}
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if not values:
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        target_values = [
            record["cosines_by_source"][TARGET_SOURCE]
            if record["cosines_by_source"][TARGET_SOURCE] is not None else 0.0
            for record in records
        ]
        row["target_cosine_mean"] = float(np.mean(target_values))
        row["target_recovery_0.99"] = float(np.mean(np.asarray(target_values) >= 0.99))
        summary.append(row)
    return summary


def run_suite(mode, workers):
    if mode == "smoke":
        noises = [0.0, 0.01]
        seeds = [20261016]
        calibration_maxiter = 10
    else:
        noises = [0.0, 0.005, 0.01, 0.02, 0.05]
        seeds = list(range(107001, 107021))
        calibration_maxiter = 50
    configurations = [{
        "noise_std": noise,
        "seed": seed,
        "blocks_per_support": 4,
        "block_length": 100,
        "calibration_maxiter": calibration_maxiter,
    } for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Higher-order PNL {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_case, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Higher-order PNL {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (run["noise_std"], run["seed"], run["method"]))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"higher_order_pnl{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"higher_order_pnl{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"higher_order_pnl{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "direction": DIRECTION_CONFIG,
            "pnl_coefficients": [coefficient.tolist() for coefficient in PNL_COEFFICIENTS],
            "isolated_sources": ISOLATED_SOURCES,
            "target_source": TARGET_SOURCE,
            "triple_supports": [list(support) for support in TRIPLE_SUPPORTS],
            "evidence_scope": "Depth-3 intersection lattice with regime-wise blind PNL calibration; the target source is recoverable only through the three-way intersection",
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
