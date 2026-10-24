import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import estimate_observation_noise_std, score_directions
from subspace import (
    estimate_window_bases,
    generate_higher_order_example,
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


def evaluate_case(configuration):
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    mixing, data = generate_higher_order_example(
        blocks_per_support=configuration["blocks_per_support"],
        block_length=configuration["block_length"],
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1777)
    observations = data["observations"] + noise * rng.normal(size=data["observations"].shape)
    estimated_noise_std = estimate_observation_noise_std(observations, data["block_length"], 3)
    estimated_relative_noise = estimate_observation_noise_std(
        observations, data["block_length"], 3, relative=True
    )
    subspace_threshold = float(np.clip(
        DIRECTION_CONFIG["noise_threshold_factor"] * estimated_relative_noise,
        DIRECTION_CONFIG["minimum_subspace_threshold"],
        DIRECTION_CONFIG["maximum_subspace_threshold"],
    ))
    bases, windows, _ = estimate_window_bases(
        observations,
        data["block_length"],
        max_rank=3,
        relative_threshold=DIRECTION_CONFIG["relative_threshold"],
    )
    output = []
    for method in ["lattice", "pairwise_lattice", "rank_one"]:
        start = time.perf_counter()
        if method == "rank_one":
            directions, _, diagnostics = recover_rank_one_directions(
                bases,
                7,
                windows,
                subspace_threshold=subspace_threshold,
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
        output.append({
            "method": method,
            "noise_std": noise,
            "seed": seed,
            "blocks_per_support": configuration["blocks_per_support"],
            "block_length": configuration["block_length"],
            "estimated_noise_std": estimated_noise_std,
            "subspace_threshold": subspace_threshold,
            "n_prototypes": diagnostics["n_prototypes"],
            "n_rank_one_prototypes": diagnostics["n_rank_one_prototypes"],
            "n_direction_candidates": diagnostics.get("n_candidates", 0),
            "runtime_seconds": time.perf_counter() - start,
            **record,
        })
    return output


def summarize_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["method"], run["noise_std"]), []).append(run)
    metrics = [
        "mean_direction_cosine",
        "direction_recovery_0.99",
        "direction_recovery_0.95",
        "coverage_0.99",
        "n_prototypes",
        "n_direction_candidates",
        "runtime_seconds",
    ]
    summary = []
    for (method, noise), records in sorted(groups.items()):
        row = {"method": method, "noise_std": noise, "n_runs": len(records)}
        for metric in metrics:
            values = [record[metric] for record in records]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        for source in range(7):
            values = [
                record["cosines_by_source"][source]
                if record["cosines_by_source"][source] is not None else 0.0
                for record in records
            ]
            row[f"source_{source + 1}_cosine_mean"] = float(np.mean(values))
            row[f"source_{source + 1}_recovery_0.99"] = float(np.mean(np.asarray(values) >= 0.99))
        summary.append(row)
    return summary


def run_suite(mode, workers):
    if mode == "smoke":
        noises = [0.0, 0.005]
        seeds = [20261016]
    else:
        noises = [0.0, 0.02, 0.05, 0.1, 0.2]
        seeds = list(range(103001, 103021))
    configurations = [{
        "noise_std": noise,
        "seed": seed,
        "blocks_per_support": 4,
        "block_length": 100,
    } for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Higher-order {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_case, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Higher-order {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (run["noise_std"], run["seed"], run["method"]))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"higher_order{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"higher_order{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"higher_order{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "direction": DIRECTION_CONFIG,
            "evidence_scope": "Linear support-subspace geometry; no PNL calibration is used in this experiment",
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
