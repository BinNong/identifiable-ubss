import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import (
    align_sources_to_truth,
    estimate_observation_noise_std,
    recover_sources,
    score_directions,
    score_source_recovery,
)
from subspace import (
    estimate_window_bases,
    generate_scaling_example,
    recover_directions_intersection_lattice,
    recover_rank_one_directions,
)


def evaluate_case(configuration):
    n_sources = configuration["n_sources"]
    blocks = configuration["blocks_per_support"]
    seed = configuration["seed"]
    mixing, regimes = generate_scaling_example(
        n_sources,
        blocks_per_support=blocks,
        block_length=80,
        seed=seed,
    )
    observations = np.concatenate([regime["observations"] for regime in regimes], axis=0)
    true_sources = np.concatenate([regime["sources"] for regime in regimes], axis=0)
    block_length = regimes[0]["block_length"]
    estimated_noise = estimate_observation_noise_std(
        observations, block_length, 2, relative=True
    )
    subspace_threshold = float(np.clip(5 * estimated_noise, 0.03, 0.15))
    bases, windows, _ = estimate_window_bases(observations, block_length, max_rank=2)
    output = []
    for method in ["lattice", "rank_one"]:
        start = time.perf_counter()
        if method == "lattice":
            directions, _, _, diagnostics = recover_directions_intersection_lattice(
                bases,
                n_sources,
                2,
                windows,
                subspace_threshold=subspace_threshold,
                intersection_threshold=0.995,
                cluster_threshold=0.995,
            )
        else:
            directions, _, diagnostics = recover_rank_one_directions(
                bases,
                n_sources,
                windows,
                subspace_threshold=subspace_threshold,
            )
        record = score_directions(mixing, directions)
        record.update({
            "method": method,
            "n_sources": n_sources,
            "blocks_per_support": blocks,
            "seed": seed,
            "n_windows": len(bases),
            "n_prototypes": diagnostics["n_prototypes"],
            "n_direction_candidates": diagnostics.get("n_candidates", 0),
            "runtime_seconds": time.perf_counter() - start,
        })
        if method == "lattice":
            recovery_start = time.perf_counter()
            estimates, _, _ = recover_sources(observations, directions, 2, solver="omp")
            estimates = align_sources_to_truth(mixing, directions, estimates)
            record.update(score_source_recovery(estimates, true_sources))
            record["recovery_runtime_seconds"] = time.perf_counter() - recovery_start
            if n_sources <= 6:
                exact_record = dict(record)
                exact_start = time.perf_counter()
                estimates, _, _ = recover_sources(observations, directions, 2, noise_std=0.0)
                estimates = align_sources_to_truth(mixing, directions, estimates)
                exact_record.update(score_source_recovery(estimates, true_sources))
                exact_record["method"] = "lattice_exact_recovery"
                exact_record["recovery_runtime_seconds"] = time.perf_counter() - exact_start
                output.append(exact_record)
        output.append(record)
    return output


def summarize_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["method"], run["n_sources"], run["blocks_per_support"]), []).append(run)
    metrics = [
        "mean_direction_cosine",
        "direction_recovery_0.99",
        "coverage_0.99",
        "support_f1",
        "mean_active_correlation",
        "mean_scaled_error",
        "n_windows",
        "n_prototypes",
        "n_direction_candidates",
        "runtime_seconds",
        "recovery_runtime_seconds",
    ]
    summary = []
    for (method, n_sources, blocks), records in sorted(groups.items()):
        row = {
            "method": method,
            "n_sources": n_sources,
            "blocks_per_support": blocks,
            "n_runs": len(records),
        }
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_median"] = float(np.median(values))
                row[f"{metric}_q25"] = float(np.percentile(values, 25))
                row[f"{metric}_q75"] = float(np.percentile(values, 75))
                critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
                row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def run_suite(mode, workers):
    if mode == "smoke":
        source_counts = [5, 6]
        block_counts = [2]
        seeds = [20261016]
    else:
        source_counts = [5, 6, 8, 10]
        block_counts = [2, 8, 16]
        seeds = list(range(104001, 104021))
    configurations = [{
        "n_sources": n_sources,
        "blocks_per_support": blocks,
        "seed": seed,
    } for n_sources in source_counts for blocks in block_counts for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Scaling {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_case, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Scaling {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (run["n_sources"], run["blocks_per_support"], run["seed"], run["method"]))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"scaling{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"scaling{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"scaling{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "evidence_scope": "Noiseless linear support-subspace geometry with OMP source recovery; no PNL calibration",
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
