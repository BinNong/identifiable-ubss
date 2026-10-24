import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coverage import generate_full_spark, support_basis
from evaluation import score_directions
from subspace import recover_directions_intersection_lattice


def generate_case(n_extra, sparsity, seed):
    rng = np.random.default_rng(seed)
    n_sources = 2 * sparsity + 1
    mixing = generate_full_spark(2 * sparsity, n_sources, rng, min_determinant=1e-5)
    target = n_sources - 1
    supports = [(index,) for index in range(target)]
    required = [
        tuple(index for index in range(sparsity) if index != omitted) + (target,)
        for omitted in range(sparsity)
    ]
    supports.extend(required)
    available = [
        support for support in combinations(range(target), sparsity)
        if support not in required
    ]
    selected = rng.choice(len(available), size=n_extra, replace=False)
    supports.extend(available[index] for index in selected)
    bases = [support_basis(mixing, support) for support in supports]
    return mixing, supports, bases


def evaluate_case(configuration):
    sparsity = configuration["sparsity"]
    n_extra = configuration["n_extra_supports"]
    seed = configuration["seed"]
    mixing, supports, bases = generate_case(n_extra, sparsity, seed)
    n_sources = mixing.shape[1]

    def estimate():
        return recover_directions_intersection_lattice(
            bases,
            n_sources,
            sparsity,
            window_indices=list(range(len(bases))),
            subspace_threshold=1e-8,
            intersection_threshold=1.0 - 1e-10,
            cluster_threshold=0.995,
        )

    estimate()
    start = time.perf_counter()
    directions, _, _, diagnostics = estimate()
    runtime = time.perf_counter() - start
    record = score_directions(mixing, directions)
    n_multi = sparsity + n_extra
    record.update({
        "sparsity": sparsity,
        "n_sources": n_sources,
        "n_extra_supports": n_extra,
        "seed": seed,
        "n_supports": len(supports),
        "n_prototypes": diagnostics["n_prototypes"],
        "n_multi_prototypes": n_multi,
        "n_direction_candidates": diagnostics["n_candidates"],
        "enumerated_combinations": sum(math.comb(n_multi, depth) for depth in range(2, sparsity + 1)),
        "runtime_seconds": runtime,
    })
    return record


def summarize_runs(runs):
    groups = {}
    for run in runs:
        key = (run["sparsity"], run["n_sources"], run["n_extra_supports"])
        groups.setdefault(key, []).append(run)
    metrics = [
        "mean_direction_cosine",
        "direction_recovery_0.99",
        "coverage_0.99",
        "n_supports",
        "n_prototypes",
        "n_multi_prototypes",
        "n_direction_candidates",
        "enumerated_combinations",
        "runtime_seconds",
    ]
    summary = []
    for (sparsity, n_sources, n_extra), records in sorted(groups.items()):
        row = {
            "sparsity": sparsity,
            "n_sources": n_sources,
            "n_extra_supports": n_extra,
            "n_runs": len(records),
        }
        for metric in metrics:
            values = np.asarray([record[metric] for record in records], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q25"] = float(np.percentile(values, 25))
            row[f"{metric}_q75"] = float(np.percentile(values, 75))
            critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = critical * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def configurations(mode):
    if mode == "smoke":
        seeds = [20261016]
        extra_by_sparsity = {2: [0, 2], 3: [0, 4]}
    else:
        seeds = list(range(105001, 105021))
        extra_by_sparsity = {2: [0, 2, 4, 6], 3: [0, 4, 8, 12], 4: [0, 4, 8, 12]}
    return [
        {"sparsity": sparsity, "n_extra_supports": n_extra, "seed": seed}
        for sparsity, extras in extra_by_sparsity.items()
        for n_extra in extras
        for seed in seeds
    ]


def run_suite(mode, workers):
    configs = configurations(mode)
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configs, start=1):
            runs.append(evaluate_case(configuration))
            print(f"Lattice complexity {index}/{len(configs)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_case, configuration): configuration for configuration in configs}
            for index, future in enumerate(as_completed(futures), start=1):
                runs.append(future.result())
                print(f"Lattice complexity {index}/{len(configs)} completed", flush=True)
    runs.sort(key=lambda run: (run["sparsity"], run["n_extra_supports"], run["seed"]))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"lattice_complexity{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"lattice_complexity{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"lattice_complexity{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configs,
            "evidence_scope": "Noiseless linear population subspaces; no PNL calibration or source recovery",
            "construction": "The target source requires depth k; extra multi-source supports exclude the target",
            "timing": "One complete untimed warmup call per case, followed by one independently recomputed timed call; no seeds removed",
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
