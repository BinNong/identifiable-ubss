import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import generate_full_spark, intersection_conditioning, is_separating
from pnl import forward_nonlinearity, inverse_nonlinearity
from run_coverage_experiments import (
    DICTIONARY_LEARNING_CONFIG,
    calibrate_regimes,
    run_pipeline,
    summarize_runs,
)


def generate_random_support_example(seed, blocks_per_support=6, block_length=80):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        mixing = generate_full_spark(4, 5, rng, min_determinant=0.02)
        if np.min(np.abs(mixing[:, :4])) >= 0.2:
            break
    else:
        raise RuntimeError("Failed to generate a calibrable random-support matrix")

    while True:
        n_cross = int(rng.integers(3, 5))
        partners = rng.choice(4, size=n_cross, replace=True)
        if len(set(partners.tolist())) >= 2:
            break
    while True:
        regimes_for_cross = rng.integers(0, 2, size=n_cross)
        if len(set(regimes_for_cross.tolist())) == 2:
            break
    available_pairs = list(combinations(range(4), 2))
    n_extra = int(rng.integers(0, 3))
    extra_indices = rng.choice(len(available_pairs), size=n_extra, replace=False) if n_extra else []
    extra_regimes = rng.integers(0, 2, size=n_extra) if n_extra else []
    calibration = [(0,), (1,), (2,), (3,)]
    support_families = [list(calibration), list(calibration)]
    for partner, regime in zip(partners, regimes_for_cross):
        support_families[int(regime)].append(tuple(sorted((int(partner), 4))))
    for index, regime in zip(extra_indices, extra_regimes):
        support_families[int(regime)].append(available_pairs[int(index)])
    support_families = [sorted(set(family)) for family in support_families]

    regimes = []
    for regime, supports in enumerate(support_families):
        observation_blocks = []
        source_blocks = []
        block_supports = []
        for support in supports:
            for _ in range(blocks_per_support):
                sources = np.zeros((block_length, 5))
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
    metadata = {
        "support_families": [[list(support) for support in family] for family in support_families],
        "cross_partners": (np.asarray(partners) + 1).tolist(),
        "cross_regimes": (np.asarray(regimes_for_cross) + 1).tolist(),
        "n_extra_pair_supports": n_extra,
        "regime_separating": [is_separating(family, 5) for family in support_families],
        "pooled_separating": is_separating(support_families[0] + support_families[1], 5),
    }
    return mixing, regimes, metadata


def target_intersection_geometry(mixing, support_families):
    pooled = sorted({tuple(support) for family in support_families for support in family})
    cross_supports = [support for support in pooled if len(support) == 2 and 4 in support]
    records = []
    for left, right in combinations(cross_supports, 2):
        if set(left) & set(right) != {4}:
            continue
        records.append({
            "left_support": list(left),
            "right_support": list(right),
            **intersection_conditioning(mixing, left, right),
        })
    if not records:
        raise RuntimeError("The random support family has no pairwise target intersection")
    gammas = [record["intersection_gamma"] for record in records]
    distances = [record["cross_subspace_distance"] for record in records]
    return {
        "target_intersections": records,
        "n_target_intersections": len(records),
        "target_intersection_gamma_min": float(np.min(gammas)),
        "target_intersection_gamma_median": float(np.median(gammas)),
        "target_intersection_gamma_max": float(np.max(gammas)),
        "target_cross_subspace_distance_min": float(np.min(distances)),
        "target_cross_subspace_distance_median": float(np.median(distances)),
        "target_cross_subspace_distance_max": float(np.max(distances)),
    }


def evaluate_case(configuration):
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    mixing, regimes, metadata = generate_random_support_example(
        seed,
        blocks_per_support=configuration["blocks_per_support"],
        block_length=configuration["block_length"],
    )
    metadata.update(target_intersection_geometry(mixing, metadata["support_families"]))
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
    raw_values = [regime["observations"] for regime in regimes]
    true_sources = np.concatenate([regime["sources"] for regime in regimes], axis=0)
    output = {}
    for name, values, method in [
        ("joint_blind_lattice", blind_values, "lattice"),
        ("joint_blind_lattice_score", blind_values, "lattice_score"),
        ("joint_blind_rank_one", blind_values, "rank_one"),
        ("joint_blind_dictionary_learning", blind_values, "dictionary_learning"),
        ("joint_blind_dictionary_learning_warm", blind_values, "dictionary_learning_warm"),
        ("joint_oracle_lattice", oracle_values, "lattice"),
        ("joint_no_calibration_lattice", raw_values, "lattice"),
    ]:
        output[name] = run_pipeline(
            mixing,
            np.concatenate(values, axis=0),
            true_sources,
            configuration["block_length"],
            method,
            recover=True,
        )
    output["joint_blind_lattice"]["calibration_coefficient_rmse"] = float(np.mean([
        record["coefficient_rmse"] for record in calibration_records
    ]))
    output["joint_blind_lattice"]["calibration_relative_error"] = float(np.mean([
        record["relative_linearization_error"] for record in calibration_records
    ]))
    runs = []
    for method, metrics in output.items():
        runs.append({
            "experiment": "random_support",
            "method": method,
            "noise_std": noise,
            "seed": seed,
            "blocks_per_support": configuration["blocks_per_support"],
            "block_length": configuration["block_length"],
            **metadata,
            **metrics,
        })
    return runs


def diagnostic_group(records, oracle_lookup):
    group = {
        "n_cases": len(records),
        "seeds": [record["seed"] for record in records],
    }
    if not records:
        return group
    metrics = {
        "calibration_relative_error": "calibration_relative_error",
        "n_prototypes": "n_prototypes",
        "n_direction_candidates": "n_direction_candidates",
        "target_intersection_gamma_max": "target_intersection_gamma_max",
        "target_cross_subspace_distance_min": "target_cross_subspace_distance_min",
    }
    for label, key in metrics.items():
        values = [record[key] for record in records]
        group[f"{label}_median"] = float(np.median(values))
        group[f"{label}_min"] = float(np.min(values))
        group[f"{label}_max"] = float(np.max(values))
    paired_oracle = [oracle_lookup[record["seed"]] for record in records]
    group["oracle_source_5_recovery_rate"] = float(np.mean([
        record["cosines_by_source"][4] is not None and record["cosines_by_source"][4] >= 0.99
        for record in paired_oracle
    ]))
    group["blind_complete_direction_recovery_rate"] = float(np.mean([
        record["coverage_0.99"] == 5 for record in records
    ]))
    return group


def random_support_diagnostics(runs):
    diagnostics = {
        "source_5_recovery_threshold": 0.99,
        "descriptive_not_causal": True,
        "groups": [],
    }
    for noise in sorted({run["noise_std"] for run in runs}):
        blind = [
            run for run in runs
            if run["method"] == "joint_blind_lattice" and run["noise_std"] == noise
        ]
        oracle_lookup = {
            run["seed"]: run for run in runs
            if run["method"] == "joint_oracle_lattice" and run["noise_std"] == noise
        }
        for success in [True, False]:
            records = [
                record for record in blind
                if (record["cosines_by_source"][4] is not None and record["cosines_by_source"][4] >= 0.99) == success
            ]
            diagnostics["groups"].append({
                "noise_std": noise,
                "source_5_success": success,
                **diagnostic_group(records, oracle_lookup),
            })
    diagnostics["cluster_count_rule"] = cluster_count_rule_table(runs)
    diagnostics["selection_rule_comparison"] = selection_rule_comparison(runs)
    diagnostics["rank_one_prototype_stratification"] = rank_one_stratification(runs)
    return diagnostics


def _source_5_success(record):
    return record["cosines_by_source"][4] is not None and record["cosines_by_source"][4] >= 0.99


def cluster_count_rule_table(runs):
    table = []
    methods = sorted({run["method"] for run in runs if "n_direction_clusters" in run})
    for noise in sorted({run["noise_std"] for run in runs}):
        for method in methods:
            records = [
                run for run in runs
                if run["method"] == method and run["noise_std"] == noise
            ]
            if not records:
                continue
            for within in [True, False]:
                subset = [
                    record for record in records
                    if (record["n_direction_clusters"] <= 5) == within
                ]
                if not subset:
                    continue
                table.append({
                    "noise_std": noise,
                    "method": method,
                    "n_clusters_within_source_count": within,
                    "n_cases": len(subset),
                    "source_5_success_rate": float(np.mean([_source_5_success(record) for record in subset])),
                    "mean_direction_cosine": float(np.mean([record["mean_direction_cosine"] for record in subset])),
                })
    return table


def selection_rule_comparison(runs):
    comparison = []
    for noise in sorted({run["noise_std"] for run in runs}):
        lattice = {
            run["seed"]: run for run in runs
            if run["method"] == "joint_blind_lattice" and run["noise_std"] == noise
        }
        score = {
            run["seed"]: run for run in runs
            if run["method"] == "joint_blind_lattice_score" and run["noise_std"] == noise
        }
        common = sorted(set(lattice) & set(score))
        if not common:
            continue
        comparison.append({
            "noise_std": noise,
            "n_cases": len(common),
            "family_consistency_success": int(sum(_source_5_success(lattice[seed]) for seed in common)),
            "score_rule_success": int(sum(_source_5_success(score[seed]) for seed in common)),
            "family_consistency_mean_cosine": float(np.mean([lattice[seed]["mean_direction_cosine"] for seed in common])),
            "score_rule_mean_cosine": float(np.mean([score[seed]["mean_direction_cosine"] for seed in common])),
            "rescued_seeds": [
                seed for seed in common
                if _source_5_success(lattice[seed]) and not _source_5_success(score[seed])
            ],
            "lost_seeds": [
                seed for seed in common
                if not _source_5_success(lattice[seed]) and _source_5_success(score[seed])
            ],
        })
    return comparison


def rank_one_stratification(runs):
    strata = []
    for noise in sorted({run["noise_std"] for run in runs}):
        for method in ["joint_blind_lattice", "joint_blind_lattice_score"]:
            records = [
                run for run in runs
                if run["method"] == method and run["noise_std"] == noise
                and "n_rank_one_prototypes" in run
            ]
            if not records:
                continue
            for saturated in [False, True]:
                subset = [
                    record for record in records
                    if (record["n_rank_one_prototypes"] >= 5) == saturated
                ]
                if not subset:
                    continue
                strata.append({
                    "noise_std": noise,
                    "method": method,
                    "rank_one_prototypes_saturated": saturated,
                    "n_cases": len(subset),
                    "source_5_success_rate": float(np.mean([_source_5_success(record) for record in subset])),
                })
    return strata


def run_suite(mode, workers):
    if mode == "smoke":
        noises = [0.005]
        seeds = [20261016]
    else:
        noises = [0.005, 0.02, 0.05]
        seeds = list(range(102001, 102101))
    configurations = [{
        "noise_std": noise,
        "seed": seed,
        "blocks_per_support": 6,
        "block_length": 80,
    } for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Random support {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_case, configuration): configuration for configuration in configurations}
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Random support {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (run["noise_std"], run["seed"], run["method"]))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"random_support{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"random_support{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"random_support{suffix}_diagnostics.json").open("w") as handle:
        json.dump(random_support_diagnostics(runs), handle, indent=2)
    with (results_dir / f"random_support{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "dictionary_learning": DICTIONARY_LEARNING_CONFIG,
            "evidence_scope": "Random separating support families with source 5 never isolated and sources 1--4 retained for calibration",
            "n_cases": len(configurations),
            "n_runs": len(runs),
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
