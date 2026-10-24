import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import intersection_conditioning, support_basis
from evaluation import estimate_observation_noise_std
from pnl import forward_nonlinearity
from run_coverage_experiments import (
    CALIBRATION_CONFIG,
    DIRECTION_CONFIG,
    calibrate_regimes,
)
from subspace import (
    estimate_window_bases,
    generate_running_example,
    recover_directions_intersection_lattice,
)


def window_support(regimes, window_index):
    blocks_per_regime = len(regimes[0]["block_supports"])
    regime = window_index // blocks_per_regime
    block = window_index % blocks_per_regime
    return tuple(sorted(regimes[regime]["block_supports"][block]))


def evaluate_case(configuration):
    noise = configuration["noise_std"]
    seed = configuration["seed"]
    mixing, regimes = generate_running_example(
        blocks_per_support=8, block_length=80, noise_std=0.0, seed=seed,
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
            regime["linear_observations"], regime["forward_coefficients"]
        )
        regime["observations"] += noise * noise_rng.normal(size=regime["observations"].shape)
    blind_values, _ = calibrate_regimes(regimes, configuration["calibration_maxiter"], seed)
    observations = np.concatenate(blind_values, axis=0)
    block_length = regimes[0]["block_length"]
    estimated_relative_noise = estimate_observation_noise_std(observations, block_length, 2, relative=True)
    subspace_threshold = float(np.clip(
        DIRECTION_CONFIG["noise_threshold_factor"] * estimated_relative_noise,
        DIRECTION_CONFIG["minimum_subspace_threshold"],
        DIRECTION_CONFIG["maximum_subspace_threshold"],
    ))
    bases, windows, _ = estimate_window_bases(
        observations, block_length, max_rank=2,
        relative_threshold=DIRECTION_CONFIG["relative_threshold"],
    )
    directions, selected, clusters, diagnostics = recover_directions_intersection_lattice(
        bases, 5, 2, windows,
        subspace_threshold=subspace_threshold,
        intersection_threshold=DIRECTION_CONFIG["intersection_threshold"],
        cluster_threshold=DIRECTION_CONFIG["cluster_threshold"],
    )
    target_direction = mixing[:, 4] / np.linalg.norm(mixing[:, 4])
    lattice_clusters = [
        cluster for cluster in selected
        if any(member.get("kind") == "lattice" for member in cluster["members"])
    ]
    if not lattice_clusters:
        return {"noise_std": noise, "seed": seed, "records": []}
    cluster = max(lattice_clusters, key=lambda item: abs(item["direction"] @ target_direction))
    member = max(
        (member for member in cluster["members"] if member.get("kind") == "lattice"),
        key=lambda member: member.get("s1_min", 0.0),
    )
    prototype_ids = member["prototypes"]
    member_supports = []
    for prototype_id in prototype_ids:
        prototype = diagnostics["prototypes"][prototype_id]
        support_counts = {}
        for window in prototype["windows"]:
            support = window_support(regimes, window)
            support_counts[support] = support_counts.get(support, 0) + 1
        member_supports.append(max(support_counts, key=support_counts.get))
    if len(member_supports) != 2 or not all(4 in support for support in member_supports):
        return {"noise_std": noise, "seed": seed, "records": []}
    left_support, right_support = member_supports
    shared = sorted(set(left_support) & set(right_support))
    if shared != [4]:
        return {"noise_std": noise, "seed": seed, "records": []}
    geometry = intersection_conditioning(mixing, left_support, right_support)
    true_left = support_basis(mixing, left_support)
    true_right = support_basis(mixing, right_support)
    estimated_left = diagnostics["prototypes"][prototype_ids[0]]["basis"]
    estimated_right = diagnostics["prototypes"][prototype_ids[1]]["basis"]
    delta_left = float(np.linalg.norm(
        estimated_left @ estimated_left.T - true_left @ true_left.T, ord=2
    ))
    delta_right = float(np.linalg.norm(
        estimated_right @ estimated_right.T - true_right @ true_right.T, ord=2
    ))
    epsilon = float(member["epsilon_equiv"])
    gamma = geometry["intersection_gamma"]
    direction = cluster["direction"]
    realized = float(np.sqrt(max(0.0, 1.0 - (direction @ target_direction) ** 2)))
    bound = delta_left + (epsilon + delta_left + delta_right) / gamma
    threshold_only_bound = epsilon / gamma
    return {
        "noise_std": noise,
        "seed": seed,
        "records": [{
            "left_support": list(left_support),
            "right_support": list(right_support),
            "gamma": float(gamma),
            "delta_left": delta_left,
            "delta_right": delta_right,
            "epsilon": epsilon,
            "realized_error": realized,
            "prop3_bound": float(bound),
            "threshold_only_bound": float(threshold_only_bound),
            "bound_margin": float(bound - realized),
            "realized_angle_degrees": float(np.degrees(np.arcsin(min(1.0, realized)))),
            "prop3_bound_degrees": float(np.degrees(np.arcsin(min(1.0, bound)))),
            "threshold_only_degrees": float(np.degrees(np.arcsin(min(1.0, threshold_only_bound)))),
        }],
    }


def summarize(records):
    if not records:
        return {}
    margins = [record["bound_margin"] for record in records]
    summary = {
        "n_records": len(records),
        "n_bound_violations": int(sum(margin < 0 for margin in margins)),
        "min_bound_margin": float(min(margins)),
        "median_realized_angle_degrees": float(np.median([record["realized_angle_degrees"] for record in records])),
        "median_prop3_bound_degrees": float(np.median([record["prop3_bound_degrees"] for record in records])),
        "median_threshold_only_degrees": float(np.median([record["threshold_only_degrees"] for record in records])),
        "median_gamma": float(np.median([record["gamma"] for record in records])),
        "median_epsilon": float(np.median([record["epsilon"] for record in records])),
        "median_delta": float(np.median([
            max(record["delta_left"], record["delta_right"]) for record in records
        ])),
    }
    return summary


def run_suite(mode, workers):
    if mode == "smoke":
        noises = [0.005]
        seeds = [20261016]
        calibration_maxiter = 10
    else:
        noises = [0.005, 0.01, 0.02]
        seeds = list(range(101001, 101021))
        calibration_maxiter = 50
    configurations = [{
        "noise_std": noise,
        "seed": seed,
        "calibration_maxiter": calibration_maxiter,
    } for noise in noises for seed in seeds]
    if workers == 1:
        outputs = [evaluate_case(configuration) for configuration in configurations]
    else:
        outputs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_case, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                outputs.append(future.result())
                print(f"Stability budget {index}/{len(configurations)} completed", flush=True)
    records = [record for output in outputs for record in output["records"]]
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    payload = {
        "description": "Proposition 3 evaluated with pipeline-estimated subspaces and true geometry: realized target-direction error versus the deterministic bound",
        "records": records,
        "summary": summarize(records),
    }
    with (results_dir / f"stability_budget{suffix}.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
