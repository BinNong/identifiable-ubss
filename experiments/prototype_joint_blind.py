import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import score_directions
from pnl import calibrate_multichannel_pnl, forward_nonlinearity, inverse_nonlinearity
from subspace import (
    estimate_window_bases,
    generate_higher_order_example,
    generate_running_example,
    recover_directions_from_bases,
    recover_directions_intersection_lattice,
)


def run_core(noise, seed, calibration_maxiter):
    mixing, regimes = generate_running_example(noise_std=0.0, seed=seed)
    coefficients = [
        np.array([0.45, -0.25, 0.55, -0.35]),
        np.array([-0.30, 0.50, -0.20, 0.60]),
    ]
    noise_rng = np.random.default_rng(seed + 313)
    calibrated = []
    calibration_records = []
    for regime in regimes:
        regime["linear_observations"] = regime["observations"].copy()
        regime["forward_coefficients"] = coefficients[regime["regime"]]
        regime["observations"] = forward_nonlinearity(
            regime["linear_observations"], regime["forward_coefficients"]
        )
        regime["observations"] += noise * noise_rng.normal(size=regime["observations"].shape)
    for regime in regimes:
        start = time.perf_counter()
        result = calibrate_multichannel_pnl(
            regime["observations"],
            regime["block_length"],
            keep_fraction=0.8,
            upper_bound=1.2,
            max_iter=calibration_maxiter,
            seed=seed + regime["regime"],
        )
        elapsed = time.perf_counter() - start
        calibrated.append(result["linearized"])
        calibration_records.append({
            "regime": regime["regime"],
            "seconds": elapsed,
            "estimated": result["inverse_coefficients"].tolist(),
            "truth": regime["forward_coefficients"].tolist(),
            "rmse": float(np.sqrt(np.mean((result["inverse_coefficients"] - regime["forward_coefficients"]) ** 2))),
            "objective": result["objective"],
            "evaluations": result["n_evaluations"],
            "candidate_objectives": result["candidate_objectives"],
            "candidate_coefficients": result["candidate_coefficients"],
            "selected_candidate": result["selected_candidate"],
        })
    joint = np.concatenate(calibrated, axis=0)
    oracle = np.concatenate([
        inverse_nonlinearity(regime["observations"], regime["forward_coefficients"])
        for regime in regimes
    ])
    output = {"noise": noise, "seed": seed, "calibration": calibration_records}
    for method, values in [("blind", joint), ("oracle", oracle)]:
        bases, windows, _ = estimate_window_bases(values, regimes[0]["block_length"], max_rank=2)
        directions, selected, _, diagnostics = recover_directions_intersection_lattice(
            bases, 5, 2, windows
        )
        output[method] = score_directions(mixing, directions)
        output[method]["diagnostics"] = {
            "n_prototypes": diagnostics["n_prototypes"],
            "n_rank_one_prototypes": diagnostics["n_rank_one_prototypes"],
            "n_candidates": diagnostics["n_candidates"],
        }
    return output


def run_higher_order(seed):
    mixing, data = generate_higher_order_example(seed=seed)
    bases, windows, _ = estimate_window_bases(data["observations"], data["block_length"], max_rank=3)
    lattice, _, _, diagnostics = recover_directions_intersection_lattice(bases, 7, 3, windows)
    pairwise, _, _ = recover_directions_from_bases(bases, 7, windows)
    return {
        "lattice": score_directions(mixing, lattice),
        "pairwise": score_directions(mixing, pairwise),
        "diagnostics": {
            "n_prototypes": diagnostics["n_prototypes"],
            "n_rank_one_prototypes": diagnostics["n_rank_one_prototypes"],
            "n_candidates": diagnostics["n_candidates"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--calibration-maxiter", type=int, default=30)
    parser.add_argument("--higher-only", action="store_true")
    args = parser.parse_args()
    if not args.higher_only:
        print(json.dumps(run_core(args.noise, args.seed, args.calibration_maxiter), indent=2), flush=True)
    print(json.dumps({"higher_order": run_higher_order(args.seed)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
