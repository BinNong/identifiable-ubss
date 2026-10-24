import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_conditioning_experiments import evaluate_case


def summarize_runs(runs):
    groups = {}
    for run in runs:
        key = (
            run["method"],
            run["min_calibration_entry"],
            run["noise_std"],
            run["direction_score_rule"],
        )
        groups.setdefault(key, []).append(run)
    summary = []
    for (method, threshold, noise, score_rule), records in sorted(groups.items()):
        fifth_cosines = [
            record["cosines_by_source"][4] if record["cosines_by_source"][4] is not None else 0.0
            for record in records
        ]
        summary.append({
            "method": method,
            "min_calibration_entry": threshold,
            "noise_std": noise,
            "direction_score_rule": score_rule,
            "n_runs": len(records),
            "mean_direction_cosine": float(np.mean([record["mean_direction_cosine"] for record in records])),
            "coverage_0.99": float(np.mean([record["coverage_0.99"] for record in records])),
            "source_5_cosine": float(np.mean(fifth_cosines)),
            "source_5_recovery_0.99": float(np.mean(np.asarray(fifth_cosines) >= 0.99)),
            "complete_direction_recovery_rate": float(np.mean([
                record["complete_direction_recovery"] for record in records
            ])),
        })
    return summary


def run_suite(mode, workers):
    if mode == "smoke":
        thresholds = [0.05]
        noises = [0.02]
        seeds = [91001]
    else:
        thresholds = [0.0, 0.05, 0.1, 0.2, 0.4]
        noises = [0.005, 0.02, 0.05]
        seeds = list(range(91001, 91011))
    configurations = [{
        "min_calibration_entry": threshold,
        "noise_std": noise,
        "seed": seed,
        "score_rules": ["max", "mean", "legacy"],
        "recover": False,
    } for threshold in thresholds for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_case(configuration))
            print(f"Selection ablation {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_case, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Selection ablation {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (
        run["min_calibration_entry"],
        run["noise_std"],
        run["seed"],
        run["method"],
        run["direction_score_rule"],
    ))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "_smoke" if mode == "smoke" else "_development"
    with (results_dir / f"selection_ablation{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"selection_ablation{suffix}_summary.json").open("w") as handle:
        json.dump(summarize_runs(runs), handle, indent=2)
    with (results_dir / f"selection_ablation{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "evidence_scope": "Development diagnostics only; these previously inspected seeds are not final confirmation data",
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "development"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
