import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import clopper_pearson


def source_success(index, threshold=0.99):
    def evaluate(run):
        value = run["cosines_by_source"][index]
        return value is not None and value >= threshold
    return evaluate


def complete_success(n_sources):
    return lambda run: run["coverage_0.99"] == n_sources


def complete_dynamic_success(run):
    return run["coverage_0.99"] == run["n_sources"]


def add_dataset(output, runs, dataset, group_keys, metrics):
    groups = {}
    for run in runs:
        key = tuple(run[key] for key in group_keys)
        groups.setdefault(key, []).append(run)
    for key, records in groups.items():
        base = dict(zip(group_keys, key))
        if dataset == "lattice_complexity":
            base["method"] = "lattice"
        for metric_name, success_function in metrics.items():
            successes = sum(bool(success_function(run)) for run in records)
            output.append({
                "dataset": dataset,
                **base,
                "metric": metric_name,
                **clopper_pearson(successes, len(records)),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    results = project / "results"
    suffix = "" if args.mode == "full" else "_smoke"
    output = []
    specifications = [
        (
            "coverage",
            ["experiment", "method", "noise_std", "blocks_per_support", "block_length"],
            {"source_5_recovery": source_success(4), "complete_5_recovery": complete_success(5)},
        ),
        (
            "random_support",
            ["method", "noise_std", "blocks_per_support", "block_length"],
            {"source_5_recovery": source_success(4), "complete_5_recovery": complete_success(5)},
        ),
        (
            "higher_order",
            ["method", "noise_std", "blocks_per_support", "block_length"],
            {"source_4_recovery": source_success(3), "complete_7_recovery": complete_success(7)},
        ),
        (
            "lattice_complexity",
            ["sparsity", "n_sources", "n_extra_supports"],
            {"complete_direction_recovery": complete_dynamic_success},
        ),
        (
            "conditioning",
            ["method", "min_calibration_entry", "noise_std"],
            {"complete_5_recovery": complete_success(5)},
        ),
    ]
    for dataset, group_keys, metrics in specifications:
        with (results / f"{dataset}{suffix}_runs.json").open() as handle:
            runs = json.load(handle)
        add_dataset(output, runs, dataset, group_keys, metrics)
    output.sort(key=lambda record: (
        record["dataset"], record.get("experiment", ""), record["method"],
        record.get("noise_std", 0.0), record.get("min_calibration_entry", 0.0),
        record.get("blocks_per_support", 0), record.get("block_length", 0), record["metric"],
    ))
    with (results / f"recovery_intervals{suffix}.json").open("w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {len(output)} exact binomial intervals")


if __name__ == "__main__":
    main()
