import argparse
import json
import math
from pathlib import Path


EXPECTED_FULL_COUNTS = {
    "coverage_runs.json": 6900,
    "random_support_runs.json": 2100,
    "higher_order_runs.json": 300,
    "higher_order_pnl_runs.json": 500,
    "scaling_runs.json": 600,
    "lattice_complexity_runs.json": 240,
    "pnl_runs.json": 1440,
    "semisynthetic_runs.json": 240,
    "conditioning_runs.json": 200,
}


def finite_values(value):
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    results = project / "results"
    checks = []

    def check(name, condition, details=None):
        checks.append({"name": name, "passed": bool(condition), "details": details})

    suffix = "" if args.mode == "full" else "_smoke"
    datasets = {}
    for base in ["coverage", "random_support", "higher_order", "higher_order_pnl", "scaling", "lattice_complexity", "pnl", "semisynthetic", "conditioning"]:
        runs_path = results / f"{base}{suffix}_runs.json"
        summary_path = results / f"{base}{suffix}_summary.json"
        with runs_path.open() as handle:
            runs = json.load(handle)
        with summary_path.open() as handle:
            summary = json.load(handle)
        datasets[base] = (runs, summary)
        check(f"{base}_finite_runs", finite_values(runs), {"n_runs": len(runs)})
        check(f"{base}_finite_summary", finite_values(summary), {"n_rows": len(summary)})
        if args.mode == "full":
            check(
                f"{base}_run_count",
                len(runs) == EXPECTED_FULL_COUNTS[f"{base}_runs.json"],
                {"actual": len(runs), "expected": EXPECTED_FULL_COUNTS[f"{base}_runs.json"]},
            )

    expected_confirmation_seeds = {
        "coverage": set(range(101001, 101101)),
        "random_support": set(range(102001, 102101)),
        "higher_order": set(range(103001, 103021)),
        "higher_order_pnl": set(range(107001, 107021)),
        "scaling": set(range(104001, 104021)),
        "lattice_complexity": set(range(105001, 105021)),
        "pnl": set(range(101001, 101021)),
        "semisynthetic": set(range(106001, 106021)),
        "conditioning": set(range(101001, 101011)),
    }
    configurations = {}
    confirmation_seeds = set()
    for base in datasets:
        with (results / f"{base}{suffix}_config.json").open() as handle:
            configuration = json.load(handle)
        configurations[base] = configuration
        actual_seeds = {record["seed"] for record in configuration["configurations"]}
        confirmation_seeds.update(actual_seeds)
        if args.mode == "full":
            check(
                f"{base}_confirmation_seeds",
                actual_seeds == expected_confirmation_seeds[base],
                {"actual": sorted(actual_seeds), "expected": sorted(expected_confirmation_seeds[base])},
            )
    check(
        "direction_score_rule_frozen",
        configurations["coverage"]["direction"]["score_rule"] == "max",
        configurations["coverage"]["direction"],
    )
    selection_mode = "development" if args.mode == "full" else "smoke"
    with (results / f"selection_ablation_{selection_mode}_runs.json").open() as handle:
        selection_runs = json.load(handle)
    with (results / f"selection_ablation_{selection_mode}_config.json").open() as handle:
        selection_config = json.load(handle)
    development_seeds = {record["seed"] for record in selection_config["configurations"]}
    check("selection_ablation_finite", finite_values(selection_runs), {"n_runs": len(selection_runs)})
    check("selection_ablation_accounting", (
        len(selection_runs) == (900 if args.mode == "full" else 6)
        and len(selection_config["configurations"]) == (150 if args.mode == "full" else 1)
        and selection_config["mode"] == selection_mode
        and {record["direction_score_rule"] for record in selection_runs} == {"max", "mean", "legacy"}
    ))
    check(
        "development_confirmation_seed_separation",
        development_seeds.isdisjoint(confirmation_seeds),
        {"development": sorted(development_seeds), "confirmation": sorted(confirmation_seeds)},
    )

    with (results / "theory_verification.json").open() as handle:
        theory = json.load(handle)
    check("stability_bound", theory["stability_bound"]["max_violation"] < 1e-9, theory["stability_bound"])
    check("higher_order_lattice", theory["blind_higher_order"]["lattice"]["coverage_0.99"] == 7)
    check("higher_order_pairwise_control", theory["blind_higher_order"]["pairwise"]["coverage_0.99"] == 6)
    check("non_identifiable_alternative", theory["running_example"]["ambiguity_projection_error"] < 1e-8)
    check("partial_direction_sparse_recovery", theory["running_example"]["single_column_k_cap_error"] < 1e-8)
    scale = theory["sensor_scale"]
    check("sensor_scale_observation_identity", scale["observation_identity_error"] < 1e-12, scale)
    check("sensor_scale_changes_direction", scale["minimum_column_direction_cosine"] < 0.999)
    check("normalized_class_excludes_scale", scale["outside_unit_linear_coefficient_class"])
    check("cross_regime_scale_alignment", scale["misaligned_cross_regime_intersection_rank"] == 0 and scale["aligned_cross_regime_intersection_rank"] == 1)
    counterexample = theory["calibration_counterexample"]
    check("calibration_counterexample_identity", counterexample["max_observation_identity_error"] < 1e-8, counterexample)
    check("calibration_counterexample_nonlinear_sources", counterexample["max_source_formula_error"] < 1e-8 and counterexample["max_nonlinear_source_change"] > 1e-3)
    check("noise_scale_invariance", theory["noise_scale_invariance"]["max_relative_error"] < 1e-12, theory["noise_scale_invariance"])
    selection = theory["direction_selection_invariance"]
    check(
        "direction_selection_redundancy_invariance",
        selection["max_score_absolute_error"] < 1e-12
        and selection["selected_direction_cosine"] > 1.0 - 1e-12
        and selection["legacy_selection_reversal"],
        selection,
    )
    check(
        "higher_order_depth_necessity",
        theory["blind_higher_order"]["target_pairwise_intersection_ranks"] == [2, 2, 2]
        and theory["blind_higher_order"]["target_triple_intersection_rank"] == 1,
        {key: theory["blind_higher_order"][key] for key in [
            "target_pairwise_support_sizes",
            "target_pairwise_intersection_ranks",
            "target_triple_support",
            "target_triple_intersection_rank",
        ]},
    )

    coverage_summary = datasets["coverage"][1]
    blind = next(row for row in coverage_summary if row.get("experiment") == "main" and row["method"] == "joint_blind_lattice" and row["noise_std"] == (0.005 if args.mode == "full" else 0.01))
    rank_one = next(row for row in coverage_summary if row.get("experiment") == "main" and row["method"] == "joint_blind_rank_one" and row["noise_std"] == blind["noise_std"])
    if args.mode == "smoke":
        check("central_blind_coverage", blind["coverage_0.99_mean"] == 5.0, blind)
        check("central_rank_one_control", rank_one["coverage_0.99_mean"] == 4.0, rank_one)
    else:
        check("central_confirmation_sample_sizes", blind["n_runs"] == 100 and rank_one["n_runs"] == 100)

    random_runs = datasets["random_support"][0]
    check("random_support_all_separating", all(run["pooled_separating"] for run in random_runs))
    check("random_support_source_5_never_isolated", all(
        4 not in support
        for run in random_runs
        for family in run["support_families"]
        for support in family
        if len(support) == 1
    ))
    families = {json.dumps(run["support_families"], sort_keys=True) for run in random_runs}
    check("random_support_family_diversity", len(families) >= (15 if args.mode == "full" else 1), {"n_families": len(families)})
    check("random_support_target_intersections", all(
        run["n_target_intersections"] >= 1
        and all(record["intersection_rank"] == 1 for record in run["target_intersections"])
        for run in random_runs
    ))
    with (results / f"random_support{suffix}_diagnostics.json").open() as handle:
        random_diagnostics = json.load(handle)
    check("random_support_diagnostics_finite", finite_values(random_diagnostics))
    random_blind_runs = [run for run in random_runs if run["method"] == "joint_blind_lattice"]
    for noise in sorted({run["noise_std"] for run in random_blind_runs}):
        groups = [group for group in random_diagnostics["groups"] if group["noise_std"] == noise]
        expected = sum(run["noise_std"] == noise for run in random_blind_runs)
        check(
            f"random_support_diagnostic_counts_{noise}",
            len(groups) == 2 and sum(group["n_cases"] for group in groups) == expected,
            groups,
        )

    higher_summary = datasets["higher_order"][1]
    for row in higher_summary:
        if args.mode == "smoke" and row["method"] == "lattice":
            check(f"higher_lattice_noise_{row['noise_std']}", row["coverage_0.99_mean"] == 7.0)
        elif args.mode == "smoke" and row["method"] == "pairwise_lattice":
            check(f"higher_pairwise_noise_{row['noise_std']}", row["coverage_0.99_mean"] == 6.0)
    if args.mode == "full":
        check("higher_confirmation_sample_sizes", all(row["n_runs"] == 20 for row in higher_summary))

    with (results / f"recovery_intervals{suffix}.json").open() as handle:
        recovery_intervals = json.load(handle)
    check("recovery_intervals_finite", finite_values(recovery_intervals), {"n_rows": len(recovery_intervals)})
    check("recovery_intervals_ordered", all(
        record["ci95_lower"] <= record["rate"] <= record["ci95_upper"]
        and 0.0 <= record["ci95_lower"] <= record["ci95_upper"] <= 1.0
        for record in recovery_intervals
    ))
    central_interval = next(
        record for record in recovery_intervals
        if record["dataset"] == "coverage"
        and record.get("experiment") == "main"
        and record["method"] == "joint_blind_lattice"
        and record["noise_std"] == blind["noise_std"]
        and record["metric"] == "source_5_recovery"
    )
    if args.mode == "full" and central_interval["n_successes"] == central_interval["n_trials"]:
        expected_lower = 0.025 ** (1.0 / central_interval["n_trials"])
        check(
            "exact_binomial_not_degenerate",
            abs(central_interval["ci95_lower"] - expected_lower) < 1e-6,
            {**central_interval, "expected_lower": expected_lower},
        )

    for row in datasets["lattice_complexity"][1]:
        check(
            f"lattice_complexity_k{row['sparsity']}_extra{row['n_extra_supports']}",
            row["coverage_0.99_mean"] == row["n_sources"],
            row,
        )

    with (results / f"conditioning{suffix}_diagnostics.json").open() as handle:
        conditioning = json.load(handle)
    check("conditioning_diagnostics_finite", finite_values(conditioning))
    for row in conditioning["failure_attribution"]:
        blind_only_attributed = sum(row[key] for key in [
            "calibration_associated_failures",
            "subspace_clustering_failures",
            "intersection_selection_failures",
        ])
        paired_total = sum(row[key] for key in [
            "both_successes",
            "blind_only_failures",
            "oracle_only_failures",
            "both_failures",
        ])
        check(f"conditioning_failure_accounting_{row['noise_std']}", (
            paired_total == row["n_cases"]
            and row["blind_failures"] == row["blind_only_failures"] + row["both_failures"]
            and row["oracle_failures"] == row["oracle_only_failures"] + row["both_failures"]
            and row["shared_direction_stage_failures"] == row["both_failures"]
            and row["blind_only_failures"] == blind_only_attributed
        ), row)

    pnl_blind_runs = [run for run in datasets["pnl"][0] if run["method"] == "joint_blind"]
    check("pnl_calibration_diagnostics", all(
        len(run["calibration_records"]) == 2
        and all(
            len(record["estimated_coefficients"]) == 2
            and len(record["coefficient_bound_hit_channels"]) == 2
            and record["objective_cost"] >= 0.0
            and record["zero_calibration_cost"] >= 0.0
            for record in run["calibration_records"]
        )
        for run in pnl_blind_runs
    ))

    with (results / f"semisynthetic{suffix}_config.json").open() as handle:
        semi_config = json.load(handle)
    check("semi_synthetic_evidence_boundary", semi_config["not_real_mixture_validation"] is True)

    with (results / f"stability_budget{suffix}.json").open() as handle:
        stability_budget = json.load(handle)
    check("stability_budget_finite", finite_values(stability_budget))
    if args.mode == "full":
        budget_summary = stability_budget["summary"]
        check(
            "stability_budget_no_violations",
            budget_summary["n_records"] >= 40 and budget_summary["n_bound_violations"] == 0,
            budget_summary,
        )
    check("cluster_count_rule_table_present", len(random_diagnostics["cluster_count_rule"]) > 0)
    check("selection_rule_comparison_present", len(random_diagnostics["selection_rule_comparison"]) > 0)
    check("rank_one_stratification_present", len(random_diagnostics["rank_one_prototype_stratification"]) > 0)

    validation = {
        "mode": args.mode,
        "passed": all(record["passed"] for record in checks),
        "n_checks": len(checks),
        "checks": checks,
    }
    with (results / f"validation{suffix}.json").open("w") as handle:
        json.dump(validation, handle, indent=2)
    print(json.dumps({"passed": validation["passed"], "n_checks": len(checks)}, indent=2))
    if not validation["passed"]:
        failed = [record for record in checks if not record["passed"]]
        raise AssertionError(f"Validation failed: {failed}")


if __name__ == "__main__":
    main()
