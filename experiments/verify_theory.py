import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import orth

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coverage import (
    construct_ambiguity,
    generate_full_spark,
    intersect_subspaces,
    is_full_spark,
    is_separating,
    normalize_supports,
    recover_sparse_sample,
    reveal_all_columns,
    spark,
    support_intersection,
)
from evaluation import score_directions
from subspace import (
    _finalize_direction_clusters,
    cluster_direction_candidates,
    estimate_window_bases,
    generate_higher_order_example,
    generate_running_example,
    recover_directions_from_bases,
    recover_directions_intersection_lattice,
)


def check_stability_bound(rng, n_cases=30):
    records = []
    for _ in range(n_cases):
        m = 6
        shared = orth(rng.normal(size=(m, 1)))
        left = np.column_stack([shared, orth(rng.normal(size=(m, 2)))])
        left = orth(left)
        right = np.column_stack([shared, orth(rng.normal(size=(m, 2)))])
        right = orth(right)
        intersection = intersect_subspaces(left, right)
        if intersection.shape[1] != 1:
            continue
        left_perp = orth((np.eye(m) - intersection @ intersection.T) @ left)
        residual_map = (np.eye(m) - right @ right.T) @ left_perp
        gamma = float(np.min(np.linalg.svd(residual_map, compute_uv=False)))
        if gamma < 1e-7:
            continue

        left_hat = orth(left + 0.01 * rng.normal(size=left.shape))
        right_hat = orth(right + 0.01 * rng.normal(size=right.shape))
        projector_left = left @ left.T
        projector_right = right @ right.T
        projector_left_hat = left_hat @ left_hat.T
        projector_right_hat = right_hat @ right_hat.T
        delta_left = float(np.linalg.norm(projector_left_hat - projector_left, ord=2))
        delta_right = float(np.linalg.norm(projector_right_hat - projector_right, ord=2))

        residual_operator = (np.eye(m) - projector_right_hat) @ left_hat
        _, _, vh = np.linalg.svd(residual_operator, full_matrices=False)
        candidate = left_hat @ vh[-1]
        candidate /= np.linalg.norm(candidate)
        epsilon = float(np.linalg.norm(residual_operator @ vh[-1]))
        projector_intersection = intersection @ intersection.T
        actual = float(np.linalg.norm((np.eye(m) - projector_intersection) @ candidate))
        bound = delta_left + (epsilon + delta_left + delta_right) / gamma
        records.append({
            "gamma": gamma,
            "epsilon": epsilon,
            "delta_left": delta_left,
            "delta_right": delta_right,
            "actual": actual,
            "bound": bound,
            "violation": actual - bound,
        })
    return records


def check_sensor_scale(matrix, rng):
    from coverage import support_basis
    from pnl import forward_nonlinearity

    diagonal = np.array([0.7, 1.1, 1.5, 0.9])
    transform = np.diag(diagonal)
    coefficients = np.array([0.45, -0.25, 0.55, -0.35])
    sources = rng.normal(size=(100, matrix.shape[1]))
    linear = sources @ matrix.T
    observations = forward_nonlinearity(linear, coefficients)
    alternative_matrix = transform @ matrix
    alternative_linear = sources @ alternative_matrix.T
    alternative_observations = forward_nonlinearity(
        alternative_linear / diagonal,
        coefficients,
    )
    original_normalized = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    alternative_normalized = alternative_matrix / np.linalg.norm(
        alternative_matrix, axis=0, keepdims=True
    )
    direction_cosines = np.abs(np.sum(original_normalized * alternative_normalized, axis=0))
    misaligned_left = support_basis(matrix, (0, 4))
    misaligned_right = support_basis(alternative_matrix, (1, 4))
    aligned_left = support_basis(matrix, (0, 4))
    aligned_right = support_basis(matrix, (1, 4))
    return {
        "observation_identity_error": float(np.max(np.abs(observations - alternative_observations))),
        "alternative_full_spark": is_full_spark(alternative_matrix),
        "minimum_column_direction_cosine": float(np.min(direction_cosines)),
        "transformed_linear_coefficients": (1.0 / diagonal).tolist(),
        "outside_unit_linear_coefficient_class": bool(np.any(np.abs(1.0 / diagonal - 1.0) > 1e-12)),
        "misaligned_cross_regime_intersection_rank": int(intersect_subspaces(
            misaligned_left, misaligned_right
        ).shape[1]),
        "aligned_cross_regime_intersection_rank": int(intersect_subspaces(
            aligned_left, aligned_right
        ).shape[1]),
    }


def check_calibration_counterexample():
    from pnl import forward_nonlinearity

    matrix = np.column_stack([np.eye(4), np.ones(4)])
    supports = [(0,), (1,), (2,), (3,), (0, 4), (1, 4)]
    coefficients = np.array([0.45, -0.25, 0.55, -0.35])
    rng = np.random.default_rng(20261017)

    def transform(values):
        return values + values**3

    def inverse_transform(values):
        values = np.asarray(values, dtype=float)
        estimate = values.copy()
        for _ in range(40):
            residual = estimate + estimate**3 - values
            step = residual / (1.0 + 3.0 * estimate**2)
            estimate -= step
            if np.max(np.abs(step)) < 1e-12:
                break
        return estimate

    max_sparse_error = 0.0
    max_observation_error = 0.0
    max_formula_error = 0.0
    max_source_change = 0.0
    support_matches = []
    for support in supports:
        for _ in range(20):
            source = np.zeros(5)
            source[list(support)] = rng.normal(size=len(support))
            linear = matrix @ source
            transformed = transform(linear)
            alternative_source, estimated_support, _ = recover_sparse_sample(matrix, transformed, 2)
            alternative_linear = matrix @ alternative_source
            observations = forward_nonlinearity(linear, coefficients)
            alternative_nonlinearity = forward_nonlinearity(
                inverse_transform(alternative_linear), coefficients
            )
            max_sparse_error = max(max_sparse_error, float(np.linalg.norm(alternative_linear - transformed)))
            max_observation_error = max(max_observation_error, float(np.max(np.abs(observations - alternative_nonlinearity))))
            max_source_change = max(max_source_change, float(np.max(np.abs(alternative_source - source))))
            support_matches.append(set(estimated_support) == set(support))
            if support == (0, 4):
                expected_first = transform(source[0] + source[4]) - transform(source[4])
                expected_fifth = transform(source[4])
                max_formula_error = max(max_formula_error, float(max(
                    abs(alternative_source[0] - expected_first),
                    abs(alternative_source[4] - expected_fifth),
                )))
    return {
        "full_spark": is_full_spark(matrix),
        "spark": spark(matrix),
        "separating": is_separating(supports, 5),
        "max_sparse_reconstruction_error": max_sparse_error,
        "max_observation_identity_error": max_observation_error,
        "max_source_formula_error": max_formula_error,
        "max_nonlinear_source_change": max_source_change,
        "support_accuracy": float(np.mean(support_matches)),
        "cubic_derivative_at_zero": 1.0,
    }


def check_noise_scale_invariance(rng):
    from evaluation import estimate_observation_noise_std

    factors = rng.normal(size=(2, 4))
    observations = rng.normal(size=(320, 2)) @ factors
    observations += 0.01 * rng.normal(size=observations.shape)
    estimates = [
        estimate_observation_noise_std(scale * observations, 80, 2, relative=True)
        for scale in [1e-6, 1.0, 1e6]
    ]
    return {
        "relative_estimates": estimates,
        "max_relative_error": float(max(abs(value - estimates[1]) for value in estimates)),
    }


def check_direction_selection_invariance():
    target = np.array([1.0, 0.0, 0.0])
    distractor = np.array([0.0, 1.0, 0.0])
    target_metadata = {"windows": (0, 1), "prototypes": (0,), "kind": "lattice", "depth": 2}
    distractor_metadata = {"windows": (2, 3), "prototypes": (1,), "kind": "lattice", "depth": 2}
    base_candidates = [(target, 4.0, target_metadata), (distractor, 3.0, distractor_metadata)]
    redundant_candidates = base_candidates + [
        (distractor.copy(), 3.0, dict(distractor_metadata)),
        (distractor.copy(), 3.0, dict(distractor_metadata)),
    ]

    def finalize(candidates):
        return _finalize_direction_clusters(cluster_direction_candidates(candidates), [], 1)

    def nearest(clusters, direction):
        return max(clusters, key=lambda cluster: abs(cluster["direction"] @ direction))

    base = finalize(base_candidates)
    redundant = finalize(redundant_candidates)
    base_target = nearest(base, target)
    base_distractor = nearest(base, distractor)
    redundant_target = nearest(redundant, target)
    redundant_distractor = nearest(redundant, distractor)
    return {
        "max_score_absolute_error": float(max(
            abs(base_target["score"] - redundant_target["score"]),
            abs(base_distractor["score"] - redundant_distractor["score"]),
        )),
        "selected_direction_cosine": float(abs(redundant[0]["direction"] @ target)),
        "target_direction_cosine": float(abs(redundant_target["direction"] @ target)),
        "base_distractor_candidates": int(base_distractor["n_candidates"]),
        "redundant_distractor_candidates": int(redundant_distractor["n_candidates"]),
        "target_legacy_score": float(redundant_target["legacy_score"]),
        "redundant_distractor_legacy_score": float(redundant_distractor["legacy_score"]),
        "legacy_selection_reversal": bool(
            base_target["legacy_score"] > base_distractor["legacy_score"]
            and redundant_distractor["legacy_score"] > redundant_target["legacy_score"]
        ),
    }


def check_selection_rule_saturation(rng):
    """Demonstrate the rank-one saturation failure of score-based selection.

    Six columns in R^5: five isolated sources (singleton windows) plus a sixth
    source that only appears in the two cross supports {1,6} and {2,6}. A
    spurious rank-one prototype brings the rank-one count to n = 6. Score-based
    top-n selection can never pick the lattice candidate for source 6, while
    family-consistency selection must, because the {1,6} and {2,6} windows can
    only be explained by a direction set containing a_6.
    """
    from subspace import (
        _cluster_consistency_membership,
        _finalize_direction_clusters,
        cluster_direction_candidates,
        select_directions_by_family_consistency,
    )
    from coverage import support_basis

    mixing = generate_full_spark(5, 6, rng, min_determinant=0.02)
    supports = [(0,), (1,), (2,), (3,), (4,), (0, 5), (1, 5)]
    bases = [support_basis(mixing, support) for support in supports for _ in range(4)]
    spurious = orth(rng.normal(size=(5, 1)))[:, 0]
    candidates = []
    for index in range(5):
        direction = mixing[:, index] / np.linalg.norm(mixing[:, index])
        candidates.append((direction, 16.0, {
            "windows": tuple(range(4 * index, 4 * index + 4)),
            "prototypes": (index,),
            "kind": "singleton",
            "depth": 1,
            "s1_min": 1.0,
            "epsilon_equiv": 0.0,
        }))
    candidates.append((spurious, 16.0, {
        "windows": (100, 101),
        "prototypes": (5,),
        "kind": "singleton",
        "depth": 1,
        "s1_min": 1.0,
        "epsilon_equiv": 0.0,
    }))
    target = mixing[:, 5] / np.linalg.norm(mixing[:, 5])
    candidates.append((target, 14.4, {
        "windows": (20, 21, 24, 25),
        "prototypes": (6, 7),
        "kind": "lattice",
        "depth": 2,
        "s1_min": 0.9,
        "epsilon_equiv": float(np.sqrt(1 - 0.9**2)),
    }))
    clusters = _finalize_direction_clusters(cluster_direction_candidates(candidates), [], 1)
    score_selected = clusters[:6]
    score_cosine = max(
        abs(cluster["direction"] @ target) for cluster in score_selected
    )
    consistent_indices, info = select_directions_by_family_consistency(
        clusters, bases, 6, 2
    )
    _cluster_consistency_membership(clusters, consistent_indices)
    consistent_cosine = max(
        abs(clusters[index]["direction"] @ target) for index in consistent_indices
    )
    return {
        "n_rank_one_prototypes": 6,
        "n_sources": 6,
        "score_selection_target_cosine": float(score_cosine),
        "family_consistency_target_cosine": float(consistent_cosine),
        "family_consistency_explained_windows": info["explained_windows"],
        "n_windows": info["n_windows"],
    }


def check_support_selection_false_alarm(rng, n_trials=2000):
    """Empirical error rate of the chi-squared support-selection rule.

    The nominal per-sample level is 1%. The rule scans support sizes and keeps
    the smallest passing one, and the post-inversion noise is only first-order
    Gaussian, so the achieved level is measured here rather than claimed.
    """
    from pnl import forward_nonlinearity, inverse_nonlinearity

    mixing = generate_full_spark(4, 5, rng, min_determinant=0.02)
    coefficients = np.array([0.45, -0.25, 0.55, -0.35])
    noise_std = 0.01
    nominal = 0.01
    overshoot_linear = 0
    errors_linear = 0
    overshoot_pnl = 0
    errors_pnl = 0
    supports = [(i,) for i in range(5)] + [(0, 4), (1, 4), (2, 3), (1, 2)]
    for trial in range(n_trials):
        support = supports[trial % len(supports)]
        source = np.zeros(5)
        source[list(support)] = rng.normal(size=len(support))
        linear = mixing @ source
        noisy = linear + noise_std * rng.normal(size=4)
        estimate, estimated_support, _ = recover_sparse_sample(
            mixing, noisy, 2, noise_std=noise_std, false_alarm=nominal
        )
        errors_linear += int(set(estimated_support) != set(support))
        overshoot_linear += int(len(estimated_support) > len(support))
        observed = forward_nonlinearity(linear, coefficients) + noise_std * rng.normal(size=4)
        inverted = inverse_nonlinearity(observed, coefficients)
        estimate, estimated_support, _ = recover_sparse_sample(
            mixing, inverted, 2, noise_std=noise_std, false_alarm=nominal
        )
        errors_pnl += int(set(estimated_support) != set(support))
        overshoot_pnl += int(len(estimated_support) > len(support))
    return {
        "n_trials": n_trials,
        "nominal_false_alarm": nominal,
        "noise_std": noise_std,
        "linear_support_error_rate": errors_linear / n_trials,
        "linear_overshoot_rate": overshoot_linear / n_trials,
        "pnl_support_error_rate": errors_pnl / n_trials,
        "pnl_overshoot_rate": overshoot_pnl / n_trials,
    }


def main():
    rng = np.random.default_rng(20261016)
    output = {}

    matrix = generate_full_spark(4, 5, rng)
    calibration = [(0,), (1,), (2,), (3,)]
    regime_1 = normalize_supports(calibration + [(0, 4)])
    regime_2 = normalize_supports(calibration + [(1, 4)])
    pooled = normalize_supports(regime_1 + regime_2)
    _, pooled_cosines, pooled_ranks = reveal_all_columns(matrix, pooled)
    ambiguity = construct_ambiguity(matrix, regime_1, 4)

    recovery_errors = []
    support_matches = []
    for support in pooled:
        for _ in range(20):
            source = np.zeros(5)
            source[list(support)] = rng.normal(size=len(support))
            observation = matrix @ source
            estimate, _, _ = recover_sparse_sample(matrix, observation, 2)
            recovery_errors.append(float(np.linalg.norm(estimate - source)))
            estimated_support = set(np.flatnonzero(np.abs(estimate) > 1e-7))
            support_matches.append(estimated_support == set(support))

    partial_estimate, _, _ = recover_sparse_sample(matrix[:, [0]], matrix[:, 0], 2)
    partial_error = float(np.linalg.norm(partial_estimate - np.ones(1)))

    output["running_example"] = {
        "full_spark": is_full_spark(matrix),
        "spark": spark(matrix),
        "regime_1_separating": is_separating(regime_1, 5),
        "regime_2_separating": is_separating(regime_2, 5),
        "pooled_separating": is_separating(pooled, 5),
        "source_5_regime_1_intersection": sorted(support_intersection(regime_1, 4)),
        "source_5_pooled_intersection": sorted(support_intersection(regime_2, 4)),
        "pooled_direction_cosines": pooled_cosines,
        "pooled_intersection_ranks": pooled_ranks,
        "ambiguity_projection_error": ambiguity["max_projection_error"],
        "ambiguity_changed_column_cosine": ambiguity["original_cosine"],
        "ambiguity_full_spark": ambiguity["full_spark"],
        "max_sparse_recovery_error": max(recovery_errors),
        "sparse_support_accuracy": float(np.mean(support_matches)),
        "single_column_k_cap_error": partial_error,
    }
    output["sensor_scale"] = check_sensor_scale(matrix, rng)
    output["calibration_counterexample"] = check_calibration_counterexample()
    output["noise_scale_invariance"] = check_noise_scale_invariance(rng)
    output["direction_selection_invariance"] = check_direction_selection_invariance()
    output["selection_rule_saturation"] = check_selection_rule_saturation(rng)
    output["support_selection_false_alarm"] = check_support_selection_false_alarm(rng)

    matrix_multi = generate_full_spark(6, 7, rng)
    multi_supports = normalize_supports([
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 4),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
    ])
    _, multi_cosines, multi_ranks = reveal_all_columns(matrix_multi, multi_supports)
    output["multi_support_intersection"] = {
        "source_0_containing_intersection": sorted(support_intersection(multi_supports, 0)),
        "separating": is_separating(multi_supports, 7),
        "direction_cosines": multi_cosines,
        "intersection_ranks": multi_ranks,
    }

    from coverage import support_basis

    higher_matrix, higher_data = generate_higher_order_example(seed=20261016)
    higher_bases, higher_windows, _ = estimate_window_bases(
        higher_data["observations"], higher_data["block_length"], max_rank=3
    )
    higher_lattice, _, _, higher_diagnostics = recover_directions_intersection_lattice(
        higher_bases, 7, 3, higher_windows
    )
    higher_pairwise, _, _ = recover_directions_from_bases(higher_bases, 7, higher_windows)
    higher_target_supports = [(0, 1, 3), (0, 2, 3), (1, 2, 3)]
    higher_target_bases = [support_basis(higher_matrix, support) for support in higher_target_supports]
    higher_pairwise_ranks = [
        intersect_subspaces(higher_target_bases[left], higher_target_bases[right]).shape[1]
        for left, right in [(0, 1), (0, 2), (1, 2)]
    ]
    higher_pairwise_support_sizes = [
        len(set(higher_target_supports[left]) & set(higher_target_supports[right]))
        for left, right in [(0, 1), (0, 2), (1, 2)]
    ]
    higher_triple_basis = intersect_subspaces(
        intersect_subspaces(higher_target_bases[0], higher_target_bases[1]),
        higher_target_bases[2],
    )
    output["blind_higher_order"] = {
        "lattice": score_directions(higher_matrix, higher_lattice),
        "pairwise": score_directions(higher_matrix, higher_pairwise),
        "n_prototypes": higher_diagnostics["n_prototypes"],
        "n_candidates": higher_diagnostics["n_candidates"],
        "target_pairwise_support_sizes": higher_pairwise_support_sizes,
        "target_pairwise_intersection_ranks": higher_pairwise_ranks,
        "target_triple_support": sorted(set(higher_target_supports[0]) & set(higher_target_supports[1]) & set(higher_target_supports[2])),
        "target_triple_intersection_rank": int(higher_triple_basis.shape[1]),
    }

    running_matrix, running_regimes = generate_running_example(noise_std=0.0, seed=20261016)
    running_observations = np.concatenate([regime["observations"] for regime in running_regimes])
    running_bases, running_windows, _ = estimate_window_bases(
        running_observations, running_regimes[0]["block_length"], max_rank=2
    )
    running_directions, _, _, running_diagnostics = recover_directions_intersection_lattice(
        running_bases, 5, 2, running_windows
    )
    output["blind_running_example"] = {
        "directions": score_directions(running_matrix, running_directions),
        "n_prototypes": running_diagnostics["n_prototypes"],
        "n_candidates": running_diagnostics["n_candidates"],
    }

    missing_supports = normalize_supports([(0,), (1,), (2,), (0, 4), (1, 4)])
    output["missing_coverage"] = {
        "separating": is_separating(missing_supports, 5),
        "source_3_intersection": support_intersection(missing_supports, 3),
    }

    stability_records = check_stability_bound(rng)
    output["stability_bound"] = {
        "n_cases": len(stability_records),
        "max_violation": max(record["violation"] for record in stability_records),
        "minimum_margin": min(record["bound"] - record["actual"] for record in stability_records),
        "records": stability_records,
    }

    output["tolerances"] = {
        "direction_cosine_min_expected": 1.0 - 1e-8,
        "projection_error_max_expected": 1e-8,
        "stability_max_violation": 1e-9,
    }

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "theory_verification.json").open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))

    assert output["running_example"]["full_spark"]
    assert not output["running_example"]["regime_1_separating"]
    assert not output["running_example"]["regime_2_separating"]
    assert output["running_example"]["pooled_separating"]
    assert min(pooled_cosines) > 1.0 - 1e-8
    assert max(pooled_ranks) == 1
    assert ambiguity["max_projection_error"] < 1e-8
    assert ambiguity["original_cosine"] < 1.0 - 1e-6
    assert ambiguity["full_spark"]
    assert max(recovery_errors) < 1e-8
    assert np.mean(support_matches) == 1.0
    assert partial_error < 1e-8
    scale = output["sensor_scale"]
    assert scale["observation_identity_error"] < 1e-12
    assert scale["alternative_full_spark"]
    assert scale["minimum_column_direction_cosine"] < 0.999
    assert scale["outside_unit_linear_coefficient_class"]
    assert scale["misaligned_cross_regime_intersection_rank"] == 0
    assert scale["aligned_cross_regime_intersection_rank"] == 1
    counterexample = output["calibration_counterexample"]
    assert counterexample["full_spark"] and counterexample["spark"] == 5
    assert counterexample["separating"]
    assert counterexample["max_sparse_reconstruction_error"] < 1e-8
    assert counterexample["max_observation_identity_error"] < 1e-8
    assert counterexample["max_source_formula_error"] < 1e-8
    assert counterexample["max_nonlinear_source_change"] > 1e-3
    assert counterexample["support_accuracy"] == 1.0
    assert output["noise_scale_invariance"]["max_relative_error"] < 1e-12
    saturation = output["selection_rule_saturation"]
    assert saturation["score_selection_target_cosine"] < 0.999
    assert saturation["family_consistency_target_cosine"] > 0.999
    assert saturation["family_consistency_explained_windows"] == saturation["n_windows"]
    false_alarm = output["support_selection_false_alarm"]
    assert 0.0 <= false_alarm["linear_support_error_rate"] <= 0.2
    assert 0.0 <= false_alarm["pnl_support_error_rate"] <= 0.2
    selection = output["direction_selection_invariance"]
    assert selection["max_score_absolute_error"] < 1e-12
    assert selection["selected_direction_cosine"] > 1.0 - 1e-12
    assert selection["target_direction_cosine"] > 1.0 - 1e-12
    assert selection["redundant_distractor_candidates"] == selection["base_distractor_candidates"] + 2
    assert selection["legacy_selection_reversal"]
    assert output["multi_support_intersection"]["separating"]
    assert min(multi_cosines) > 1.0 - 1e-8
    assert max(multi_ranks) == 1
    assert not output["missing_coverage"]["separating"]
    assert output["blind_higher_order"]["lattice"]["coverage_0.99"] == 7
    assert output["blind_higher_order"]["pairwise"]["coverage_0.99"] == 6
    assert output["blind_higher_order"]["pairwise"]["cosines_by_source"][3] is None
    assert output["blind_higher_order"]["target_pairwise_support_sizes"] == [2, 2, 2]
    assert output["blind_higher_order"]["target_pairwise_intersection_ranks"] == [2, 2, 2]
    assert output["blind_higher_order"]["target_triple_support"] == [3]
    assert output["blind_higher_order"]["target_triple_intersection_rank"] == 1
    assert output["blind_running_example"]["directions"]["coverage_0.99"] == 5
    assert output["stability_bound"]["n_cases"] >= 20
    assert output["stability_bound"]["max_violation"] < 1e-9


if __name__ == "__main__":
    main()
