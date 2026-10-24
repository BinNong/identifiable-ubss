from itertools import combinations

import numpy as np
from scipy.linalg import orth


def estimate_window_bases(observations, window_length, max_rank, relative_threshold=1e-3):
    observations = np.asarray(observations, dtype=float)
    bases = []
    window_indices = []
    eigenvalue_ratios = []
    n_windows = len(observations) // window_length
    for window in range(n_windows):
        values = observations[window * window_length:(window + 1) * window_length]
        values = values - np.mean(values, axis=0)
        covariance = values.T @ values / len(values)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        ratios = eigenvalues / (eigenvalues[0] + 1e-15)
        rank = min(max_rank, int(np.sum(ratios > relative_threshold)))
        if rank == 0:
            continue
        bases.append(eigenvectors[:, :rank])
        window_indices.append(window)
        eigenvalue_ratios.append(ratios)
    return bases, window_indices, eigenvalue_ratios


def approximate_intersection(left, right, cosine_threshold=0.98):
    cross = left.T @ right
    u, singular_values, _ = np.linalg.svd(cross, full_matrices=False)
    keep = singular_values > cosine_threshold
    if not np.any(keep):
        return np.zeros((left.shape[0], 0)), np.zeros(0)
    vectors = left @ u[:, keep]
    return orth(vectors), singular_values[keep]


def _intersection_step(left, right, cosine_threshold=0.98):
    cross = left.T @ right
    u, singular_values, _ = np.linalg.svd(cross, full_matrices=False)
    keep = singular_values > cosine_threshold
    s1 = float(singular_values[0]) if singular_values.size else 0.0
    s2 = float(singular_values[1]) if singular_values.size > 1 else 0.0
    spectrum = {"s1": s1, "s2": s2, "margin": s1 - s2}
    if not np.any(keep):
        return np.zeros((left.shape[0], 0)), np.zeros(0), spectrum
    vectors = left @ u[:, keep]
    return orth(vectors), singular_values[keep], spectrum


def _intersection_gamma_estimate(left_before, right, direction):
    if left_before.shape[1] < 2:
        return None
    left_perp = orth((np.eye(left_before.shape[0]) - np.outer(direction, direction)) @ left_before)
    if left_perp.shape[1] == 0:
        return None
    residual_map = (np.eye(right.shape[0]) - right @ right.T) @ left_perp
    return float(np.min(np.linalg.svd(residual_map, compute_uv=False)))


def cluster_direction_candidates(candidates, cosine_threshold=0.995):
    clusters = []
    for vector, weight, metadata in candidates:
        vector = vector / np.linalg.norm(vector)
        best_index = None
        best_cosine = -np.inf
        for index, cluster in enumerate(clusters):
            cosine = abs(cluster["direction"] @ vector)
            if cosine > best_cosine:
                best_cosine = cosine
                best_index = index
        if best_index is not None and best_cosine >= cosine_threshold:
            cluster = clusters[best_index]
            aligned = vector * np.sign(cluster["direction"] @ vector)
            cluster["direction"] = cluster["direction"] * cluster["weight"] + aligned * weight
            cluster["direction"] /= np.linalg.norm(cluster["direction"])
            cluster["weight"] += weight
            cluster["member_weights"].append(float(weight))
            cluster["members"].append(metadata)
        else:
            clusters.append({
                "direction": vector.copy(),
                "weight": float(weight),
                "member_weights": [float(weight)],
                "members": [metadata],
            })
    return clusters


def recover_directions_from_bases(bases, n_sources, window_indices=None, intersection_threshold=0.98, cluster_threshold=0.995, min_members=2):
    if window_indices is None:
        window_indices = list(range(len(bases)))
    candidates = []
    for index, basis in enumerate(bases):
        if basis.shape[1] == 1:
            candidates.append((basis[:, 0], 1.0, {"windows": (window_indices[index],), "kind": "singleton"}))
    for left_index, right_index in combinations(range(len(bases)), 2):
        intersection, strengths = approximate_intersection(bases[left_index], bases[right_index], intersection_threshold)
        if intersection.shape[1] == 1:
            candidates.append((intersection[:, 0], float(strengths[0]), {
                "windows": (window_indices[left_index], window_indices[right_index]),
                "kind": "intersection",
            }))
    clusters = cluster_direction_candidates(candidates, cluster_threshold)
    clusters = [cluster for cluster in clusters if len(cluster["members"]) >= min_members]
    clusters.sort(key=lambda cluster: len(cluster["members"]), reverse=True)
    selected_indices, _ = select_directions_by_family_consistency(
        clusters,
        bases,
        n_sources,
        2,
    )
    _cluster_consistency_membership(clusters, selected_indices)
    selected = [clusters[index] for index in selected_indices]
    if selected:
        directions = np.column_stack([cluster["direction"] for cluster in selected])
    else:
        directions = np.zeros((bases[0].shape[0], 0))
    return directions, selected, clusters


def _projector(basis):
    return basis @ basis.T


def _subspace_distance(left, right):
    rank = left.shape[1]
    return float(np.linalg.norm(_projector(left) - _projector(right), ord="fro") / np.sqrt(2 * rank))


def cluster_window_subspaces(bases, window_indices=None, distance_threshold=0.15):
    from sklearn.cluster import AgglomerativeClustering

    if window_indices is None:
        window_indices = list(range(len(bases)))
    prototypes = []
    assignments = np.full(len(bases), -1, dtype=int)
    for rank in sorted({basis.shape[1] for basis in bases}):
        indices = [index for index, basis in enumerate(bases) if basis.shape[1] == rank]
        if not indices:
            continue
        if len(indices) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            distances = np.zeros((len(indices), len(indices)))
            for left_position, left_index in enumerate(indices):
                for right_position in range(left_position + 1, len(indices)):
                    right_index = indices[right_position]
                    distance = _subspace_distance(bases[left_index], bases[right_index])
                    distances[left_position, right_position] = distance
                    distances[right_position, left_position] = distance
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = clustering.fit_predict(distances)
        for label in np.unique(labels):
            members = [indices[position] for position in np.flatnonzero(labels == label)]
            mean_projector = np.mean([_projector(bases[index]) for index in members], axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(mean_projector)
            order = np.argsort(eigenvalues)[::-1][:rank]
            prototype = {
                "basis": eigenvectors[:, order],
                "windows": tuple(int(window_indices[index]) for index in members),
                "base_indices": tuple(int(index) for index in members),
                "rank": rank,
            }
            assignments[members] = len(prototypes)
            prototypes.append(prototype)
    return prototypes, assignments


def _finalize_direction_clusters(clusters, prototypes, min_members, score_rule="max"):
    if score_rule not in {"max", "mean", "legacy"}:
        raise ValueError(f"Unknown direction score rule: {score_rule}")
    finalized = []
    for cluster in clusters:
        windows = sorted({window for member in cluster["members"] for window in member.get("windows", ())})
        prototype_ids = sorted({prototype for member in cluster["members"] for prototype in member.get("prototypes", ())})
        cluster["n_windows"] = len(windows)
        cluster["n_prototypes"] = len(prototype_ids)
        cluster["prototype_ids"] = prototype_ids
        cluster["n_candidates"] = len(cluster["members"])
        cluster["mean_score"] = float(cluster["weight"] / cluster["n_candidates"])
        cluster["max_score"] = float(max(cluster["member_weights"]))
        cluster["legacy_score"] = float(cluster["weight"] * (1.0 + np.log1p(cluster["n_candidates"])))
        cluster["score"] = float(cluster[f"{score_rule}_score"])
        if cluster["n_candidates"] >= min_members:
            finalized.append(cluster)
    finalized.sort(key=lambda cluster: (cluster["score"], cluster["n_windows"]), reverse=True)
    return finalized


def _direction_cluster_diagnostics(clusters, n_sources):
    legacy_order = sorted(
        range(len(clusters)),
        key=lambda index: (clusters[index]["legacy_score"], clusters[index]["n_windows"]),
        reverse=True,
    )
    legacy_ranks = {cluster_index: rank + 1 for rank, cluster_index in enumerate(legacy_order)}
    diagnostic_limit = min(len(clusters), n_sources + 2)
    retained = set(range(diagnostic_limit)) | set(legacy_order[:diagnostic_limit])
    records = []
    for index in sorted(retained):
        cluster = clusters[index]
        depths = [member.get("depth", 1) for member in cluster["members"]]
        s1_values = [member["s1_min"] for member in cluster["members"] if member.get("s1_min") is not None]
        epsilon_values = [member["epsilon_equiv"] for member in cluster["members"] if member.get("epsilon_equiv") is not None]
        gamma_values = [member["gamma_min"] for member in cluster["members"] if member.get("gamma_min") is not None]
        margin_values = [member["margin_min"] for member in cluster["members"] if member.get("margin_min") is not None]
        records.append({
            "direction": cluster["direction"].tolist(),
            "weight": float(cluster["weight"]),
            "score": float(cluster["score"]),
            "mean_score": float(cluster["mean_score"]),
            "max_score": float(cluster["max_score"]),
            "legacy_score": float(cluster["legacy_score"]),
            "n_candidates": int(cluster["n_candidates"]),
            "n_windows": int(cluster["n_windows"]),
            "n_prototypes": int(cluster["n_prototypes"]),
            "prototype_ids": [int(value) for value in cluster["prototype_ids"]],
            "kinds": sorted({member.get("kind", "unknown") for member in cluster["members"]}),
            "min_depth": int(min(depths)),
            "max_depth": int(max(depths)),
            "s1_min": float(min(s1_values)) if s1_values else None,
            "epsilon_max": float(max(epsilon_values)) if epsilon_values else None,
            "gamma_min": float(min(gamma_values)) if gamma_values else None,
            "margin_min": float(min(margin_values)) if margin_values else None,
            "selection_rank": index + 1,
            "selected": bool(cluster.get("selected", index < n_sources)),
            "legacy_selection_rank": legacy_ranks[index],
            "legacy_selected": legacy_ranks[index] <= n_sources,
        })
    return records


def _window_subset_residuals(directions, bases, max_depth, max_subsets=5000):
    from itertools import combinations as _combinations

    n_clusters = directions.shape[1]
    subset_sizes = range(1, max_depth + 1)
    subsets = [
        combo
        for size in subset_sizes
        for combo in _combinations(range(n_clusters), size)
    ]
    if len(subsets) > max_subsets:
        return None
    membership = np.zeros((len(subsets), n_clusters), dtype=bool)
    projectors = []
    for position, combo in enumerate(subsets):
        membership[position, list(combo)] = True
        component = directions[:, list(combo)]
        q, _ = np.linalg.qr(component)
        projectors.append(q @ q.T)
    per_window = []
    for basis in bases:
        rank = basis.shape[1]
        residuals = []
        for combo, projector in zip(subsets, projectors):
            if len(combo) < rank:
                residuals.append(np.inf)
                continue
            residuals.append(float(np.linalg.norm(basis - projector @ basis) / np.sqrt(rank)))
        per_window.append((membership, np.asarray(residuals)))
    return per_window


def select_directions_by_family_consistency(
    clusters,
    bases,
    n_sources,
    max_depth,
    tau_min=0.05,
    relative_factor=2.0,
    max_subsets=5000,
    max_selections=30000,
    max_selection_clusters=40,
):
    from itertools import combinations as _combinations
    from math import comb

    info = {
        "rule": "family_consistency",
        "tau_min": float(tau_min),
        "relative_factor": float(relative_factor),
        "n_windows": int(len(bases)),
        "search": "exhaustive",
    }
    n_clusters = len(clusters)
    if n_clusters == 0 or not bases:
        info["explained_windows"] = 0
        return [], info
    if n_clusters <= n_sources:
        info["explained_windows"] = int(len(bases))
        return list(range(n_clusters)), info
    order = sorted(range(n_clusters), key=lambda index: clusters[index].get("score", clusters[index]["weight"]), reverse=True)
    pool = order[:max_selection_clusters]
    remap = {pool[position]: position for position in range(len(pool))}
    pool_directions = np.column_stack([
        clusters[index]["direction"] / np.linalg.norm(clusters[index]["direction"]) for index in pool
    ])
    per_window = _window_subset_residuals(pool_directions, bases, max_depth, max_subsets)
    if per_window is None:
        info["search"] = "score_fallback"
        info["explained_windows"] = None
        return sorted(pool[:n_sources]), info
    scores = np.asarray([clusters[index].get("score", clusters[index]["weight"]) for index in pool])

    good_subsets = []
    for membership, residuals in per_window:
        best = float(np.min(residuals))
        threshold = max(tau_min, relative_factor * best)
        good_subsets.append(membership[residuals <= threshold])

    def explained_count(selection_mask):
        total = 0
        for membership in good_subsets:
            if membership.size and bool(np.any(~(membership & ~selection_mask).any(axis=1))):
                total += 1
        return total

    n_pool = len(pool)
    n_selections = comb(n_pool, n_sources) if n_pool >= n_sources else 0
    best_selection = None
    best_key = None
    if n_pool <= n_sources:
        best_selection = list(range(n_pool))
        mask = np.zeros(n_pool, dtype=bool)
        mask[best_selection] = True
        best_key = (explained_count(mask), float(sum(scores[best_selection])))
    elif n_selections <= max_selections:
        for combo in _combinations(range(n_pool), n_sources):
            mask = np.zeros(n_pool, dtype=bool)
            mask[list(combo)] = True
            count = explained_count(mask)
            key = (count, float(scores[list(combo)].sum()))
            if best_key is None or key > best_key:
                best_key = key
                best_selection = list(combo)
    else:
        info["search"] = "greedy"
        chosen = []
        remaining = set(range(n_pool))
        while len(chosen) < n_sources:
            best_candidate = None
            best_candidate_key = None
            for candidate in sorted(remaining):
                trial = chosen + [candidate]
                mask = np.zeros(n_pool, dtype=bool)
                mask[trial] = True
                key = (explained_count(mask), float(scores[candidate]))
                if best_candidate_key is None or key > best_candidate_key:
                    best_candidate_key = key
                    best_candidate = candidate
            chosen.append(best_candidate)
            remaining.remove(best_candidate)
        best_selection = chosen
        mask = np.zeros(n_pool, dtype=bool)
        mask[best_selection] = True
        best_key = (explained_count(mask), float(scores[best_selection].sum()))
    info["explained_windows"] = int(best_key[0])
    info["selected_score_sum"] = float(best_key[1])
    return sorted(pool[index] for index in best_selection), info


def _cluster_consistency_membership(clusters, selected_indices):
    selected_set = set(selected_indices)
    for cluster in clusters:
        cluster["selected"] = False
    for index in selected_set:
        clusters[index]["selected"] = True


def recover_directions_intersection_lattice(
    bases,
    n_sources,
    sparsity,
    window_indices=None,
    subspace_threshold=0.15,
    intersection_threshold=0.98,
    cluster_threshold=0.995,
    min_members=1,
    max_depth=None,
    score_rule="max",
    selection_rule="family_consistency",
):
    if window_indices is None:
        window_indices = list(range(len(bases)))
    if not bases:
        return np.zeros((0, 0)), [], [], {
            "n_prototypes": 0,
            "n_candidates": 0,
            "direction_clusters": [],
        }
    prototypes, assignments = cluster_window_subspaces(bases, window_indices, subspace_threshold)
    candidates = []
    for index, prototype in enumerate(prototypes):
        if prototype["rank"] == 1:
            weight = float(max(1, len(prototype["windows"])))
            candidates.append((prototype["basis"][:, 0], weight, {
                "windows": prototype["windows"],
                "prototypes": (index,),
                "kind": "singleton",
                "depth": 1,
                "s1_min": 1.0,
                "epsilon_equiv": 0.0,
                "margin_min": None,
                "gamma_min": None,
            }))

    higher = [index for index, prototype in enumerate(prototypes) if prototype["rank"] > 1]
    depth_limit = min(sparsity, len(higher)) if max_depth is None else min(max_depth, len(higher))
    for depth in range(2, depth_limit + 1):
        for higher_combo in combinations(range(len(higher)), depth):
            prototype_combo = tuple(higher[position] for position in higher_combo)
            basis = prototypes[prototype_combo[0]]["basis"]
            strength = 1.0
            chain_s1 = 1.0
            chain_margin = np.inf
            chain_gamma = np.inf
            valid = True
            for prototype_index in prototype_combo[1:]:
                before = basis
                basis, strengths, spectrum = _intersection_step(
                    basis,
                    prototypes[prototype_index]["basis"],
                    intersection_threshold,
                )
                if basis.shape[1] == 0:
                    valid = False
                    break
                strength = min(strength, float(np.min(strengths)))
                chain_s1 = min(chain_s1, spectrum["s1"])
                chain_margin = min(chain_margin, spectrum["margin"])
                if basis.shape[1] == 1:
                    gamma_step = _intersection_gamma_estimate(
                        before, prototypes[prototype_index]["basis"], basis[:, 0]
                    )
                    if gamma_step is not None:
                        chain_gamma = min(chain_gamma, gamma_step)
            if not valid or basis.shape[1] != 1:
                continue
            windows = tuple(sorted({window for index in prototype_combo for window in prototypes[index]["windows"]}))
            candidates.append((basis[:, 0], strength * float(len(windows)), {
                "windows": windows,
                "prototypes": prototype_combo,
                "kind": "lattice",
                "depth": depth,
                "s1_min": float(chain_s1),
                "epsilon_equiv": float(np.sqrt(max(0.0, 1.0 - chain_s1**2))),
                "margin_min": float(chain_margin),
                "gamma_min": float(chain_gamma) if np.isfinite(chain_gamma) else None,
            }))

    candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    clusters = cluster_direction_candidates(candidates, cluster_threshold)
    clusters = _finalize_direction_clusters(clusters, prototypes, min_members, score_rule)
    if selection_rule == "family_consistency":
        selected_indices, selection_info = select_directions_by_family_consistency(
            clusters,
            bases,
            n_sources,
            sparsity,
        )
        _cluster_consistency_membership(clusters, selected_indices)
        selected = sorted((clusters[index] for index in selected_indices), key=lambda cluster: cluster["score"], reverse=True)
    elif selection_rule == "score":
        selection_info = {"rule": "score", "n_windows": int(len(bases))}
        for position, cluster in enumerate(clusters):
            cluster["selected"] = position < n_sources
        selected = clusters[:n_sources]
    else:
        raise ValueError(f"Unknown selection rule: {selection_rule}")
    if selected:
        directions = np.column_stack([cluster["direction"] for cluster in selected])
    else:
        directions = np.zeros((bases[0].shape[0], 0))
    diagnostics = {
        "n_prototypes": len(prototypes),
        "n_rank_one_prototypes": sum(prototype["rank"] == 1 for prototype in prototypes),
        "n_candidates": len(candidates),
        "n_clusters": len(clusters),
        "cluster_count_within_source_count": bool(len(clusters) <= n_sources),
        "selection": selection_info,
        "direction_clusters": _direction_cluster_diagnostics(clusters, n_sources),
        "assignments": assignments,
        "prototypes": prototypes,
    }
    return directions, selected, clusters, diagnostics


def recover_rank_one_directions(
    bases,
    n_sources,
    window_indices=None,
    subspace_threshold=0.15,
):
    if window_indices is None:
        window_indices = list(range(len(bases)))
    prototypes, assignments = cluster_window_subspaces(bases, window_indices, subspace_threshold)
    rank_one = [prototype for prototype in prototypes if prototype["rank"] == 1]
    rank_one.sort(key=lambda prototype: len(prototype["windows"]), reverse=True)
    selected = rank_one[:n_sources]
    if selected:
        directions = np.column_stack([prototype["basis"][:, 0] for prototype in selected])
    else:
        directions = np.zeros((bases[0].shape[0], 0))
    diagnostics = {
        "n_prototypes": len(prototypes),
        "n_rank_one_prototypes": len(rank_one),
        "assignments": assignments,
    }
    return directions, selected, diagnostics


def generate_running_example(
    blocks_per_support=8,
    block_length=80,
    noise_std=0.0,
    seed=20261016,
    min_calibration_entry=0.2,
):
    from coverage import generate_full_spark

    rng = np.random.default_rng(seed)
    for _ in range(1000):
        mixing = generate_full_spark(4, 5, rng, min_determinant=0.02)
        if np.min(np.abs(mixing[:, :4])) >= min_calibration_entry:
            break
    else:
        raise RuntimeError("Failed to generate a sufficiently calibrable mixing matrix")
    calibration = [(0,), (1,), (2,), (3,)]
    support_families = [calibration + [(0, 4)], calibration + [(1, 4)]]
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
                observations = sources @ mixing.T
                observations += noise_std * rng.normal(size=observations.shape)
                observation_blocks.append(observations)
                source_blocks.append(sources)
                block_supports.append(support)
        regimes.append({
            "observations": np.concatenate(observation_blocks, axis=0),
            "sources": np.concatenate(source_blocks, axis=0),
            "block_supports": block_supports,
            "block_length": block_length,
            "regime": regime,
        })
    return mixing, regimes


def generate_higher_order_example(blocks_per_support=4, block_length=100, seed=20261016):
    from coverage import generate_full_spark

    rng = np.random.default_rng(seed)
    mixing = generate_full_spark(6, 7, rng, min_determinant=0.005)
    supports = [
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0,),
        (1,),
        (2,),
        (4,),
        (5,),
        (6,),
    ]
    observation_blocks = []
    source_blocks = []
    block_supports = []
    for support in supports:
        for _ in range(blocks_per_support):
            sources = np.zeros((block_length, 7))
            for index in support:
                sources[:, index] = rng.normal(size=block_length)
            observations = sources @ mixing.T
            observation_blocks.append(observations)
            source_blocks.append(sources)
            block_supports.append(support)
    return mixing, {
        "observations": np.concatenate(observation_blocks, axis=0),
        "sources": np.concatenate(source_blocks, axis=0),
        "block_supports": block_supports,
        "block_length": block_length,
    }


def generate_scaling_example(n_sources, blocks_per_support=4, block_length=80, seed=20261016):
    from coverage import generate_full_spark

    if n_sources < 5:
        raise ValueError("The scaling example requires at least five sources")
    rng = np.random.default_rng(seed)
    n_observations = n_sources - 1
    mixing = generate_full_spark(n_observations, n_sources, rng, min_determinant=1e-5)
    calibration = [(index,) for index in range(n_sources - 1)]
    support_families = [
        calibration + [(0, n_sources - 1)],
        calibration + [(1, n_sources - 1)],
    ]
    regimes = []
    for regime, supports in enumerate(support_families):
        observation_blocks = []
        source_blocks = []
        for support in supports:
            for _ in range(blocks_per_support):
                sources = np.zeros((block_length, n_sources))
                for index in support:
                    sources[:, index] = rng.normal(size=block_length)
                observation_blocks.append(sources @ mixing.T)
                source_blocks.append(sources)
        regimes.append({
            "observations": np.concatenate(observation_blocks, axis=0),
            "sources": np.concatenate(source_blocks, axis=0),
            "block_length": block_length,
            "regime": regime,
        })
    return mixing, regimes
