from itertools import combinations

import numpy as np
from scipy.linalg import orth


def normalize_supports(supports):
    return tuple(sorted({tuple(sorted(int(i) for i in support)) for support in supports}))


def containing_supports(supports, index):
    return tuple(support for support in normalize_supports(supports) if index in support)


def support_intersection(supports, index):
    containing = containing_supports(supports, index)
    if not containing:
        return None
    result = set(containing[0])
    for support in containing[1:]:
        result.intersection_update(support)
    return frozenset(result)


def is_separating(supports, n_sources):
    return all(support_intersection(supports, i) == frozenset({i}) for i in range(n_sources))


def is_full_spark(matrix, tolerance=1e-9):
    m, n = matrix.shape
    if n < m:
        return False
    return all(np.linalg.matrix_rank(matrix[:, list(columns)], tol=tolerance) == m for columns in combinations(range(n), m))


def spark(matrix, tolerance=1e-9):
    m, n = matrix.shape
    for size in range(1, m + 2):
        for columns in combinations(range(n), size):
            if np.linalg.matrix_rank(matrix[:, list(columns)], tol=tolerance) < size:
                return size
    return m + 1


def generate_full_spark(m, n, rng, min_determinant=1e-3, max_attempts=10000):
    for _ in range(max_attempts):
        matrix = rng.normal(size=(m, n))
        if n == m:
            if abs(np.linalg.det(matrix)) > min_determinant:
                return matrix
            continue
        if all(abs(np.linalg.det(matrix[:, list(columns)])) > min_determinant for columns in combinations(range(n), m)):
            return matrix
    raise RuntimeError("Failed to generate a well-conditioned full-spark matrix")


def support_basis(matrix, support):
    return orth(matrix[:, list(support)])


def intersect_subspaces(left, right, tolerance=1e-9):
    m = left.shape[0]
    joint = np.concatenate([left, -right], axis=1)
    _, singular_values, vh = np.linalg.svd(joint, full_matrices=True)
    rank = int(np.sum(singular_values > tolerance))
    null_basis = vh[rank:].T
    if null_basis.shape[1] == 0:
        return np.zeros((m, 0))
    vectors = left @ null_basis[: left.shape[1], :]
    norms = np.linalg.norm(vectors, axis=0)
    vectors = vectors[:, norms > tolerance]
    if vectors.shape[1] == 0:
        return np.zeros((m, 0))
    return orth(vectors, rcond=tolerance)


def intersection_conditioning(matrix, left_support, right_support):
    left = support_basis(matrix, left_support)
    right = support_basis(matrix, right_support)
    intersection = intersect_subspaces(left, right)
    left_perp = orth((np.eye(matrix.shape[0]) - intersection @ intersection.T) @ left)
    residual_map = (np.eye(matrix.shape[0]) - right @ right.T) @ left_perp
    gamma = float(np.min(np.linalg.svd(residual_map, compute_uv=False)))
    distance = float(
        np.linalg.norm(left @ left.T - right @ right.T, ord="fro") / np.sqrt(2 * left.shape[1])
    )
    return {
        "intersection_gamma": gamma,
        "cross_subspace_distance": distance,
        "intersection_rank": int(intersection.shape[1]),
    }


def reveal_column(matrix, supports, index, tolerance=1e-9):
    containing = containing_supports(supports, index)
    if not containing:
        return np.zeros((matrix.shape[0], 0))
    revealed = support_basis(matrix, containing[0])
    for support in containing[1:]:
        revealed = intersect_subspaces(revealed, support_basis(matrix, support), tolerance)
        if revealed.shape[1] == 0:
            break
    return revealed


def direction_cosine(left, right):
    if left.size == 0 or right.size == 0:
        return 0.0
    u = left[:, 0] / np.linalg.norm(left[:, 0])
    v = right[:, 0] / np.linalg.norm(right[:, 0])
    return float(abs(u @ v))


def reveal_all_columns(matrix, supports):
    revealed = [reveal_column(matrix, supports, i) for i in range(matrix.shape[1])]
    cosines = [direction_cosine(revealed[i], matrix[:, [i]]) for i in range(matrix.shape[1])]
    ranks = [int(basis.shape[1]) for basis in revealed]
    return revealed, cosines, ranks


def construct_ambiguity(matrix, supports, index, alpha=0.1):
    intersection = support_intersection(supports, index)
    if intersection is None or intersection == frozenset({index}):
        raise ValueError("The selected source is already separated or absent")
    other = next(i for i in intersection if i != index)
    transform = np.eye(matrix.shape[1])
    transform[other, index] = alpha
    alternative = matrix @ transform
    max_projection_error = 0.0
    for support in normalize_supports(supports):
        original_basis = support_basis(matrix, support)
        alternative_basis = support_basis(alternative, support)
        projection_error = np.linalg.norm(
            original_basis @ original_basis.T - alternative_basis @ alternative_basis.T,
            ord="fro",
        )
        max_projection_error = max(max_projection_error, float(projection_error))
    return {
        "alternative": alternative,
        "transform": transform,
        "other_index": other,
        "max_projection_error": max_projection_error,
        "original_cosine": direction_cosine(alternative[:, [index]], matrix[:, [index]]),
        "full_spark": is_full_spark(alternative),
    }


def recover_sparse_sample(matrix, observation, k, noise_std=None, false_alarm=0.01):
    from scipy.stats import chi2

    m, n = matrix.shape
    if n == 0:
        return np.zeros(0), None, float(np.linalg.norm(observation) ** 2)
    k = min(k, n)
    best_error = np.inf
    best_support = None
    best_source = None
    candidates_by_size = {}
    for size in range(1, k + 1):
        size_error = np.inf
        size_candidate = None
        for support in combinations(range(n), size):
            columns = matrix[:, list(support)]
            coefficients, _, rank, _ = np.linalg.lstsq(columns, observation, rcond=None)
            error = float(np.linalg.norm(observation - columns @ coefficients) ** 2)
            if error < size_error:
                size_error = error
                source = np.zeros(n)
                source[list(support)] = coefficients
                size_candidate = (source, support, error, m - rank)
            if error < best_error:
                best_error = error
                best_support = support
                best_source = np.zeros(n)
                best_source[list(support)] = coefficients
        candidates_by_size[size] = size_candidate
    if noise_std is not None and noise_std > 0:
        for size in range(1, k + 1):
            source, support, error, degrees_freedom = candidates_by_size[size]
            threshold = noise_std**2 * chi2.ppf(1.0 - false_alarm, degrees_freedom)
            if error <= threshold:
                return source, support, error
    return best_source, best_support, best_error


def recover_sparse_sample_omp(matrix, observation, k, tolerance=1e-10):
    matrix = np.asarray(matrix, dtype=float)
    observation = np.asarray(observation, dtype=float)
    n = matrix.shape[1]
    column_norms = np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
    support = []
    residual = observation.copy()
    coefficients = np.zeros(0)
    for _ in range(k):
        if np.linalg.norm(residual) <= tolerance or len(support) == n:
            break
        correlations = np.abs(matrix.T @ residual) / column_norms
        correlations[support] = -np.inf
        selected = int(np.argmax(correlations))
        support.append(selected)
        coefficients, *_ = np.linalg.lstsq(matrix[:, support], observation, rcond=None)
        residual = observation - matrix[:, support] @ coefficients
    source = np.zeros(n)
    source[support] = coefficients
    return source, tuple(sorted(support)), float(np.linalg.norm(residual) ** 2)
