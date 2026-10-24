import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment


def forward_nonlinearity(values, coefficients):
    values = np.asarray(values)
    coefficients = np.asarray(coefficients)
    return values + coefficients * np.tanh(values)


def inverse_nonlinearity(values, coefficients, max_iter=40):
    values = np.asarray(values, dtype=float)
    coefficients = np.broadcast_to(np.asarray(coefficients, dtype=float), (values.shape[-1],))
    estimates = np.array(values, copy=True)
    for channel, coefficient in enumerate(coefficients):
        target = values[..., channel]
        estimate = target / (1.0 + coefficient)
        for _ in range(max_iter):
            hyperbolic = np.tanh(estimate)
            residual = estimate + coefficient * hyperbolic - target
            derivative = 1.0 + coefficient * (1.0 - hyperbolic**2)
            step = residual / derivative
            estimate -= step
            if np.max(np.abs(step)) < 1e-11:
                break
        estimates[..., channel] = estimate
    return estimates


def window_rank_one_residuals(observations, window_length):
    observations = np.asarray(observations, dtype=float)
    n_windows = len(observations) // window_length
    n_channels = observations.shape[1]
    windows = observations[:n_windows * window_length].reshape(n_windows, window_length, n_channels)
    windows = windows - np.mean(windows, axis=1, keepdims=True)
    covariances = np.einsum("wti,wtj->wij", windows, windows) / window_length
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariances)[:, ::-1], 0.0)
    total = np.sum(eigenvalues, axis=1) + 1e-15
    return np.sqrt(np.sum(eigenvalues[:, 1:], axis=1) / total)


def calibrate_multichannel_pnl(
    observations,
    window_length,
    keep_fraction=0.75,
    max_iter=50,
    popsize=7,
    local_max_iter=250,
    seed=20261016,
    upper_bound=2.0,
    selection_standard_errors=2.0,
    n_local_starts=4,
    local_start_scale=0.6,
):
    from scipy.optimize import differential_evolution, minimize

    observations = np.asarray(observations, dtype=float)
    n_channels = observations.shape[1]
    n_windows = len(observations) // window_length
    data = observations[:n_windows * window_length]
    n_keep = max(1, int(np.floor(keep_fraction * n_windows)))
    bounds = [(-0.95, upper_bound)] * n_channels
    evaluations = 0

    def residuals(parameters):
        transformed = inverse_nonlinearity(data, parameters, max_iter=25)
        return window_rank_one_residuals(transformed, window_length)

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        values = residuals(parameters)
        lowest = np.partition(values, n_keep - 1)[:n_keep]
        return float(np.mean(lowest))

    def local_search(start):
        return minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": local_max_iter, "xatol": 1e-6, "fatol": 1e-10},
        )

    global_result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-7,
        atol=1e-10,
        polish=False,
        updating="immediate",
        workers=1,
    )
    global_local_result = local_search(global_result.x)
    rng = np.random.default_rng(seed)
    local_results = [local_search(np.zeros(n_channels))]
    for _ in range(max(0, n_local_starts - 1)):
        local_results.append(local_search(rng.uniform(-local_start_scale, local_start_scale, size=n_channels)))
    zero_local_result = local_results[0]
    candidates = [global_result, global_local_result, *local_results]
    best_candidate = min(candidates, key=lambda candidate: float(candidate.fun))
    best_objective = float(best_candidate.fun)
    best_residuals = residuals(best_candidate.x)
    best_lowest = np.partition(best_residuals, n_keep - 1)[:n_keep]
    best_standard_error = float(np.std(best_lowest, ddof=1) / np.sqrt(n_keep)) if n_keep > 1 else 0.0
    selection_threshold = best_objective + selection_standard_errors * best_standard_error
    eligible = [candidate for candidate in candidates if candidate.fun <= selection_threshold]
    selected_result = min(eligible, key=lambda candidate: float(np.mean(candidate.x**2)))
    coefficients = selected_result.x
    final_residuals = residuals(coefficients)
    selected = np.argsort(final_residuals)[:n_keep]
    calibration_mask = np.zeros(n_windows, dtype=bool)
    calibration_mask[selected] = True
    return {
        "inverse_coefficients": np.asarray(coefficients),
        "linearized": inverse_nonlinearity(observations, coefficients),
        "window_residuals": final_residuals,
        "calibration_window_mask": calibration_mask,
        "objective": float(objective(coefficients)),
        "n_evaluations": evaluations,
        "global_success": bool(global_result.success),
        "local_success": bool(global_local_result.success and zero_local_result.success),
        "candidate_objectives": [float(candidate.fun) for candidate in candidates],
        "candidate_coefficients": [np.asarray(candidate.x).tolist() for candidate in candidates],
        "selected_candidate": int(next(index for index, candidate in enumerate(candidates) if candidate is selected_result)),
        "selection_threshold": float(selection_threshold),
        "best_standard_error": best_standard_error,
    }


def kmeans_1d(values, n_clusters, n_iterations=100):
    values = np.asarray(values, dtype=float)
    quantiles = (np.arange(n_clusters) + 0.5) / n_clusters
    centers = np.quantile(values, quantiles)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(n_iterations):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster in range(n_clusters):
            mask = new_labels == cluster
            if np.any(mask):
                new_centers[cluster] = np.mean(values[mask])
            else:
                new_centers[cluster] = values[np.argmax(np.min(distances, axis=1))]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return labels, centers


def ratio_feature(values):
    ratio = np.abs(values[:, 0]) / (np.abs(values[:, 1]) + 1e-10)
    return np.clip(np.log(ratio + 1e-10), -10.0, 10.0)


def cluster_zone_curves(observations, n_sources, zone_length, degree=4, grid_size=12):
    from sklearn.cluster import KMeans

    observations = np.asarray(observations, dtype=float)
    n_samples = len(observations)
    n_zones = int(np.ceil(n_samples / zone_length))
    scale = np.std(observations, axis=0) + 1e-12
    standardized = observations / scale
    features = []
    valid_zones = []
    for zone in range(n_zones):
        start = zone * zone_length
        stop = min(start + zone_length, n_samples)
        values = standardized[start:stop]
        if len(values) < degree + 2 or np.ptp(values[:, 0]) < 1e-6:
            continue
        order = np.argsort(values[:, 0])
        x = values[order, 0]
        y = values[order, 1]
        x_normalized = np.linspace(-1.0, 1.0, len(x))
        coefficients = np.polyfit(x_normalized, y, degree)
        grid = np.linspace(-1.0, 1.0, grid_size)
        curve = np.polyval(coefficients, grid)
        fitted = np.polyval(coefficients, x_normalized)
        residual = np.linalg.norm(y - fitted) / (np.linalg.norm(y) + 1e-12)
        endpoint_slope = (curve[-1] - curve[0]) / 2.0
        features.append(np.concatenate([curve, [residual, endpoint_slope]]))
        valid_zones.append(zone)
    features = np.asarray(features)
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0) + 1e-8
    normalized_features = (features - feature_mean) / feature_std
    zone_clusters = KMeans(n_clusters=n_sources, n_init=30, random_state=20261016).fit_predict(normalized_features)
    labels = np.zeros(n_samples, dtype=int)
    for zone, cluster in zip(valid_zones, zone_clusters):
        start = zone * zone_length
        stop = min(start + zone_length, n_samples)
        labels[start:stop] = cluster
    return labels


def calibrate_two_channel_pnl(observations, n_sources, n_iterations=5, labels=None, n_starts=8, zone_length=None):
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    observations = np.asarray(observations, dtype=float)
    norms = np.linalg.norm(observations, axis=1)
    mask = norms > 0.02 * np.max(norms)
    data = observations[mask]
    labels_are_given = labels is not None
    if labels is None:
        if zone_length is None:
            labels, _ = kmeans_1d(ratio_feature(data), n_sources)
        else:
            labels = cluster_zone_curves(observations, n_sources, zone_length)[mask]
    else:
        labels = np.asarray(labels, dtype=int)[mask]
    coefficients = np.zeros(2)
    rng = np.random.default_rng(20261016)
    objective_cost = np.inf

    for _ in range(n_iterations):
        lower = np.full(2, -0.95)
        upper = np.full(2, 5.0)

        def residual(parameters):
            transformed = inverse_nonlinearity(data, parameters)
            normalized = np.zeros(len(data))
            for source in range(n_sources):
                source_mask = labels == source
                source_data = transformed[source_mask]
                if len(source_data) == 0:
                    continue
                slope = np.sum(source_data[:, 0] * source_data[:, 1]) / (np.sum(source_data[:, 1] ** 2) + 1e-12)
                errors = source_data[:, 0] - slope * source_data[:, 1]
                scale = np.median(np.linalg.norm(source_data, axis=1)) + 1e-8
                normalized[source_mask] = errors / scale
            return normalized

        starts = [coefficients]
        starts.extend(rng.uniform(-0.8, 1.5, size=(max(0, n_starts - 1), 2)))
        best_solution = None
        for start in starts:
            solution = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=180,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            if best_solution is None or solution.cost < best_solution.cost:
                best_solution = solution
        coefficients = best_solution.x
        objective_cost = float(best_solution.cost)
        transformed = inverse_nonlinearity(data, coefficients)
        if labels_are_given:
            break
        new_labels, _ = kmeans_1d(ratio_feature(transformed), n_sources)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    transformed = inverse_nonlinearity(data, coefficients)
    line_errors = []
    for source in range(n_sources):
        source_data = transformed[labels == source]
        if len(source_data) < 2:
            line_errors.append(np.inf)
            continue
        centered = source_data - np.mean(source_data, axis=0)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        line_errors.append(float(singular_values[-1] / (singular_values[0] + 1e-12)))
    bound_hits = np.isclose(coefficients, lower, atol=1e-6, rtol=0.0) | np.isclose(
        coefficients, upper, atol=1e-6, rtol=0.0
    )
    final_objective_cost = float(0.5 * np.sum(residual(coefficients) ** 2))
    zero_calibration_cost = float(0.5 * np.sum(residual(np.zeros(2)) ** 2))
    return {
        "inverse_coefficients": coefficients,
        "linearized": transformed,
        "labels": labels,
        "mask": mask,
        "line_errors": np.asarray(line_errors),
        "objective_cost": final_objective_cost,
        "zero_calibration_cost": zero_calibration_cost,
        "coefficient_bound_hit": bool(np.any(bound_hits)),
        "coefficient_bound_hit_channels": bound_hits.tolist(),
    }


def estimate_columns_from_linearized(values, n_sources):
    values = np.asarray(values, dtype=float)
    labels, _ = kmeans_1d(ratio_feature(values), n_sources)
    columns = []
    for source in range(n_sources):
        source_data = values[labels == source]
        if len(source_data) == 0:
            columns.append(np.zeros(2))
            continue
        covariance = source_data.T @ source_data / len(source_data)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        columns.append(eigenvectors[:, np.argmax(eigenvalues)])
    return np.column_stack(columns), labels


def match_directions(true_matrix, estimated_matrix):
    true_normalized = true_matrix / np.linalg.norm(true_matrix, axis=0, keepdims=True)
    estimated_normalized = estimated_matrix / np.maximum(np.linalg.norm(estimated_matrix, axis=0, keepdims=True), 1e-12)
    cosine = np.abs(true_normalized.T @ estimated_normalized)
    rows, columns = linear_sum_assignment(-cosine)
    matches = []
    for row, column in zip(rows, columns):
        matches.append({"true_index": int(row), "estimated_index": int(column), "cosine": float(cosine[row, column])})
    return matches


def generate_two_regime_pnl(n_samples=3000, noise_std=0.0, seed=20261016, zone_length=100):
    rng = np.random.default_rng(seed)
    mixing = np.array([[1.0, 0.8, 0.5], [0.4, -0.6, 1.0]])
    regimes = []
    specifications = [
        {"active": (0, 1), "coefficients": np.array([0.8, -0.45])},
        {"active": (1, 2), "coefficients": np.array([-0.35, 0.9])},
    ]
    for regime, specification in enumerate(specifications):
        active = specification["active"]
        sources = np.zeros((n_samples, 3))
        zone_labels = rng.integers(0, len(active), size=int(np.ceil(n_samples / zone_length)))
        labels = np.repeat(zone_labels, zone_length)[:n_samples]
        for local_index, source_index in enumerate(active):
            mask = labels == local_index
            sources[mask, source_index] = rng.laplace(scale=1.0, size=np.sum(mask))
        linear = sources @ mixing.T
        observations = forward_nonlinearity(linear, specification["coefficients"])
        observations += noise_std * rng.normal(size=observations.shape)
        regimes.append({
            "observations": observations,
            "sources": sources,
            "linear": linear,
            "active": active,
            "labels": labels,
            "zone_length": zone_length,
            "forward_coefficients": specification["coefficients"],
            "regime": regime,
        })
    return mixing, regimes
