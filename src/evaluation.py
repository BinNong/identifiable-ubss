import numpy as np

from coverage import recover_sparse_sample, recover_sparse_sample_omp
from pnl import match_directions


def align_matrix_to_truth(true_matrix, estimated_matrix):
    matches = match_directions(true_matrix, estimated_matrix)
    aligned = np.zeros_like(true_matrix)
    for match in matches:
        true_index = match["true_index"]
        estimated_index = match["estimated_index"]
        vector = estimated_matrix[:, estimated_index]
        sign = np.sign(true_matrix[:, true_index] @ vector)
        aligned[:, true_index] = sign * vector
    return aligned, matches


def align_sources_to_truth(true_matrix, estimated_matrix, estimated_sources):
    matches = match_directions(true_matrix, estimated_matrix)
    aligned = np.zeros((len(estimated_sources), true_matrix.shape[1]))
    for match in matches:
        true_index = match["true_index"]
        estimated_index = match["estimated_index"]
        sign = np.sign(true_matrix[:, true_index] @ estimated_matrix[:, estimated_index])
        aligned[:, true_index] = sign * estimated_sources[:, estimated_index]
    return aligned


def estimate_observation_noise_std(observations, window_length, max_rank, relative=False):
    observations = np.asarray(observations, dtype=float)
    n_windows = len(observations) // window_length
    n_channels = observations.shape[1]
    windows = observations[:n_windows * window_length].reshape(n_windows, window_length, n_channels)
    windows = windows - np.mean(windows, axis=1, keepdims=True)
    covariances = np.einsum("wti,wtj->wij", windows, windows) / window_length
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariances), 0.0)
    n_noise = max(1, n_channels - max_rank)
    noise_variance = float(np.median(eigenvalues[:, :n_noise]))
    if relative:
        leading_variance = float(np.median(eigenvalues[:, -1]))
        noise_variance /= leading_variance * (1.0 + 1e-15)
    return float(np.sqrt(max(noise_variance, 0.0)))


def recover_sources(observations, mixing, k, noise_std=None, solver="exhaustive"):
    observations = np.asarray(observations, dtype=float)
    estimates = np.zeros((len(observations), mixing.shape[1]))
    supports = []
    residuals = []
    for index, observation in enumerate(observations):
        if solver == "exhaustive":
            source, support, residual = recover_sparse_sample(mixing, observation, k, noise_std=noise_std)
        elif solver == "omp":
            source, support, residual = recover_sparse_sample_omp(mixing, observation, k)
        else:
            raise ValueError(solver)
        estimates[index] = source
        supports.append(support)
        residuals.append(residual)
    return estimates, supports, np.asarray(residuals)


def score_source_recovery(estimated_sources, true_sources, threshold=1e-3):
    estimated_sources = np.asarray(estimated_sources)
    true_sources = np.asarray(true_sources)
    support_matches = []
    correlations = []
    scaled_errors = []
    true_activity = np.abs(true_sources) > threshold
    estimated_activity = np.abs(estimated_sources) > threshold
    true_positive = int(np.sum(true_activity & estimated_activity))
    false_positive = int(np.sum(~true_activity & estimated_activity))
    false_negative = int(np.sum(true_activity & ~estimated_activity))
    precision = true_positive / (true_positive + false_positive + 1e-12)
    recall = true_positive / (true_positive + false_negative + 1e-12)
    for source in range(true_sources.shape[1]):
        active = true_activity[:, source]
        if not np.any(active):
            correlations.append(None)
            scaled_errors.append(None)
            continue
        truth = true_sources[active, source]
        estimate = estimated_sources[active, source]
        if np.std(truth) < 1e-12 or np.std(estimate) < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(truth, estimate)[0, 1]))
        scale = np.sum(truth * estimate) / (np.sum(estimate**2) + 1e-12)
        scaled_errors.append(float(np.linalg.norm(truth - scale * estimate) / (np.linalg.norm(truth) + 1e-12)))
    for index in range(len(true_sources)):
        true_support = set(np.flatnonzero(true_activity[index]))
        estimated_support = set(np.flatnonzero(estimated_activity[index]))
        support_matches.append(true_support == estimated_support)
    valid_correlations = [value for value in correlations if value is not None]
    valid_scaled_errors = [value for value in scaled_errors if value is not None]
    return {
        "support_accuracy": float(np.mean(support_matches)),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(2 * precision * recall / (precision + recall + 1e-12)),
        "zero_threshold": float(threshold),
        "source_correlations": correlations,
        "mean_active_correlation": float(np.mean(valid_correlations)) if valid_correlations else None,
        "scaled_errors": scaled_errors,
        "mean_scaled_error": float(np.mean(valid_scaled_errors)) if valid_scaled_errors else None,
    }


def score_directions(true_matrix, estimated_matrix):
    matches = match_directions(true_matrix, estimated_matrix)
    by_source = [None] * true_matrix.shape[1]
    for match in matches:
        by_source[match["true_index"]] = match["cosine"]
    valid = [value for value in by_source if value is not None]
    penalized = [value if value is not None else 0.0 for value in by_source]
    return {
        "matches": matches,
        "cosines_by_source": by_source,
        "mean_matched_cosine": float(np.mean(valid)) if valid else 0.0,
        "mean_direction_cosine": float(np.mean(penalized)),
        "direction_recovery_0.99": float(np.mean([value >= 0.99 for value in penalized])),
        "direction_recovery_0.95": float(np.mean([value >= 0.95 for value in penalized])),
        "coverage_0.99": int(sum(value >= 0.99 for value in penalized)),
        "coverage_0.95": int(sum(value >= 0.95 for value in penalized)),
    }
