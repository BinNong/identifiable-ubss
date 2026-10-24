import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning


DICTIONARY_LEARNING_CONFIG = {
    "alpha": 1.0,
    "max_iter": 200,
    "fit_algorithm": "lars",
    "batch_size": 256,
    "shuffle": True,
    "transform_algorithm": "omp",
    "tol": 1e-4,
    "max_no_improvement": 20,
    "transform_max_iter": 200,
}


def _reconstruction_error(model, values, max_rows=2048, seed=20261016):
    if len(values) > max_rows:
        rng = np.random.default_rng(seed)
        rows = values[rng.choice(len(values), size=max_rows, replace=False)]
    else:
        rows = values
    codes = model.transform(rows)
    residual = rows - codes @ model.components_
    return float(np.mean(residual**2))


def dictionary_learning_directions(
    values,
    n_sources,
    sparsity,
    seed=20261016,
    config=None,
    n_init=1,
    dict_init=None,
):
    values = np.asarray(values, dtype=float)
    settings = dict(DICTIONARY_LEARNING_CONFIG if config is None else config)
    starts = [("warm", dict_init)] if dict_init is not None else []
    starts += [("random", None)] * max(1, n_init)
    fit_records = []
    best = None
    for start_index, (label, initial_dictionary) in enumerate(starts):
        keyword = {}
        if initial_dictionary is not None:
            keyword["dict_init"] = np.asarray(initial_dictionary, dtype=float).T
        model = MiniBatchDictionaryLearning(
            n_components=n_sources,
            transform_n_nonzero_coefs=sparsity,
            n_jobs=1,
            random_state=seed + start_index,
            **settings,
            **keyword,
        )
        model.fit(values)
        error = _reconstruction_error(model, values, seed=seed)
        fit_records.append({
            "start": label,
            "start_index": int(start_index),
            "reconstruction_error": error,
            "n_iter": int(getattr(model, "n_iter_", 0)),
        })
        if best is None or error < best[0]:
            best = (error, model, start_index)
    error, model, best_index = best
    directions = model.components_.T
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    return directions, {
        "n_iter": int(getattr(model, "n_iter_", 0)),
        "fit_records": fit_records,
        "selected_start": int(best_index),
        "selected_start_label": starts[best_index][0],
        "reconstruction_error": float(error),
    }
