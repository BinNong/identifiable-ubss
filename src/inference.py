from scipy.stats import beta


def clopper_pearson(successes, total, confidence=0.95):
    if total < 1:
        raise ValueError("A binomial interval requires at least one trial")
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial success count")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return {
        "n_trials": int(total),
        "n_successes": int(successes),
        "rate": float(successes / total),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "interval_method": "Clopper-Pearson",
        "confidence": confidence,
    }
