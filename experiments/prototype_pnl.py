import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pnl import (
    calibrate_two_channel_pnl,
    estimate_columns_from_linearized,
    generate_two_regime_pnl,
    match_directions,
)


def main():
    mixing, regimes = generate_two_regime_pnl(n_samples=2500, noise_std=0.002)
    calibrated = [calibrate_two_channel_pnl(regime["observations"], 2, zone_length=regime["zone_length"]) for regime in regimes]
    joint_values = np.concatenate([result["linearized"] for result in calibrated], axis=0)
    joint_columns, _ = estimate_columns_from_linearized(joint_values, 3)
    joint_matches = match_directions(mixing, joint_columns)

    separate_matches = []
    raw_columns, _ = estimate_columns_from_linearized(
        np.concatenate([regime["observations"] for regime in regimes], axis=0),
        3,
    )
    raw_matches = match_directions(mixing, raw_columns)
    for regime, result in zip(regimes, calibrated):
        columns, _ = estimate_columns_from_linearized(result["linearized"], 2)
        separate_matches.append(match_directions(mixing, columns))

    output = {
        "joint_direction_matches": joint_matches,
        "joint_mean_direction_cosine": float(np.mean([match["cosine"] for match in joint_matches])),
        "separate_direction_matches": separate_matches,
        "without_calibration_matches": raw_matches,
        "without_calibration_mean_direction_cosine": float(np.mean([match["cosine"] for match in raw_matches])),
        "inverse_coefficients": [result["inverse_coefficients"].tolist() for result in calibrated],
        "forward_coefficients": [regime["forward_coefficients"].tolist() for regime in regimes],
        "line_errors": [result["line_errors"].tolist() for result in calibrated],
    }
    print(json.dumps(output, indent=2))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "pnl_prototype.json").open("w") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
