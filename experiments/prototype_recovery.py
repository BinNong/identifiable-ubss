import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import align_matrix_to_truth, recover_sources, score_source_recovery
from pnl import (
    calibrate_two_channel_pnl,
    estimate_columns_from_linearized,
    generate_two_regime_pnl,
)


def main():
    _, regimes = generate_two_regime_pnl(n_samples=2500, noise_std=0.002)
    mixing = np.array([[1.0, 0.8, 0.5], [0.4, -0.6, 1.0]])
    calibrated = [
        calibrate_two_channel_pnl(regime["observations"], 2, zone_length=regime["zone_length"])
        for regime in regimes
    ]
    linearized = np.concatenate([result["linearized"] for result in calibrated], axis=0)
    true_sources = np.concatenate([regime["sources"][result["mask"]] for regime, result in zip(regimes, calibrated)], axis=0)
    columns, _ = estimate_columns_from_linearized(linearized, 3)
    aligned, _ = align_matrix_to_truth(mixing, columns)
    estimates, _, _ = recover_sources(linearized, aligned, 1)
    output = score_source_recovery(estimates, true_sources)
    print(json.dumps(output, indent=2))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "recovery_prototype.json").open("w") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
