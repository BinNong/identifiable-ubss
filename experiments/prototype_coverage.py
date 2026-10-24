import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pnl import match_directions
from subspace import (
    estimate_window_bases,
    generate_running_example,
    recover_directions_from_bases,
)


def main():
    mixing, regimes = generate_running_example(noise_std=0.001)
    all_bases = []
    all_windows = []
    separate_results = []
    for regime in regimes:
        bases, windows, _ = estimate_window_bases(
            regime["observations"],
            regime["block_length"],
            max_rank=2,
        )
        directions, selected, _ = recover_directions_from_bases(bases, 5, windows)
        separate_results.append({
            "matches": match_directions(mixing, directions),
            "n_directions": int(directions.shape[1]),
            "cluster_sizes": [len(cluster["members"]) for cluster in selected],
        })
        offset = len(all_bases)
        all_bases.extend(bases)
        all_windows.extend([window + offset for window in windows])

    joint_directions, joint_selected, _ = recover_directions_from_bases(all_bases, 5, all_windows)
    joint_matches = match_directions(mixing, joint_directions)
    output = {
        "joint_matches": joint_matches,
        "joint_mean_cosine": float(np.mean([match["cosine"] for match in joint_matches])),
        "joint_cluster_sizes": [len(cluster["members"]) for cluster in joint_selected],
        "separate_results": separate_results,
    }
    print(json.dumps(output, indent=2))
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "coverage_prototype.json").open("w") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
