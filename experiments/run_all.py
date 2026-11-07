import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.smoke and args.full:
        parser.error("Choose at most one of --smoke and --full")
    if args.workers < 1:
        parser.error("--workers must be positive")
    mode = "full" if args.full else "smoke"

    project = Path(__file__).resolve().parents[1]
    (project / "results").mkdir(exist_ok=True)
    (project / "paper").mkdir(exist_ok=True)
    scripts = [
        ("verify_theory.py", []),
        ("run_selection_ablation.py", ["--mode", "development" if mode == "full" else "smoke", "--workers", str(args.workers)]),
        ("run_coverage_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_random_support_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_higher_order_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_higher_order_pnl_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_scaling_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_lattice_complexity.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_pnl_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_semisynthetic.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_conditioning_experiments.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("run_stability_budget.py", ["--mode", mode, "--workers", str(args.workers)]),
        ("compute_recovery_intervals.py", ["--mode", mode]),
        ("make_figures.py", ["--mode", mode]),
        ("validate_results.py", ["--mode", mode]),
    ]
    for script, arguments in scripts:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(project / "experiments" / script), *arguments], cwd=project, check=True)

    import matplotlib
    import numpy
    import scipy
    import sklearn
    try:
        import torch
    except ImportError:
        torch = None

    environment = {
        "mode": mode,
        "workers": args.workers,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__ if torch is not None else None,
        "cuda_available": torch.cuda.is_available() if torch is not None else False,
        "commands": [f"python experiments/run_all.py --{mode} --workers {args.workers}"],
    }
    for name in ["environment.json", f"environment_{mode}.json"]:
        with (project / "results" / name).open("w") as handle:
            json.dump(environment, handle, indent=2)
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
