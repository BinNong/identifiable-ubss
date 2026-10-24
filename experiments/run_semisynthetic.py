import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnl import forward_nonlinearity
from run_pnl_experiments import evaluate_regimes, summarize_runs


def load_sources(data_dir):
    sources = []
    for name in ["jackson", "nicolas", "theo"]:
        sample_rate, audio = wavfile.read(data_dir / f"speech_{name}.wav")
        if sample_rate != 8000:
            raise ValueError(f"Unexpected sample rate for {name}: {sample_rate}")
        audio = np.asarray(audio, dtype=float)
        audio -= np.mean(audio)
        audio /= np.std(audio) + 1e-12
        sources.append(audio)
    return sources


def draw_segment(audio, length, rng):
    best = None
    best_rms = -np.inf
    for _ in range(40):
        start = int(rng.integers(0, len(audio) - length + 1))
        segment = audio[start:start + length].copy()
        rms = np.sqrt(np.mean(segment**2))
        if rms > best_rms:
            best = segment
        if rms > 0.15:
            break
    best -= np.mean(best)
    best /= np.sqrt(np.mean(best**2)) + 1e-12
    return best


def generate_semisynthetic(source_audio, noise_std=0.002, seed=20261016):
    rng = np.random.default_rng(seed)
    mixing = np.array([[1.0, 0.8, 0.5], [0.4, -0.6, 1.0]])
    specifications = [
        {"active": (0, 1), "coefficients": np.array([0.8, -0.45])},
        {"active": (1, 2), "coefficients": np.array([-0.35, 0.9])},
    ]
    regimes = []
    n_blocks = 16
    block_length = 400
    for regime, specification in enumerate(specifications):
        active = specification["active"]
        labels = np.tile(np.arange(len(active)), n_blocks // 2)
        sources = np.zeros((n_blocks * block_length, 3))
        for block, local_index in enumerate(labels):
            source_index = active[local_index]
            segment = draw_segment(source_audio[source_index], block_length, rng)
            start = block * block_length
            sources[start:start + block_length, source_index] = segment
        labels = np.repeat(labels, block_length)
        linear = sources @ mixing.T
        observations = forward_nonlinearity(linear, specification["coefficients"])
        observations += noise_std * rng.normal(size=observations.shape)
        regimes.append({
            "observations": observations,
            "sources": sources,
            "linear": linear,
            "active": active,
            "labels": labels,
            "zone_length": 100,
            "forward_coefficients": specification["coefficients"],
            "regime": regime,
        })
    return mixing, regimes


def evaluate_configuration(configuration):
    project_dir = Path(__file__).resolve().parents[1]
    source_audio = load_sources(project_dir / "data")
    mixing, regimes = generate_semisynthetic(
        source_audio,
        noise_std=configuration["noise_std"],
        seed=configuration["seed"],
    )
    output = evaluate_regimes(mixing, regimes)
    return [{
        "method": method,
        "noise_std": configuration["noise_std"],
        "nonlinearity_strength": 1.0,
        "seed": configuration["seed"],
        **metrics,
    } for method, metrics in output.items() if method in [
        "joint_blind", "joint_oracle", "pooled_no_calibration"
    ]]


def run_suite(mode, workers):
    if mode == "smoke":
        noises = [0.002]
        seeds = [20261016]
    else:
        noises = [0.001, 0.002, 0.005, 0.01]
        seeds = list(range(106001, 106021))
    configurations = [{"noise_std": noise, "seed": seed} for noise in noises for seed in seeds]
    runs = []
    if workers == 1:
        for index, configuration in enumerate(configurations, start=1):
            runs.extend(evaluate_configuration(configuration))
            print(f"Semi-synthetic {index}/{len(configurations)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_configuration, configuration) for configuration in configurations]
            for index, future in enumerate(as_completed(futures), start=1):
                runs.extend(future.result())
                print(f"Semi-synthetic {index}/{len(configurations)} completed", flush=True)
    runs.sort(key=lambda run: (run["noise_std"], run["seed"], run["method"]))
    summary = summarize_runs(runs)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "" if mode == "full" else "_smoke"
    with (results_dir / f"semisynthetic{suffix}_runs.json").open("w") as handle:
        json.dump(runs, handle, indent=2)
    with (results_dir / f"semisynthetic{suffix}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (results_dir / f"semisynthetic{suffix}_config.json").open("w") as handle:
        json.dump({
            "mode": mode,
            "workers": workers,
            "configurations": configurations,
            "evidence_type": "Measured speech sources with synthetic activation blocks, shared mixing and regime-specific PNL distortions",
            "source_dataset": "Free Spoken Digit Dataset recordings 0_jackson_0, 0_nicolas_0, and 0_theo_0",
            "speakers": ["Jackson", "Nicolas", "Theo"],
            "not_real_mixture_validation": True,
        }, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_suite(args.mode, args.workers)


if __name__ == "__main__":
    main()
