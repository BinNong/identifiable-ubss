import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
MODE = "full"
CORE_NOISE = 0.005
HIGHER_NOISE = 0.02


def load(name):
    filename = name if MODE == "full" else name.replace("_summary", "_smoke_summary")
    if MODE != "full" and filename.endswith("_diagnostics"):
        filename = filename.replace("_diagnostics", "_smoke_diagnostics")
    if MODE != "full" and filename == "recovery_intervals":
        filename = "recovery_intervals_smoke"
    if MODE != "full" and filename == "stability_budget":
        filename = "stability_budget_smoke"
    with (RESULTS / f"{filename}.json").open() as handle:
        return json.load(handle)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def row(rows, method, noise=None, experiment=None, **constraints):
    matches = []
    for record in rows:
        if record["method"] != method:
            continue
        if noise is not None and record.get("noise_std") != noise:
            continue
        if experiment is not None and record.get("experiment") != experiment:
            continue
        if any(record.get(key) != value for key, value in constraints.items()):
            continue
        matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"Expected one row, found {len(matches)}: {method}, {noise}, {experiment}, {constraints}")
    return matches[0]


def series(rows, method, experiment=None, **constraints):
    selected = []
    for record in rows:
        if record["method"] != method:
            continue
        if experiment is not None and record.get("experiment") != experiment:
            continue
        if any(record.get(key) != value for key, value in constraints.items()):
            continue
        selected.append(record)
    return sorted(selected, key=lambda record: record["noise_std"])


def metric(record, name, default=np.nan):
    return record.get(name, default)


def plot_coverage(summary):
    methods = [
        ("joint_blind_lattice", "Joint blind lattice"),
        ("joint_blind_lattice_score", "Score-selection ablation"),
        ("joint_oracle_lattice", "Oracle calibration"),
        ("joint_blind_rank_one", "PNL-SCA-SSZ"),
        ("joint_blind_dictionary_learning", "Online dictionary learning"),
        ("joint_blind_dictionary_learning_warm", "DL warm start"),
        ("joint_no_calibration_lattice", "No calibration"),
        ("missing_cross_blind_lattice", "Missing cross-support"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method, label in methods:
        records = series(summary, method, experiment="main")
        x = [record["noise_std"] for record in records]
        for axis, metric_name in zip(axes, ["mean_direction_cosine", "source_5_recovery_0.99", "support_f1"]):
            if metric_name == "source_5_recovery_0.99":
                y = [record[metric_name] for record in records]
                yerr = None
            else:
                y = [metric(record, f"{metric_name}_mean") for record in records]
                yerr = [metric(record, f"{metric_name}_ci95", 0.0) for record in records]
            axis.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)
    axes[0].set(ylabel="Penalized direction cosine", ylim=(0, 1.03))
    axes[1].set(ylabel="Source 5 recovery at 0.99", ylim=(-0.03, 1.03))
    axes[2].set(ylabel="Support F1", ylim=(0, 1.03))
    for axis in axes:
        axis.set_xlabel("Noise standard deviation")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "coverage_noise_robustness")

    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    selected_methods = methods
    x = np.arange(5)
    width = 0.14
    for index, (method, label) in enumerate(selected_methods):
        record = row(summary, method, CORE_NOISE, experiment="main")
        values = [record[f"source_{source}_recovery_0.99"] for source in range(1, 6)]
        offset = (index - (len(selected_methods) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=label)
    axis.set(xticks=x, xticklabels=[f"Source {i}" for i in range(1, 6)], ylabel="Recovery rate at 0.99", ylim=(0, 1.08))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    save(fig, "coverage_by_source")


def plot_random_support(summary, intervals):
    methods = [
        ("joint_blind_lattice", "Joint blind lattice"),
        ("joint_blind_lattice_score", "Score-selection ablation"),
        ("joint_blind_rank_one", "PNL-SCA-SSZ"),
        ("joint_blind_dictionary_learning", "Online dictionary learning"),
        ("joint_blind_dictionary_learning_warm", "DL warm start"),
        ("joint_oracle_lattice", "Oracle calibration"),
        ("joint_no_calibration_lattice", "No calibration"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method, label in methods:
        records = series(summary, method, experiment="random_support")
        x = [record["noise_std"] for record in records]
        for axis, metric_name in zip(axes, ["mean_direction_cosine", "source_5_recovery", "support_f1"]):
            if metric_name == "source_5_recovery":
                interval_records = [
                    row(intervals, method, record["noise_std"], dataset="random_support", metric="source_5_recovery")
                    for record in records
                ]
                y = [record["rate"] for record in interval_records]
                lower = [record["rate"] - record["ci95_lower"] for record in interval_records]
                upper = [record["ci95_upper"] - record["rate"] for record in interval_records]
                axis.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=3, label=label)
            else:
                axis.errorbar(
                    x,
                    [record[f"{metric_name}_mean"] for record in records],
                    yerr=[record[f"{metric_name}_ci95"] for record in records],
                    marker="o",
                    capsize=3,
                    label=label,
                )
    axes[0].set(ylabel="Penalized direction cosine", ylim=(0, 1.03))
    axes[1].set(ylabel="Source 5 recovery [95% CP]", ylim=(-0.03, 1.03))
    axes[2].set(ylabel="Support F1", ylim=(0, 1.03))
    for axis in axes:
        axis.set_xlabel("Noise standard deviation")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "random_support_recovery")


def plot_sample_and_window(summary):
    if MODE != "full":
        return
    blind = "joint_blind_lattice"
    sample_records = [row(summary, blind, 0.005, experiment="main")]
    sample_records += [row(summary, blind, 0.005, experiment="sample_complexity", blocks_per_support=blocks) for blocks in [2, 4, 16]]
    sample_records = sorted(sample_records, key=lambda record: record["blocks_per_support"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = [record["blocks_per_support"] for record in sample_records]
    for axis, metric_name, ylabel in [
        (axes[0], "mean_direction_cosine", "Penalized direction cosine"),
        (axes[1], "support_f1", "Support F1"),
    ]:
        axis.errorbar(x, [record[f"{metric_name}_mean"] for record in sample_records], marker="o")
        axis.set(xlabel="Blocks per support", ylabel=ylabel, ylim=(0, 1.03), xticks=x)
        axis.grid(alpha=0.25)
    save(fig, "coverage_sample_complexity")

    window_records = [row(summary, blind, 0.005, experiment="main")]
    window_records += [row(summary, blind, 0.005, experiment="window_length", block_length=length) for length in [40, 160]]
    window_records = sorted(window_records, key=lambda record: record["block_length"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = [record["block_length"] for record in window_records]
    for axis, metric_name, ylabel in [
        (axes[0], "mean_direction_cosine", "Penalized direction cosine"),
        (axes[1], "support_f1", "Support F1"),
    ]:
        axis.errorbar(x, [record[f"{metric_name}_mean"] for record in window_records], marker="o")
        axis.set(xlabel="Window length", ylabel=ylabel, ylim=(0, 1.03), xticks=x)
        axis.grid(alpha=0.25)
    save(fig, "coverage_window_length")


def plot_higher_order(summary):
    methods = [("lattice", "Depth-3 lattice"), ("pairwise_lattice", "Pairwise only"), ("rank_one", "Rank-one only")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, label in methods:
        records = series(summary, method)
        x = [record["noise_std"] for record in records]
        axes[0].errorbar(x, [record["mean_direction_cosine_mean"] for record in records], marker="o", label=label)
        axes[1].errorbar(x, [record["source_4_recovery_0.99"] for record in records], marker="o", label=label)
    axes[0].set(xlabel="Noise standard deviation", ylabel="Penalized direction cosine", ylim=(0, 1.03))
    axes[1].set(xlabel="Noise standard deviation", ylabel="Source 4 recovery at 0.99", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "higher_order_recovery")


def plot_pnl(summary):
    methods = [("joint_blind", "Joint blind"), ("joint_oracle", "Oracle"), ("pooled_no_calibration", "No calibration")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, label in methods:
        records = series(summary, method, nonlinearity_strength=1.0)
        x = [record["noise_std"] for record in records]
        for axis, metric_name, ylabel in [
            (axes[0], "mean_direction_cosine", "Penalized direction cosine"),
            (axes[1], "support_f1", "Support F1"),
        ]:
            axis.errorbar(
                x,
                [record[f"{metric_name}_mean"] for record in records],
                yerr=[record[f"{metric_name}_ci95"] for record in records],
                marker="o",
                capsize=3,
                label=label,
            )
            axis.set(xlabel="Noise standard deviation", ylabel=ylabel, ylim=(0, 1.03))
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "pnl_noise_robustness")

    if not any(record["noise_std"] == 0.03 for record in summary):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    strengths = sorted({record["nonlinearity_strength"] for record in summary})
    for method, label in methods:
        records = [row(summary, method, 0.03, nonlinearity_strength=strength) for strength in strengths]
        for axis, metric_name, ylabel in [
            (axes[0], "mean_direction_cosine", "Penalized direction cosine"),
            (axes[1], "support_f1", "Support F1"),
        ]:
            axis.errorbar(strengths, [record[f"{metric_name}_mean"] for record in records], marker="o", label=label)
            axis.set(xlabel="Nonlinearity strength", ylabel=ylabel, ylim=(0, 1.03), xticks=strengths)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "pnl_strength_robustness")


def plot_semisynthetic(summary):
    methods = [("joint_blind", "Joint blind"), ("joint_oracle", "Oracle"), ("pooled_no_calibration", "No calibration")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, label in methods:
        records = series(summary, method)
        x = [record["noise_std"] for record in records]
        for axis, metric_name, ylabel in [
            (axes[0], "mean_direction_cosine", "Penalized direction cosine"),
            (axes[1], "support_f1", "Support F1"),
        ]:
            axis.errorbar(
                x,
                [record[f"{metric_name}_mean"] for record in records],
                yerr=[record[f"{metric_name}_ci95"] for record in records],
                marker="o",
                capsize=3,
                label=label,
            )
            axis.set(xlabel="Noise standard deviation", ylabel=ylabel, ylim=(0, 1.03))
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "semisynthetic_recovery")


def plot_scaling(summary):
    source_counts = sorted({record["n_sources"] for record in summary if record["method"] == "lattice"})
    block_counts = sorted({record["blocks_per_support"] for record in summary if record["method"] == "lattice"})
    max_blocks = max(block_counts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for blocks in block_counts:
        records = [row(summary, "lattice", None, n_sources=n, blocks_per_support=blocks) for n in source_counts]
        axes[0].errorbar(source_counts, [record["runtime_seconds_mean"] for record in records], marker="o", label=f"{blocks} blocks/support")
    axes[0].set(xlabel="Number of sources", ylabel="Direction runtime (s)", xticks=source_counts)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for method, label in [("lattice", "OMP"), ("lattice_exact_recovery", "Exhaustive")]:
        method_sources = sorted({record["n_sources"] for record in summary if record["method"] == method})
        records = [row(summary, method, None, n_sources=n, blocks_per_support=max_blocks) for n in method_sources]
        axes[1].errorbar(method_sources, [record["support_f1_mean"] for record in records], marker="o", label=label)
    axes[1].set(xlabel="Number of sources", ylabel="Support F1", ylim=(0, 1.03), xticks=source_counts)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save(fig, "scaling_runtime")


def plot_conditioning(summary):
    thresholds = sorted({record["min_calibration_entry"] for record in summary})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for noise in sorted({record["noise_std"] for record in summary}):
        blind = [row(summary, "joint_blind_lattice", noise, min_calibration_entry=threshold) for threshold in thresholds]
        oracle = [row(summary, "joint_oracle_lattice", noise, min_calibration_entry=threshold) for threshold in thresholds]
        axes[0].errorbar(
            thresholds,
            [record["mean_direction_cosine_mean"] for record in blind],
            yerr=[record["mean_direction_cosine_ci95"] for record in blind],
            marker="o",
            capsize=3,
            label=f"Blind, noise {noise}",
        )
        axes[0].errorbar(
            thresholds,
            [record["mean_direction_cosine_mean"] for record in oracle],
            yerr=[record["mean_direction_cosine_ci95"] for record in oracle],
            marker="x",
            linestyle="--",
            capsize=3,
            label=f"Oracle, noise {noise}",
        )
        axes[1].errorbar(thresholds, [record["calibration_relative_error_mean"] for record in blind], marker="o", label=f"Noise {noise}")
    axes[0].set(xlabel="Accepted minimum accessible entry", ylabel="Penalized direction cosine", ylim=(0, 1.03), xticks=thresholds)
    axes[1].set(xlabel="Accepted minimum accessible entry", ylabel="Relative linearization error", xticks=thresholds)
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    save(fig, "conditioning_robustness")


def plot_lattice_complexity(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for sparsity in sorted({record["sparsity"] for record in summary}):
        records = sorted(
            [record for record in summary if record["sparsity"] == sparsity],
            key=lambda record: record["n_multi_prototypes_mean"],
        )
        x = [record["n_multi_prototypes_mean"] for record in records]
        axes[0].errorbar(
            x,
            [record["runtime_seconds_mean"] for record in records],
            yerr=[record["runtime_seconds_ci95"] for record in records],
            marker="o",
            capsize=3,
            label=f"k={sparsity}",
        )
        axes[1].errorbar(
            x,
            [record["direction_recovery_0.99_mean"] for record in records],
            yerr=[record["direction_recovery_0.99_ci95"] for record in records],
            marker="o",
            capsize=3,
            label=f"k={sparsity}",
        )
    axes[0].set(xlabel="Multi-source prototypes", ylabel="Lattice runtime (s)")
    axes[1].set(xlabel="Multi-source prototypes", ylabel="Direction recovery at 0.99", ylim=(0, 1.03))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, "lattice_complexity")


def plot_support_schematic():
    fig, axis = plt.subplots(figsize=(9, 3.2))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.text(0.05, 0.75, r"Regime 1: $\{1\},\{2\},\{3\},\{4\},\{1,5\}$", fontsize=14)
    axis.text(0.05, 0.50, r"Regime 2: $\{1\},\{2\},\{3\},\{4\},\{2,5\}$", fontsize=14)
    axis.text(0.05, 0.20, "Separately: the fifth column is not revealed", fontsize=11, color="darkred")
    axis.text(0.60, 0.62, r"$\{1,5\}\cap\{2,5\}=\{5\}$", fontsize=17)
    axis.text(0.60, 0.32, "Jointly: all five columns are revealed", fontsize=11, color="darkgreen")
    axis.annotate("", xy=(0.56, 0.50), xytext=(0.45, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.5})
    save(fig, "support_coverage_schematic")


def tex_number(value, digits=4):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def with_ci(record, metric):
    if record.get(f"{metric}_mean") is None:
        return "--"
    return f"{tex_number(record[f'{metric}_mean'])}\\,({tex_number(record.get(f'{metric}_ci95', 0.0))})"


def percent(value):
    return f"{100 * value:.1f}\\%"


def with_cp(record):
    return (
        f"{100 * record['rate']:.1f}\\%\\,["
        f"{100 * record['ci95_lower']:.1f}, {100 * record['ci95_upper']:.1f}]"
    )


def sci(value):
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def write_generated_tables(coverage, higher, pnl, semi, intervals):
    lines = ["% Auto-generated from saved experiment results."]
    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\scriptsize",
        f"\\caption{{Five-source results at noise {CORE_NOISE} (100 held-out confirmation seeds). Parentheses are 95\\% continuous-metric confidence half-widths; brackets are exact Clopper--Pearson intervals for binary recovery. Runtime is post-inversion direction-and-recovery time; regime-only rows report direction time.}}",
        "\\label{tab:coverage_results}",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Dir. (CI) & Fifth rec. [CP95] & F1 (CI) & Runtime (s) \\\\",
        "\\midrule",
    ])
    methods = [
        ("joint_blind_lattice", "Joint blind lattice"),
        ("joint_blind_lattice_score", "\\quad score-only selection (ablation)"),
        ("joint_blind_rank_one", "PNL-SCA-SSZ"),
        ("joint_blind_dictionary_learning", "Online dictionary learning"),
        ("joint_blind_dictionary_learning_warm", "\\quad warm start from lattice"),
        ("joint_oracle_lattice", "Oracle calibration"),
        ("joint_no_calibration_lattice", "No calibration"),
        ("separate_regime_0_blind_lattice", "Regime 1 only"),
        ("separate_regime_1_blind_lattice", "Regime 2 only"),
        ("missing_cross_blind_lattice", "Missing cross-support"),
    ]
    for method, label in methods:
        record = row(coverage, method, CORE_NOISE, experiment="main")
        recovery = row(intervals, method, CORE_NOISE, experiment="main", dataset="coverage", metric="source_5_recovery")
        lines.append(
            f"{label} & {with_ci(record, 'mean_direction_cosine')} & {with_cp(recovery)} & "
            f"{with_ci(record, 'support_f1')} & {record['runtime_seconds_mean']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    with (PROJECT / "paper" / "generated_tables.tex").open("w") as handle:
        handle.write("\n".join(lines))


def write_supplement_tables(coverage, higher, pnl, semi, scaling, conditioning, complexity, conditioning_details, random_support, intervals, selection_development, higher_pnl=None, stability_budget=None, pnl_details=None, random_details=None):
    lines = ["% Auto-generated supplementary tables."]
    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{Main five-source noise sweep (100 confirmation seeds per cell), including the highest-noise regime where the tuned dictionary-learning baseline overtakes the lattice estimator on aggregate direction cosine. Parentheses are 95\\% confidence-interval half-widths.}",
        "\\label{tab:supp_coverage_noise}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "Noise & Blind dir. (CI) & Score-sel. dir. & DL dir. (CI) & Oracle dir. (CI) & Blind cov. & Oracle cov. \\\\",
        "\\midrule",
    ])
    for noise in sorted({record["noise_std"] for record in coverage if record.get("experiment") == "main"}):
        blind = row(coverage, "joint_blind_lattice", noise, experiment="main")
        score_only = row(coverage, "joint_blind_lattice_score", noise, experiment="main")
        dictionary = row(coverage, "joint_blind_dictionary_learning", noise, experiment="main")
        oracle = row(coverage, "joint_oracle_lattice", noise, experiment="main")
        lines.append(
            f"{noise} & {with_ci(blind, 'mean_direction_cosine')} & {tex_number(score_only['mean_direction_cosine_mean'])} & "
            f"{with_ci(dictionary, 'mean_direction_cosine')} & {with_ci(oracle, 'mean_direction_cosine')} & "
            f"{blind['coverage_0.99_mean']:.2f} & {oracle['coverage_0.99_mean']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    for label, caption, records, key in ([
        ("sample", "Sample-complexity sweep at noise 0.005.", None, "blocks_per_support"),
        ("window", "Window-length sweep at noise 0.005.", None, "block_length"),
    ] if MODE == "full" else []):
        if label == "sample":
            records = [row(coverage, "joint_blind_lattice", 0.005, experiment="sample_complexity", blocks_per_support=blocks) for blocks in [2, 4, 16]]
            records.append(row(coverage, "joint_blind_lattice", 0.005, experiment="main"))
        else:
            records = [row(coverage, "joint_blind_lattice", 0.005, experiment="window_length", block_length=length) for length in [40, 160]]
            records.append(row(coverage, "joint_blind_lattice", 0.005, experiment="main"))
        records = sorted(records, key=lambda record: record[key])
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering\\small",
            f"\\caption{{{caption} Parentheses are 95\\% confidence-interval half-widths.}}",
            f"\\label{{tab:supp_{label}}}",
            "\\begin{tabular}{ccccc}",
            "\\toprule",
            f"{key.replace('_', ' ').title()} & Dir. (CI) & Cov.$_{{0.99}}$ & F1 (CI) & Coef. RMSE \\\\",
            "\\midrule",
        ])
        for record in records:
            lines.append(
                f"{record[key]} & {with_ci(record, 'mean_direction_cosine')} & {record['coverage_0.99_mean']:.2f} & "
                f"{with_ci(record, 'support_f1')} & {tex_number(record['calibration_coefficient_rmse_mean'])} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{Linear scaling experiment with 16 blocks per support. Runtime parentheses are 95\\% confidence-interval half-widths.}",
        "\\label{tab:supp_scaling}",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$n$ & Dir. recovery & Runtime (s) & OMP F1 & Exact F1 & Windows \\\\",
        "\\midrule",
    ])
    scaling_source_counts = sorted({record["n_sources"] for record in scaling if record["method"] == "lattice"})
    scaling_max_blocks = max(record["blocks_per_support"] for record in scaling if record["method"] == "lattice")
    for n in scaling_source_counts:
        record = row(scaling, "lattice", None, n_sources=n, blocks_per_support=scaling_max_blocks)
        exact_records = [item for item in scaling if item["method"] == "lattice_exact_recovery" and item["n_sources"] == n]
        exact = "--" if not exact_records else tex_number(row(scaling, "lattice_exact_recovery", None, n_sources=n, blocks_per_support=scaling_max_blocks)["support_f1_mean"])
        lines.append(
            f"{n} & {percent(record['direction_recovery_0.99_mean'])} & {with_ci(record, 'runtime_seconds')} & "
            f"{tex_number(record['support_f1_mean'])} & {exact} & {record['n_windows_mean']:.0f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        "\\caption{Weak-calibration ensembles. Each threshold defines a different matrix ensemble. Parentheses are continuous-metric confidence half-widths; brackets are exact Clopper--Pearson intervals for complete five-direction recovery.}",
        "\\label{tab:supp_conditioning}",
        "\\setlength{\\tabcolsep}{1.5pt}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "Threshold & Noise & Blind dir. (CI) & Blind complete [CP95] & Rel. lin. error & Oracle dir. & Oracle complete [CP95] \\\\",
        "\\midrule",
    ])
    for threshold in sorted({record["min_calibration_entry"] for record in conditioning}):
        for noise in sorted({record["noise_std"] for record in conditioning}):
            blind = row(conditioning, "joint_blind_lattice", noise, min_calibration_entry=threshold)
            oracle = row(conditioning, "joint_oracle_lattice", noise, min_calibration_entry=threshold)
            blind_recovery = row(intervals, "joint_blind_lattice", noise, dataset="conditioning", metric="complete_5_recovery", min_calibration_entry=threshold)
            oracle_recovery = row(intervals, "joint_oracle_lattice", noise, dataset="conditioning", metric="complete_5_recovery", min_calibration_entry=threshold)
            lines.append(
                f"{threshold} & {noise} & {with_ci(blind, 'mean_direction_cosine')} & {with_cp(blind_recovery)} & "
                f"{tex_number(blind['calibration_relative_error_mean'])} & {tex_number(oracle['mean_direction_cosine_mean'])} & "
                f"{with_cp(oracle_recovery)} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    pnl_failure_noise = max(record["noise_std"] for record in pnl)
    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        f"\\caption{{Two-channel failure regime at noise {pnl_failure_noise}. Parentheses are 95\\% confidence-interval half-widths.}}",
        "\\label{tab:supp_pnl_strength}",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "Strength & Method & Dir. (CI) & F1 (CI) & Coef. RMSE \\\\",
        "\\midrule",
    ])
    for strength in sorted({record["nonlinearity_strength"] for record in pnl}):
        for method, label in [("joint_blind", "Blind"), ("joint_blind_oracle_labels", "Blind, oracle labels"), ("joint_oracle", "Oracle"), ("pooled_no_calibration", "No calibration")]:
            record = row(pnl, method, pnl_failure_noise, nonlinearity_strength=strength)
            lines.append(
                f"{strength} & {label} & {with_ci(record, 'mean_direction_cosine')} & "
                f"{with_ci(record, 'support_f1')} & {tex_number(record.get('coefficient_rmse_mean'))} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{Noiseless lattice complexity over twenty held-out matrices. Runtime parentheses are continuous confidence half-widths; brackets are exact Clopper--Pearson intervals for complete recovery.}",
        "\\label{tab:supp_lattice_complexity}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "$k$ & $n$ & $Q_{\\mathrm{multi}}$ & Combinations & Coverage & Runtime (s) & Dir. rec. [CP95] \\\\",
        "\\midrule",
    ])
    for record in complexity:
        recovery = row(
            intervals,
            "lattice",
            None,
            dataset="lattice_complexity",
            metric="complete_direction_recovery",
            sparsity=record["sparsity"],
            n_sources=record["n_sources"],
            n_extra_supports=record["n_extra_supports"],
        )
        lines.append(
            f"{record['sparsity']} & {record['n_sources']} & {record['n_multi_prototypes_mean']:.0f} & "
            f"{record['enumerated_combinations_mean']:.0f} & {record['coverage_0.99_mean']:.2f} & "
            f"{with_ci(record, 'runtime_seconds')} & {with_cp(recovery)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{Complete direction recovery within quartiles of the cross-subspace projector distance, paired on the same matrices.}",
        "\\label{tab:supp_conditioning_bins}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "Noise & Quartile & Distance range & Median $\\gamma$ & Cases & Blind success & Oracle success \\\\",
        "\\midrule",
    ])
    for record in conditioning_details["distance_bins"]:
        lines.append(
            f"{record['noise_std']} & {record['bin']} & [{record['distance_min']:.3f}, {record['distance_max']:.3f}] & "
            f"{record['median_intersection_gamma']:.4f} & {record['n_cases']} & "
            f"{percent(record['blind_complete_success_rate'])} & {percent(record['oracle_complete_success_rate'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{Operational paired failure attribution. Shared direction denotes joint blind--oracle failure only. The three middle categories decompose blind-only failures; oracle-only failures are reported separately. All labels are associations, not unique causal identifications.}",
        "\\label{tab:supp_failure_attribution}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "Noise & Blind fail. & Shared & Calibration & Subspace & Intersection & Oracle-only & Oracle fail. \\\\",
        "\\midrule",
    ])
    for record in conditioning_details["failure_attribution"]:
        lines.append(
            f"{record['noise_std']} & {record['blind_failures']} & {record['shared_direction_stage_failures']} & "
            f"{record['calibration_associated_failures']} & {record['subspace_clustering_failures']} & "
            f"{record['intersection_selection_failures']} & {record['oracle_only_failures']} & {record['oracle_failures']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        "\\caption{Two-channel PNL results at unit strength and noise 0.002. Parentheses are 95\\% confidence-interval half-widths.}",
        "\\label{tab:supp_pnl_results}",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Dir. (CI) & Cov.$_{0.99}$ & F1 (CI) & Mean corr. & Coef. RMSE \\\\",
        "\\midrule",
    ])
    for method, label in [
        ("joint_blind", "Joint blind"),
        ("pooled_no_calibration", "No calibration"),
        ("joint_oracle", "Oracle"),
        ("separate_regime_0", "Regime 1 only"),
        ("separate_regime_1", "Regime 2 only"),
    ]:
        record = row(pnl, method, 0.002, nonlinearity_strength=1.0)
        lines.append(
            f"{label} & {with_ci(record, 'mean_direction_cosine')} & {record['coverage_0.99_mean']:.2f} & "
            f"{with_ci(record, 'support_f1')} & {tex_number(record.get('mean_active_correlation_mean'))} & "
            f"{tex_number(record.get('coefficient_rmse_mean'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{Measured-speech semi-synthetic results at noise 0.002. Parentheses are 95\\% confidence-interval half-widths.}",
        "\\label{tab:supp_semisynthetic_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Dir. (CI) & F1 (CI) & Mean corr. & Coef. RMSE \\\\",
        "\\midrule",
    ])
    for method, label in [("joint_blind", "Joint blind"), ("pooled_no_calibration", "No calibration"), ("joint_oracle", "Oracle")]:
        record = row(semi, method, 0.002)
        lines.append(
            f"{label} & {with_ci(record, 'mean_direction_cosine')} & {with_ci(record, 'support_f1')} & "
            f"{tex_number(record['mean_active_correlation_mean'])} & {tex_number(record.get('coefficient_rmse_mean'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        "\\caption{Random separating support families at noise 0.005 (100 independent family and data seeds). Parentheses are continuous confidence half-widths; brackets are exact Clopper--Pearson intervals.}",
        "\\label{tab:supp_random_support}",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Dir. (CI) & Cov.$_{0.99}$ & Fifth rec. [CP95] & F1 (CI) & Runtime (s) \\\\",
        "\\midrule",
    ])
    random_methods = [
        ("joint_blind_lattice", "Joint blind lattice"),
        ("joint_blind_rank_one", "PNL-SCA-SSZ"),
        ("joint_blind_dictionary_learning", "Online dictionary learning"),
        ("joint_oracle_lattice", "Oracle calibration"),
        ("joint_no_calibration_lattice", "No calibration"),
    ]
    for method, label in random_methods:
        record = row(random_support, method, 0.005, experiment="random_support")
        recovery = row(intervals, method, 0.005, dataset="random_support", metric="source_5_recovery")
        lines.append(
            f"{label} & {with_ci(record, 'mean_direction_cosine')} & {record['coverage_0.99_mean']:.2f} & "
            f"{with_cp(recovery)} & {with_ci(record, 'support_f1')} & {record['runtime_seconds_mean']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        f"\\caption{{Higher-order linear geometry at noise {HIGHER_NOISE} (20 held-out seeds); the sweep now extends into the noise regime where the linear geometry degrades. Parentheses are continuous-metric confidence half-widths; brackets are exact Clopper--Pearson intervals.}}",
        "\\label{tab:supp_higher_order_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Dir. (CI) & Cov.$_{0.99}$ & Source 4 rec. [CP95] & Runtime (s) \\\\",
        "\\midrule",
    ])
    for method, label in [("lattice", "Depth-3 lattice"), ("pairwise_lattice", "Pairwise only"), ("rank_one", "Rank-one only")]:
        record = row(higher, method, HIGHER_NOISE)
        recovery = row(intervals, method, HIGHER_NOISE, dataset="higher_order", metric="source_4_recovery")
        lines.append(
            f"{label} & {with_ci(record, 'mean_direction_cosine')} & {record['coverage_0.99_mean']:.2f} & "
            f"{with_cp(recovery)} & {record['runtime_seconds_mean']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if higher_pnl:
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering\\footnotesize",
            "\\caption{Higher-order experiment with blind PNL calibration (20 held-out seeds): the depth-3 target source is recoverable only through the three-way intersection. Target recovery uses exact Clopper--Pearson intervals.}",
            "\\label{tab:supp_higher_order_pnl}",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Noise & Method & Dir. (CI) & Cov.$_{0.99}$ & Source 4 rec. & Coef. RMSE \\\\",
            "\\midrule",
        ])
        for noise in sorted({record["noise_std"] for record in higher_pnl}):
            for method, label in [("blind_lattice", "Blind depth-3 lattice"), ("blind_pairwise_lattice", "Blind pairwise only"), ("blind_rank_one", "Blind rank-one only"), ("oracle_lattice", "Oracle depth-3 lattice")]:
                record = row(higher_pnl, method, noise)
                coefficient_rmse = record.get("calibration_coefficient_rmse_mean")
                lines.append(
                    f"{noise} & {label} & {with_ci(record, 'mean_direction_cosine')} & {record['coverage_0.99_mean']:.2f} & "
                    f"{percent(record['target_recovery_0.99'])} & {tex_number(coefficient_rmse)} \\\\"
                )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    selection_noises = sorted({record["noise_std"] for record in selection_development})
    first_noise = selection_noises[0]
    n_per_cell = sum(
        record["noise_std"] == first_noise
        and record["method"] == "joint_blind_lattice"
        and record["direction_score_rule"] == "max"
        for record in selection_development
    )
    n_thresholds = len({record["min_calibration_entry"] for record in selection_development})
    n_seeds = len({record["seed"] for record in selection_development})
    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        f"\\caption{{Development-only direction-score ablation. Each cell aggregates {n_per_cell} configurations from {n_thresholds} calibration-visibility threshold settings and {n_seeds} previously inspected seeds, not the final confirmation seeds.}}",
        "\\label{tab:supp_selection_ablation}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "Noise & Score & Blind dir. & Blind cov. & Blind fifth & Oracle dir. & Oracle cov. & Oracle fifth \\\\",
        "\\midrule",
    ])
    for noise in selection_noises:
        for score_rule, label in [("max", "Max (frozen)"), ("mean", "Mean"), ("legacy", "Count bonus")]:
            records = [
                record for record in selection_development
                if record["noise_std"] == noise and record["direction_score_rule"] == score_rule
            ]
            values = []
            for method in ["joint_blind_lattice", "joint_oracle_lattice"]:
                selected = [record for record in records if record["method"] == method]
                if len(selected) != n_per_cell:
                    raise ValueError(f"Incomplete direction-score ablation group: {noise}, {score_rule}, {method}")
                fifth = np.asarray([
                    record["cosines_by_source"][4] if record["cosines_by_source"][4] is not None else 0.0
                    for record in selected
                ])
                values.extend([
                    f"{np.mean([record['mean_direction_cosine'] for record in selected]):.4f}",
                    f"{np.mean([record['coverage_0.99'] for record in selected]):.2f}",
                    percent(float(np.mean(fifth >= 0.99))),
                ])
            lines.append(f"{noise} & {label} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if random_details is not None and random_details.get("selection_rule_comparison"):
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering\\small",
            "\\caption{Selection-rule comparison on identical calibrated data (random support families, 100 seeds per cell). The family-consistency rule selects the direction set that explains the most observed window subspaces; the score-only rule keeps the $n$ highest-scoring clusters.}",
            "\\label{tab:supp_selection_rules}",
            "\\begin{tabular}{ccccc}",
            "\\toprule",
            "Noise & Family-consistency fifth rec. & Score-only fifth rec. & Rescued seeds & Lost seeds \\\\",
            "\\midrule",
        ])
        for record in random_details["selection_rule_comparison"]:
            lines.append(
                f"{record['noise_std']} & {record['family_consistency_success']}/{record['n_cases']} & "
                f"{record['score_rule_success']}/{record['n_cases']} & "
                f"{len(record['rescued_seeds'])} & {len(record['lost_seeds'])} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if random_details is not None and random_details.get("cluster_count_rule"):
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering\\small",
            "\\caption{Computable validity indicator: surviving direction-cluster count versus fifth-source recovery (joint blind lattice, random support families). Counts above $n$ indicate elevated selection risk rather than certain failure; the conditional certificate additionally requires the retention, zero-intersection and clustering margins of Proposition~\\ref{prop:perturbed_lattice}(v).}",
            "\\label{tab:supp_cluster_count_rule}",
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "Noise & Cluster count & Cases & Fifth-source recovery \\\\",
            "\\midrule",
        ])
        for record in random_details["cluster_count_rule"]:
            if record["method"] != "joint_blind_lattice":
                continue
            within = "$\\leq n$" if record["n_clusters_within_source_count"] else "$> n$"
            lines.append(
                f"{record['noise_std']} & {within} & {record['n_cases']} & {percent(record['source_5_success_rate'])} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if stability_budget is not None and stability_budget.get("summary"):
        budget = stability_budget["summary"]
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering\\small",
            "\\caption{Proposition~\\ref{prop:intersection_stability} evaluated with pipeline-estimated subspaces and true geometry: the deterministic bound is checked against the realized target-direction error for every retained lattice candidate of the running example.}",
            "\\label{tab:supp_stability_budget}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Noise & $n$ & Median realized (deg) & Median bound (deg) & Violations \\\\",
            "\\midrule",
        ])
        cells = {}
        for record in stability_budget["records"]:
            cells.setdefault(record.get("noise_std"), []).append(record)
        for noise, records in sorted(cells.items()):
            realized = np.median([item["realized_angle_degrees"] for item in records])
            bound = np.median([item["prop3_bound_degrees"] for item in records])
            violations = sum(item["bound_margin"] < 0 for item in records)
            lines.append(f"{noise} & {len(records)} & {realized:.3f} & {bound:.3f} & {violations} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    if pnl_details is not None and pnl_details.get("cells"):
        failure_cells = [
            cell for cell in pnl_details["cells"]
            if cell["noise_std"] == max(item["noise_std"] for item in pnl_details["cells"])
        ]
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering\\small",
            "\\caption{Two-channel calibration at the highest noise level, stratified by coefficient-bound contact. The oracle-labels variant calibrates with the true zone labels, separating clustering corruption from least-squares geometry degradation.}",
            "\\label{tab:supp_pnl_bound_hit}",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{cccccc}",
            "\\toprule",
            "Strength & Bound hit & Cases & Dir. cosine & Cluster ARI & Coef. RMSE \\\\",
            "\\midrule",
        ])
        for cell in failure_cells:
            for flag in [True, False]:
                key = f"bound_hit_{flag}"
                if key in cell:
                    subset = cell[key]
                    lines.append(
                        f"{cell['nonlinearity_strength']} & {'yes' if flag else 'no'} & {subset['n_cases']} & "
                        f"{tex_number(subset['mean_direction_cosine'])} & {tex_number(subset['mean_cluster_ari'])} & "
                        f"{tex_number(subset['mean_coefficient_rmse'])} \\\\"
                    )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    with (PROJECT / "paper" / "supplement_tables.tex").open("w") as handle:
        handle.write("\n".join(lines))


def write_macros(theory, coverage, higher, pnl, semi, scaling, random_support, random_details, intervals, higher_pnl=None, stability_budget=None, pnl_details=None):
    core = row(coverage, "joint_blind_lattice", CORE_NOISE, experiment="main")
    oracle = row(coverage, "joint_oracle_lattice", CORE_NOISE, experiment="main")
    high_lattice = row(higher, "lattice", HIGHER_NOISE)
    high_pair = row(higher, "pairwise_lattice", HIGHER_NOISE)
    max_sources = max(record["n_sources"] for record in scaling if record["method"] == "lattice")
    max_blocks = max(record["blocks_per_support"] for record in scaling if record["method"] == "lattice")
    scale = row(scaling, "lattice", None, n_sources=max_sources, blocks_per_support=max_blocks)
    pnl_failure_noise = max(record["noise_std"] for record in pnl)
    pnl_noise = row(pnl, "joint_blind", pnl_failure_noise, nonlinearity_strength=1.0)
    pnl_noise_raw = row(pnl, "pooled_no_calibration", pnl_failure_noise, nonlinearity_strength=1.0)
    max_strength = max(record["nonlinearity_strength"] for record in pnl)
    pnl_strength = row(pnl, "joint_blind", pnl_failure_noise, nonlinearity_strength=max_strength)
    pnl_strength_raw = row(pnl, "pooled_no_calibration", pnl_failure_noise, nonlinearity_strength=max_strength)
    pnl_zero_noise = min(record["noise_std"] for record in pnl)
    pnl_zero_blind = row(pnl, "joint_blind", pnl_zero_noise, nonlinearity_strength=max_strength)
    pnl_zero_raw = row(pnl, "pooled_no_calibration", pnl_zero_noise, nonlinearity_strength=max_strength)
    semi_blind = row(semi, "joint_blind", 0.002)
    semi_raw = row(semi, "pooled_no_calibration", 0.002)
    core_interval = row(intervals, "joint_blind_lattice", CORE_NOISE, experiment="main", dataset="coverage", metric="source_5_recovery")
    random_blind = row(random_support, "joint_blind_lattice", 0.005, experiment="random_support")
    random_dictionary = row(random_support, "joint_blind_dictionary_learning", 0.005, experiment="random_support")
    random_interval = row(intervals, "joint_blind_lattice", 0.005, dataset="random_support", metric="source_5_recovery")
    random_groups = [group for group in random_details["groups"] if group["noise_std"] == 0.005]
    random_success = next(group for group in random_groups if group["source_5_success"])
    random_failure = next(group for group in random_groups if not group["source_5_success"])

    def diagnostic_value(group, key, formatter):
        return formatter(group[key]) if key in group else "--"

    def diagnostic_range(group, prefix, scale=1.0, digits=1, suffix=""):
        if f"{prefix}_min" not in group:
            return "--"
        lower = scale * group[f"{prefix}_min"]
        upper = scale * group[f"{prefix}_max"]
        return f"{lower:.{digits}f}--{upper:.{digits}f}{suffix}"

    commands = {
        "TheoryProjectionError": sci(theory["running_example"]["ambiguity_projection_error"]),
        "TheorySparseAccuracy": tex_number(theory["running_example"]["sparse_support_accuracy"]),
        "TheoryStabilityCases": str(theory["stability_bound"]["n_cases"]),
        "CoreBlindDir": tex_number(core["mean_direction_cosine_mean"]),
        "CoreOracleDir": tex_number(oracle["mean_direction_cosine_mean"]),
        "CoreBlindFifthRecovery": percent(core["source_5_recovery_0.99"]),
        "CoreBlindFifthRecoveryCI": f"[{100 * core_interval['ci95_lower']:.1f}, {100 * core_interval['ci95_upper']:.1f}]",
        "CoreOracleFifthRecovery": percent(oracle["source_5_recovery_0.99"]),
        "RandomBlindDir": tex_number(random_blind["mean_direction_cosine_mean"]),
        "RandomBlindFifthRecovery": percent(random_interval["rate"]),
        "RandomBlindFifthRecoveryCI": f"[{100 * random_interval['ci95_lower']:.1f}, {100 * random_interval['ci95_upper']:.1f}]",
        "RandomDictionaryDir": tex_number(random_dictionary["mean_direction_cosine_mean"]),
        "RandomBlindRuntime": f"{random_blind['runtime_seconds_mean']:.3f}",
        "RandomDictionaryRuntime": f"{random_dictionary['runtime_seconds_mean']:.3f}",
        "RandomFailureCount": str(random_failure["n_cases"]),
        "RandomSuccessCount": str(random_success["n_cases"]),
        "RandomFailureCalibrationMedian": diagnostic_value(random_failure, "calibration_relative_error_median", percent),
        "RandomSuccessCalibrationMedian": diagnostic_value(random_success, "calibration_relative_error_median", percent),
        "RandomFailureCalibrationRange": diagnostic_range(random_failure, "calibration_relative_error", 100.0, 1, "\\%"),
        "RandomSuccessCalibrationRange": diagnostic_range(random_success, "calibration_relative_error", 100.0, 1, "\\%"),
        "RandomFailurePrototypesMedian": diagnostic_value(random_failure, "n_prototypes_median", lambda value: f"{value:.1f}"),
        "RandomSuccessPrototypesMedian": diagnostic_value(random_success, "n_prototypes_median", lambda value: f"{value:.1f}"),
        "RandomFailurePrototypesRange": diagnostic_range(random_failure, "n_prototypes", digits=0),
        "RandomSuccessPrototypesRange": diagnostic_range(random_success, "n_prototypes", digits=0),
        "RandomFailureGammaMedian": diagnostic_value(random_failure, "target_intersection_gamma_max_median", tex_number),
        "RandomSuccessGammaMedian": diagnostic_value(random_success, "target_intersection_gamma_max_median", tex_number),
        "RandomFailureOracleRecovery": diagnostic_value(random_failure, "oracle_source_5_recovery_rate", percent),
        "HigherLatticeCoverage": f"{high_lattice['coverage_0.99_mean']:.0f}/7",
        "HigherPairCoverage": f"{high_pair['coverage_0.99_mean']:.0f}/7",
        "ScalingDirectionRecovery": percent(scale["direction_recovery_0.99_mean"]),
        "ScalingRuntime": f"{scale['runtime_seconds_median']:.3f}",
        "PnlNoiseFailureDir": tex_number(pnl_noise["mean_direction_cosine_mean"]),
        "PnlNoiseNoCalibrationDir": tex_number(pnl_noise_raw["mean_direction_cosine_mean"]),
        "PnlStrengthFailureDir": tex_number(pnl_strength["mean_direction_cosine_mean"]),
        "PnlStrengthNoCalibrationDir": tex_number(pnl_strength_raw["mean_direction_cosine_mean"]),
        "PnlStrengthBoundHitRate": percent(pnl_strength["coefficient_bound_hit_rate_mean"]),
        "PnlStrengthObjectiveReduction": percent(pnl_strength["calibration_objective_relative_reduction_mean"]),
        "PnlStrengthZeroNoiseBlindDir": tex_number(pnl_zero_blind["mean_direction_cosine_mean"]),
        "PnlStrengthZeroNoiseNoCalDir": tex_number(pnl_zero_raw["mean_direction_cosine_mean"]),
        "SemiBlindDir": tex_number(semi_blind["mean_direction_cosine_mean"]),
        "SemiBlindFOne": tex_number(semi_blind["support_f1_mean"]),
        "SemiNoCalibrationDir": tex_number(semi_raw["mean_direction_cosine_mean"]),
        "SemiNoCalibrationFOne": tex_number(semi_raw["support_f1_mean"]),
    }
    if higher_pnl:
        pnl3_noise = sorted({record["noise_std"] for record in higher_pnl})
        pnl3_noise = pnl3_noise[1] if len(pnl3_noise) > 1 else pnl3_noise[0]
        pnl3_blind = row(higher_pnl, "blind_lattice", pnl3_noise)
        pnl3_pair = row(higher_pnl, "blind_pairwise_lattice", pnl3_noise)
        commands["HigherPnlLatticeDir"] = tex_number(pnl3_blind["mean_direction_cosine_mean"])
        commands["HigherPnlLatticeTargetRecovery"] = percent(pnl3_blind["target_recovery_0.99"])
        commands["HigherPnlPairwiseTargetRecovery"] = percent(pnl3_pair["target_recovery_0.99"])
    if stability_budget and stability_budget.get("summary"):
        budget = stability_budget["summary"]
        commands["StabilityBudgetRecords"] = str(budget["n_records"])
        commands["StabilityBudgetMedianRealized"] = f"{budget['median_realized_angle_degrees']:.2f}"
        commands["StabilityBudgetMedianBound"] = f"{budget['median_prop3_bound_degrees']:.2f}"
        commands["StabilityBudgetThresholdOnly"] = f"{budget['median_threshold_only_degrees']:.2f}"
        commands["StabilityBudgetViolations"] = str(budget["n_bound_violations"])
    high_noise = max(record["noise_std"] for record in coverage if record.get("experiment") == "main")
    core_high = row(coverage, "joint_blind_lattice", high_noise, experiment="main")
    dict_high = row(coverage, "joint_blind_dictionary_learning", high_noise, experiment="main")
    commands["HighNoiseValue"] = f"{high_noise}"
    commands["HighNoiseBlindDir"] = tex_number(core_high["mean_direction_cosine_mean"])
    commands["HighNoiseDictDir"] = tex_number(dict_high["mean_direction_cosine_mean"])
    random_noises = sorted({record["noise_std"] for record in random_support})
    random_high = random_noises[-1]
    random_mid = random_noises[1] if len(random_noises) > 2 else random_noises[-1]
    random_blind_high = row(random_support, "joint_blind_lattice", random_high, experiment="random_support")
    random_dict_high = row(random_support, "joint_blind_dictionary_learning", random_high, experiment="random_support")
    random_blind_mid = row(random_support, "joint_blind_lattice", random_mid, experiment="random_support")
    random_dict_mid = row(random_support, "joint_blind_dictionary_learning", random_mid, experiment="random_support")
    commands["RandomHighNoiseBlindDir"] = tex_number(random_blind_high["mean_direction_cosine_mean"])
    commands["RandomHighNoiseDictDir"] = tex_number(random_dict_high["mean_direction_cosine_mean"])
    commands["RandomMidNoiseBlindDir"] = tex_number(random_blind_mid["mean_direction_cosine_mean"])
    commands["RandomMidNoiseDictDir"] = tex_number(random_dict_mid["mean_direction_cosine_mean"])
    commands["RandomHighNoiseValue"] = f"{random_high}"
    commands["RandomMidNoiseValue"] = f"{random_mid}"
    main_mid_noise = sorted({record["noise_std"] for record in coverage if record.get("experiment") == "main"})[-2]
    main_mid_blind = row(coverage, "joint_blind_lattice", main_mid_noise, experiment="main")
    main_mid_dict = row(coverage, "joint_blind_dictionary_learning", main_mid_noise, experiment="main")
    commands["MainMidBlindSrcFive"] = percent(main_mid_blind["source_5_recovery_0.99"])
    commands["MainMidDictSrcFive"] = percent(main_mid_dict["source_5_recovery_0.99"])
    commands["MainHighBlindSrcFive"] = percent(core_high["source_5_recovery_0.99"])
    commands["MainHighDictSrcFive"] = percent(dict_high["source_5_recovery_0.99"])
    commands["RandomMidBlindSrcFive"] = percent(random_blind_mid["source_5_recovery_0.99"])
    commands["RandomMidDictSrcFive"] = percent(random_dict_mid["source_5_recovery_0.99"])
    saturation = theory.get("selection_rule_saturation")
    if saturation:
        commands["SelectionSaturationScoreCosine"] = tex_number(saturation["score_selection_target_cosine"])
    false_alarm = theory.get("support_selection_false_alarm")
    if false_alarm:
        commands["SupportSelectionLinearError"] = percent(false_alarm["linear_support_error_rate"])
        commands["SupportSelectionPnlError"] = percent(false_alarm["pnl_support_error_rate"])
        commands["SupportSelectionLinearOvershoot"] = percent(false_alarm["linear_overshoot_rate"])
        commands["SupportSelectionPnlOvershoot"] = percent(false_alarm["pnl_overshoot_rate"])
        commands["SupportSelectionTrials"] = str(false_alarm["n_trials"])
    if random_details.get("selection_rule_comparison"):
        noise_tags = {}
        tag_names = ["Low", "Mid", "High", "Vhigh"]
        for record in random_details["selection_rule_comparison"]:
            tag = noise_tags.setdefault(record["noise_std"], tag_names[len(noise_tags)])
            commands[f"SelectionFamilySuccess{tag}"] = f"{record['family_consistency_success']}/{record['n_cases']}"
            commands[f"SelectionScoreSuccess{tag}"] = f"{record['score_rule_success']}/{record['n_cases']}"
        for noise, tag in noise_tags.items():
            commands[f"SelectionNoise{tag}"] = f"{noise}"
        for tag in tag_names:
            commands.setdefault(f"SelectionFamilySuccess{tag}", "--")
            commands.setdefault(f"SelectionScoreSuccess{tag}", "--")
    if random_details.get("cluster_count_rule"):
        lattice_rows = [
            record for record in random_details["cluster_count_rule"]
            if record["method"] == "joint_blind_lattice"
        ]
        if lattice_rows:
            cluster_noise = CORE_NOISE if any(record["noise_std"] == CORE_NOISE for record in lattice_rows) else lattice_rows[0]["noise_std"]
            lattice_rows = [record for record in lattice_rows if record["noise_std"] == cluster_noise]
        for record in lattice_rows:
            tag = "Within" if record["n_clusters_within_source_count"] else "Above"
            commands[f"ClusterRule{tag}Cases"] = str(record["n_cases"])
            commands[f"ClusterRule{tag}Success"] = percent(record["source_5_success_rate"])
        commands.setdefault("ClusterRuleWithinCases", "--")
        commands.setdefault("ClusterRuleWithinSuccess", "--")
        commands.setdefault("ClusterRuleAboveCases", "--")
        commands.setdefault("ClusterRuleAboveSuccess", "--")
    if pnl_details and pnl_details.get("oracle_label_calibration"):
        worst = max(pnl_details["cells"], key=lambda cell: (cell["noise_std"], cell["nonlinearity_strength"]))
        commands["PnlFailureCorrObjectiveCosine"] = tex_number(worst["correlation_objective_reduction_direction_cosine"])
        ari_cells = [
            cell for cell in pnl_details["cells"]
            if cell["noise_std"] == worst["noise_std"] and cell["nonlinearity_strength"] == 1.0
        ]
        if ari_cells:
            commands["PnlFailureCorrAriCosine"] = tex_number(ari_cells[0]["correlation_cluster_ari_direction_cosine"])
        else:
            commands["PnlFailureCorrAriCosine"] = tex_number(worst["correlation_cluster_ari_direction_cosine"])
        oracle_label = [
            cell for cell in pnl_details["oracle_label_calibration"]
            if cell["noise_std"] == worst["noise_std"] and cell["nonlinearity_strength"] == worst["nonlinearity_strength"]
        ]
        if oracle_label:
            commands["PnlOracleLabelsDir"] = tex_number(oracle_label[0]["mean_direction_cosine"])
            commands["PnlOracleLabelsBoundHit"] = percent(oracle_label[0]["bound_hit_rate"])
            commands["PnlOracleLabelsCoefRmse"] = tex_number(oracle_label[0]["mean_coefficient_rmse"])
    with (PROJECT / "paper" / "result_macros.tex").open("w") as handle:
        handle.write("% Auto-generated result macros.\n")
        for name, value in commands.items():
            handle.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    global MODE, CORE_NOISE, HIGHER_NOISE
    MODE = args.mode
    CORE_NOISE = 0.005 if MODE == "full" else 0.01
    HIGHER_NOISE = 0.02 if MODE == "full" else 0.005
    with (RESULTS / "theory_verification.json").open() as handle:
        theory = json.load(handle)
    coverage = load("coverage_summary")
    random_support = load("random_support_summary")
    recovery_intervals = load("recovery_intervals")
    higher = load("higher_order_summary")
    higher_pnl = load("higher_order_pnl_summary")
    pnl = load("pnl_summary")
    pnl_details = load("pnl_diagnostics")
    semi = load("semisynthetic_summary")
    scaling = load("scaling_summary")
    conditioning = load("conditioning_summary")
    complexity = load("lattice_complexity_summary")
    conditioning_details = load("conditioning_diagnostics")
    random_details = load("random_support_diagnostics")
    stability_budget = load("stability_budget")
    selection_filename = "selection_ablation_development_runs.json" if MODE == "full" else "selection_ablation_smoke_runs.json"
    selection_path = RESULTS / selection_filename
    if not selection_path.is_file():
        raise FileNotFoundError(f"Required direction-score ablation is missing: {selection_path}")
    with selection_path.open() as handle:
        selection_development = json.load(handle)
    if not selection_development:
        raise ValueError(f"Direction-score ablation is empty: {selection_path}")
    plot_support_schematic()
    plot_coverage(coverage)
    plot_random_support(random_support, recovery_intervals)
    plot_sample_and_window(coverage)
    plot_higher_order(higher)
    plot_pnl(pnl)
    plot_semisynthetic(semi)
    plot_scaling(scaling)
    plot_conditioning(conditioning)
    plot_lattice_complexity(complexity)
    write_generated_tables(coverage, higher, pnl, semi, recovery_intervals)
    write_supplement_tables(coverage, higher, pnl, semi, scaling, conditioning, complexity, conditioning_details, random_support, recovery_intervals, selection_development, higher_pnl=higher_pnl, stability_budget=stability_budget, pnl_details=pnl_details, random_details=random_details)
    write_macros(theory, coverage, higher, pnl, semi, scaling, random_support, random_details, recovery_intervals, higher_pnl=higher_pnl, stability_budget=stability_budget, pnl_details=pnl_details)
    print(f"Figures and tables written for mode={MODE}")


if __name__ == "__main__":
    main()
