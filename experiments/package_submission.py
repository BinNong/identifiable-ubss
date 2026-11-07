import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


EXCLUDE_FROM_ARCHIVE = {
    # Editorial correspondence and internal to-do lists are not part of the
    # public reproducibility package.
    "paper/cover_letter.pdf",
    "paper/cover_letter.tex",
    "paper/submission_checklist.txt",
}


def selected_files(project):
    files = [project / "README.md", project / "requirements.txt"]
    files.extend(sorted((project / "src").glob("*.py")))
    files.extend(sorted(path for path in (project / "experiments").glob("*.py") if not path.name.startswith("prototype_")))
    files.append(project / "experiments" / "model_spec.json")
    for pattern in ["*.tex", "*.bib", "*.txt", "*.tsv", "*.pdf"]:
        files.extend(sorted((project / "paper").glob(pattern)))
    files.extend(sorted((project / "data").glob("*.wav")))
    files.append(project / "data" / "README.md")
    results = project / "results"
    for name in ["environment_smoke.json", "validation.json", "theory_verification.json"]:
        files.append(results / name)
    full_environment = results / "environment_full.json"
    if full_environment.is_file():
        files.append(full_environment)
    for base in ["coverage", "random_support", "higher_order", "higher_order_pnl", "scaling", "lattice_complexity", "pnl", "semisynthetic", "conditioning"]:
        for suffix in ["runs", "summary", "config"]:
            files.append(results / f"{base}_{suffix}.json")
    files.append(results / "conditioning_diagnostics.json")
    files.append(results / "random_support_diagnostics.json")
    files.append(results / "pnl_diagnostics.json")
    files.append(results / "stability_budget.json")
    for suffix in ["runs", "summary", "config"]:
        files.append(results / f"selection_ablation_development_{suffix}.json")
    files.append(results / "recovery_intervals.json")
    for pattern in ["*.pdf", "*.png"]:
        files.extend(sorted((results / "figures").glob(pattern)))
    return sorted(
        path
        for path in set(files)
        if str(path.relative_to(project)) not in EXCLUDE_FROM_ARCHIVE
    )


def main():
    project = Path(__file__).resolve().parents[1]
    files = selected_files(project)
    missing = [str(path.relative_to(project)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    archive = project / "signal-processing-submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in files:
            handle.write(path, arcname=str(path.relative_to(project)))
    manifest = {
        "archive": archive.name,
        "sha256": digest(archive),
        "n_files": len(files),
        "files": [
            {
                "path": str(path.relative_to(project)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    with (project / "results" / "submission_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"archive": archive.name, "sha256": manifest["sha256"], "n_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
