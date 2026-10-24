# Joint identifiability of underdetermined PNL mixtures

Experiment code for joint identifiability of underdetermined post-nonlinear mixtures across regimes. This source-only repository does not include the manuscript, generated results, or audio recordings.

## Model and evidence scope

The population lifting theorem assumes observed regime labels, a shared full-spark mixing matrix, known source count and sparsity, real-analytic input-scale-normalized PNL classes, and observation-domain calibration uniqueness. Scale normalization alone does not prove calibration uniqueness; Supplementary Section S1 contains a full-spark nonlinear-reparameterization counterexample. A merely C1 generalization is not claimed. The experiments use the normalized family `f(z) = z + beta*tanh(z)`.

The higher-order, scaling and lattice-complexity experiments isolate linear support-subspace geometry. Random support experiments and both baselines use the same normalized calibration. Measured speech is used only for semi-synthesis; activation blocks, mixing and PNL distortion are synthetic. These experiments are not independently recorded real-mixture validation.

## Reproduction

Create and activate a Python environment, then install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The semi-synthetic experiment requires three recordings from the Free Spoken Digit Dataset. Follow `data/README.md` to download and name them before running the pipeline. The WAV files are intentionally excluded from Git.

Run a reduced end-to-end check with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/run_all.py --smoke --workers 2
```

The full experiment suite uses confirmation seeds and takes substantially longer:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/run_all.py --full --workers 6
```

The pipeline runs on CPU and performs the following steps:

1. verifies scale and calibration counterexamples, support-coverage constructions, direction-score redundancy invariance and the local stability bound;
2. runs the direction-score ablation and all requested confirmation-seed experiments, random support families, and both adapted baselines;
3. regenerates every figure, LaTeX result table and numerical macro from saved JSON;
4. validates record counts, confirmation seeds, finite values, theory checks and paired failure-accounting consistency.

The development-only direction-score ablation uses previously inspected seeds and can also be rerun separately:

```bash
python experiments/run_selection_ablation.py --mode development --workers 6
```

Generated JSON and figures are written to `results/`. Numerical LaTeX tables are generated locally under `paper/`; no manuscript is tracked. `experiments/package_submission.py` additionally requires the local manuscript and full results and does not publish them.

## Files

- `src/`: calibration, subspace, sparse-recovery and evaluation code.
- `experiments/`: experiment entry points, prototypes, plotting, validation and local submission packaging.
- `experiments/model_spec.json`: machine-readable model assumptions and evidence scope.
- `data/README.md`: source-data attribution and download instructions; WAV files are not tracked.
- `results/`: generated experiment output, excluded from Git.
- `paper/`: generated numerical tables, excluded from Git.
