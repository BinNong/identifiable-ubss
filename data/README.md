# Measured source recordings

Download `0_jackson_0.wav`, `0_nicolas_0.wav` and `0_theo_0.wav` from the `recordings/` directory of the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset). Place them in this directory under the names `speech_jackson.wav`, `speech_nicolas.wav` and `speech_theo.wav`, respectively. The recordings are not tracked by this code-only repository.

Dataset authors and contributors are listed in the upstream repository. The dataset is distributed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/). The experiment loaders center each recording and normalize its standard deviation in memory; the stored WAV files are not edited by the pipeline.

These recordings are used only as measured source waveforms. The activation blocks, shared mixing matrix and regime-specific post-nonlinear distortions in the experiments are synthetic, so the experiment is semi-synthetic rather than real-mixture validation.
