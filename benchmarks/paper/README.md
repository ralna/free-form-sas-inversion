# Paper benchmarks: ffsi vs FFSAS

Scripts to reproduce the numerical results and figures of

> J. M. Fowkes, N. I. M. Gould, J. Huntley, T. Snow and R. Dalgleish,
> *Fast free-form inversion of polydisperse small-angle scattering data by
> simplex-constrained nonlinear least squares* (submitted to J. Appl. Cryst.).

These scripts are **not** part of the `ffsi` package distributed on PyPI.

## Requirements

* `ffsi` installed in the Python environment (e.g. `pip install -e .` from the
  repository root, or `pip install ffsi`), with CuPy and an NVIDIA GPU (the
  drivers also time the CPU path).
* For the comparison, a separate environment with
  [FFSAS](https://github.com/stfc-sciml/ffsas) and PyTorch (CUDA) installed.

All output (shared problem definitions, results, figures and the LaTeX
tables/number macros used by the paper) is written to `$FFSI_PAPER_DIR`,
which defaults to `benchmarks/paper/output`.

## Running

From the repository root, with the ffsi environment:

```bash
export FFSI_PAPER_DIR=$PWD/benchmarks/paper/output   # optional
python benchmarks/paper/bench_sphere.py             # sphere benchmark
python benchmarks/paper/bench_sphere_challenge.py   # sphere challenge (4 resolutions)
python benchmarks/paper/bench_1d_aniso.py           # orientation-averaged cylinders/ellipsoids
python benchmarks/paper/bench_cylinder2d.py         # 2D cylinders (2.1 GB Green tensor)
python benchmarks/paper/bench_ellipsoid2d.py        # 2D ellipsoids
python benchmarks/paper/bench_real.py               # SANS and SAXS data
python benchmarks/paper/scan_smoothing.py           # smoothing-parameter scan
```

Then, with the FFSAS environment (any working directory, same `FFSI_PAPER_DIR`):

```bash
python benchmarks/paper/ffsas/bench_sphere_benchmark.py
python benchmarks/paper/ffsas/bench_sphere_challenge.py
python benchmarks/paper/ffsas/bench_1d_aniso.py
python benchmarks/paper/ffsas/bench_cylinder.py cylinder2d
python benchmarks/paper/ffsas/bench_cylinder.py ellipsoid2d
python benchmarks/paper/ffsas/bench_real.py
```

Finally, with the ffsi environment:

```bash
python benchmarks/paper/make_figures.py
```

## Notes

* **Identical problems.** PyTorch and NumPy/CuPy produce different random
  streams from the same seed, so every synthetic problem (ground truth,
  intensities, error bars) is generated once by the ffsi drivers and saved to
  `$FFSI_PAPER_DIR/results/problems/*.npz`; the FFSAS scripts only read these
  files, and check that their own Green tensor reproduces the saved intensities
  before timing. The ffsi drivers must therefore be run first.
* **FFSAS settings** follow its published example notebooks (fixed numbers of
  trust-region iterations with zero convergence tolerances). FFSAS has no
  orientation-averaged models, so for those problems the ffsi Green tensor is
  passed to its model-independent solver.
* **Timing.** ffsi times are medians of three runs after a warm-up run; GPU
  work is synchronized before each timer. Do not run two GALAHAD jobs in the
  same working directory at the same time: GALAHAD's `ALIVE.d` file is shared,
  and one job finishing can terminate the other early.
