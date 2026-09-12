# Sign-aware antithetic coupling for signed gradient particles

Code and numerical evidence for **Sign-Aware Antithetic Coupling for Signed
Gradient-Particle Methods**, by Stephen Abkin and Prabir Daripa.

The method couples two mean-transport gradient-particle simulations by spatial
rank. It reflects Gaussian increments for like-sign masses and synchronizes
increments for unlike-sign masses, preserving each replica's discrete law.
The paper studies computational savings in transient-field estimation for the
heat, Burgers, and cubic-flux equations. The exact variance comparison concerns
a conditioned final diffusion stage; full-history comparisons are numerical.

[Read the manuscript](manuscript/main-2-coupling.pdf) ·
[LaTeX source](manuscript/main-2-coupling.tex)

## Quick start

Use Python 3.12 and a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-coupling.txt -r requirements-test.txt
python -m pytest -q
```

A minimal coupled simulation:

```python
import numpy as np
from coupled_gradient_particles import advance_pair
from relaxation_gbmc import reconstruct_cumulative_field

x = np.linspace(-0.5, 0.5, 100)
m = np.r_[np.full(50, 0.02), np.full(50, -0.02)]
A, B, u_left = advance_pair(
    (x, m, 0.0), nu=0.1, dt=0.005, K=200,
    rng=np.random.default_rng(42), policy="FULL",
    flux_derivative=lambda u: u,
)
grid = np.linspace(-5, 5, 400)
field = 0.5 * (
    reconstruct_cumulative_field(*A, u_left, grid)
    + reconstruct_cumulative_field(*B, u_left, grid)
)
```

`FULL` uses the sign switch, `RAW` reflects by spatial rank, `WITHIN` reflects
within each sign class, and `FINAL` switches only at the final stage. Initial
masses must be nonzero. The estimate above averages the two reconstructed fields.

## Verify and rebuild

From the repository root:

```bash
python manuscript/analysis/build_numbers.py
python manuscript/analysis/verify_evidence.py
python manuscript/analysis/check_manuscript.py
python manuscript/analysis/make_figures.py
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main-2-coupling.tex
```

The numerical commands recompute summaries, check manuscript values, and rebuild
figures from archived data; they do not rerun simulations or reproduce timings.
The PDF build requires a TeX installation with `latexmk` and the packages used
in the source. Springer class and bibliography files are included.

The coupled stepper is in `coupled_gradient_particles.py`, experiment drivers
are in `studies/`, and stored results are in `output/` and `manuscript/evidence/`.
The manuscript identifies the experiments supporting each claim. Full experiment
drivers can overwrite their historical output directories; use a separate copy
for reruns. Timings depend on the machine.

Earlier sampled-velocity and compensation studies are retained as historical
work. `python reproduce.py verify` checks their 164 archived values, not the
coupling paper. `requirements-lock.txt` records the older environment; its fitted
quantities are compared with relative tolerance 1e-8. Some historical diagnostics
were superseded; they are not additional evidence for the current paper.

Citation information is in [CITATION.cff](CITATION.cff). The paper is an
unpublished manuscript; no DOI has been assigned.
