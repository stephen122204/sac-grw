# Sign-aware antithetic coupling for signed gradient particles

Code, manuscript, and numerical evidence for **Sign-Aware Antithetic Coupling
for Signed Gradient-Particle Methods**, by Stephen Abkin and Prabir Daripa.
This is the coupled field-estimation study on the mean-transport branch. The
older sampled-velocity and variance-compensation studies remain archival;
their solver defaults and recorded results are preserved.

## Current manuscript

- [PDF](manuscript/main-2-coupling.pdf) and [LaTeX source](manuscript/main-2-coupling.tex)
- [Reproduction instructions](manuscript/README.md)
- [Claim and evidence audit](manuscript/REVISION-AUDIT.md)
- [Research archive status](docs/RESEARCH-ARCHIVE.md)

The method matches two replicas by spatial rank and reflects Gaussian increments
for like-sign masses, synchronizing them for unlike-sign masses. It preserves
each replica's discrete law. Its exact variance guarantee concerns one conditioned
final diffusion stage; full nonlinear history comparisons are numerical.

## Install and verify

Use Python 3.11 or 3.12 in a virtual environment:

```bash
python -m pip install -r requirements-coupling.txt
python -m pip install -r requirements-test.txt
python -m pytest -q
python manuscript/analysis/build_numbers.py
python manuscript/analysis/verify_evidence.py
python manuscript/analysis/check_manuscript.py
python manuscript/analysis/make_figures.py
```

`build_numbers.py` recomputes summaries and intervals from stored arrays. It does
not rerun the experiments or recover original wall-clock timings. `verify_evidence.py`
independently checks headline errors, held-out block statistics, and figure states.
The older `python reproduce.py verify` checks 164 legacy results, not the new paper.

## Use the coupled stepper

```python
import numpy as np
from coupled_gradient_particles import advance_pair
from relaxation_gbmc import reconstruct_cumulative_field

x = np.linspace(-0.5, 0.5, 100)
m = np.r_[np.full(50, 0.02), np.full(50, -0.02)]
A, B, u_left = advance_pair(
    (x, m, 0.0), nu=0.1, dt=0.005, K=200,
    rng=np.random.default_rng(42), policy="FULL", flux_derivative=lambda u: u)
grid = np.linspace(-5, 5, 400)
field = 0.5 * (reconstruct_cumulative_field(*A, u_left, grid)
             + reconstruct_cumulative_field(*B, u_left, grid))
```

Policies are `RAW` (rank reflection), `FULL` (rank sign switch), `WITHIN`
(reflection within each sign class), and `FINAL` (switch only the last stage).
Initial masses must be nonzero; both replicas retain the same initial multiset.
Terminal positions are not sorted: carried velocities from the pre-diffusion
rank update drive the next transport. The optional `pre_diffusion` callback
receives copies of the sorted state for diagnostics.

## Historical work

See [the archived README](docs/legacy-compensation-README.md) for the earlier
compensation and two-speed studies. Historical drivers retain their original
round names and may contain superseded diagnostics. Only the evidence named
in the current manuscript's map supports its claims. No convergence-rate or
universal terminal-variance improvement is asserted for the new coupling.
