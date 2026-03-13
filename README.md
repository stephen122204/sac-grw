# GRW Feasibility Study — Heat, Burgers, FitzHugh-Nagumo

> **Branch:** `main`
>
> This branch investigates whether the **Gradient Random Walk (GRW)** method can be
> formulated and applied meaningfully across three PDE classes.  The goal is not to
> use the most accurate standard solver for each equation.  The goal is to study GRW
> feasibility, qualitative fidelity, stability, and limitations — including where GRW
> works well and where it struggles.

---

## Research framing

| Equation | Solver on this branch | Status |
|----------|-----------------------|--------|
| Heat     | Thesis-faithful GRW   | Validated against exact erf solution |
| Burgers  | Experimental GRW-inspired Lagrangian particle method | Compared against FD reference |
| FHN      | Experimental GRW-inspired particle method | Compared against FD reference |

Lower accuracy than a standard FD solver is acceptable as a research outcome, provided
the implementation is methodologically honest and the limitations are measured.  The
error metrics in `verify_solver.py` are feasibility diagnostics, not accuracy claims.

Standard FD implementations of Burgers and FHN are **not** used as the primary solvers
here.  They are kept internally as `simulate_burgers_fd` and `simulate_fitzhugh_nagumo_fd`
in `simulation.py`, used only for generating reference solutions in `verify_solver.py`.

For the full mixed-method validation codebase (heat GRW + standard FD Burgers + standard
FD FHN), see the `mixed-solvers-validation` branch.

---

## Method descriptions

### Heat — thesis-faithful GRW

The heat equation u\_t = alpha \* u\_xx is transformed so the solver works on the
**gradient variable** v = u\_x rather than u directly.  Computational elements called
**globs** represent pieces of the gradient distribution.  At each time step every glob
undergoes a Brownian displacement drawn from

```
Normal(0, sqrt(2 * alpha * dt))
```

Boundary handling uses overshoot reflection:

| BC type   | Position after reflection        | Glob value after reflection  |
|-----------|----------------------------------|------------------------------|
| Dirichlet | symmetric (`x -> -x` or `2L-x`) | preserved                    |
| Neumann   | symmetric (same)                 | **negated** (anti-symmetric) |

The heat field u(x, t) is recovered by **sorting globs by position and cumulatively
summing their signed values** — numerical integration of u\_x.

Verification: exact error-function analytical solution.

### Burgers — experimental GRW-inspired Lagrangian particle method

Burgers' equation: u\_t + u \* u\_x = nu \* u\_xx.

Operator splitting at each step:

1. **Characteristic advection (Lagrangian)**: `x_i += u_i * dt`
   Each particle follows its own inviscid characteristic.  The carried velocity
   `u_i` is held fixed at its initial value (exact for nu=0; an approximation for
   nu>0 since diffusion does not feed back into the carried velocity).

2. **Viscous diffusion (GRW)**: `x_i += Normal(0, sqrt(2 * nu * dt))`
   The diffusion term is modelled by a Brownian random walk, identical to the
   heat GRW with alpha = nu.

3. **Boundary reflection**: same overshoot-reflection rules as the heat GRW.

Reconstruction: u(x, t) is read back by sorting particles by final position and
interpolating.  Near a shock, characteristics converge and cause particle clustering;
this degrades reconstruction accuracy there.

Verification: high-resolution FD reference (`simulate_burgers_fd`).

### FitzHugh-Nagumo — experimental GRW-inspired particle method

FHN equations:

```
du/dt = D * d2u/dx2  +  tau * (u - u^3/3 + v)
dv/dt =               - (1/tau) * (u - a + b*v)
```

Each particle carries a position x\_i and a local state [u\_i, v\_i].

1. **Spatial diffusion of u (GRW)**: `x_i += Normal(0, sqrt(2 * D * dt))`
   Brownian walk with alpha = D, identical to the heat GRW.

2. **Boundary reflection**: Dirichlet — symmetric; Neumann — anti-symmetric
   (same rules as heat GRW).

3. **Local reaction (explicit Euler per particle)**:
   Each particle integrates its own (u\_i, v\_i) ODE independently, without
   spatial coupling to neighbouring particles.

Known limitations of this formulation (part of the research):
- Reaction terms are local per particle; there is no explicit spatial activation
  coupling between neighbours.  Wave propagation relies on diffusion of excited
  particles into the rest region, not on the threshold-activation mechanism of
  the true field PDE.
- v is transported with the particle (v effectively diffuses via particle
  position transport, contrary to the true FHN where v has no spatial diffusion).

Verification: high-resolution FD reference (`simulate_fitzhugh_nagumo_fd`).

---

## Verification philosophy

```
Heat    → exact analytical solution (erf), valid for step IC
Burgers → high-resolution FD reference
FHN     → high-resolution FD reference
```

For heat, the GRW is expected to closely match the exact solution.  The diagnostics
(weight conservation, observed vs theoretical sigma) confirm the random walk itself
is behaving correctly.

For Burgers and FHN, the error against the FD reference is a measure of how far the
experimental GRW formulation is from a trusted numerical solution.  Large errors are
expected near shocks (Burgers) and at the wavefront (FHN), and they are part of the
feasibility finding.

---

## Quick start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install numpy matplotlib
```

### 3. Run a simulation

```bash
python main.py configs/heat_step_dirichlet.json
python main.py configs/burgers_shock.json
python main.py configs/fhn_oscillatory.json
```

### 4. Run the verification suite

```bash
python verify_solver.py              # all three equations
python verify_solver.py --equation heat
python verify_solver.py --equation burgers
python verify_solver.py --equation fhn
```

Outputs are written to `output/verify/<equation>/`:
- `comparison_plot.png`   — numerical vs reference overlay + pointwise error
- `metrics.json`          — L1, L2, max|err|, relL2, RMSE, equation-specific diagnostics

---

## Repository layout

```
heat_burgers_fhn/
├── main.py                    Entry point
├── simulation.py              GRW solvers (heat: thesis-faithful;
│                              Burgers, FHN: experimental GRW-inspired)
│                              Also contains _fd reference variants
├── config.py                  Config loading, validation, IC generators
├── utils.py                   Plotting
├── verify_solver.py           Verification: exact (heat) or FD reference (Burgers, FHN)
├── config_template.jsonc      Master config reference (annotated examples)
├── configs/                   Ready-to-run example configs
│   ├── heat_step_dirichlet.json
│   ├── heat_step_neumann.json
│   ├── burgers_shock.json
│   ├── fitzhugh_nagumo_pulse.json
│   └── fhn_oscillatory.json
├── json_tests/                Legacy test configs (still compatible)
└── output/                    Generated PNGs and verification outputs (auto-created)
```

---

## Config reference

**`config_template.jsonc`** at the repo root is the master reference.
It contains three self-contained, fully commented example configs — one for each
equation.  To create a new config: copy the relevant block, strip `//` comments,
and edit the values.

> `.jsonc` files use `//` comments and are not parseable by Python's `json` module.
> Actual run configs must be plain `.json` files.

### Global fields (all equations)

| Field                | Type    | Required | Description |
|----------------------|---------|----------|-------------|
| `equation_type`      | string  | yes      | `"heat"` \| `"burgers"` \| `"fitzhugh-nagumo"` |
| `domain_type`        | string  | yes      | `"Finite"` \| `"Semi-Infinite"` \| `"Infinite"` |
| `domain_size`        | number  | yes      | Right endpoint L; domain is [0, L]. |
| `boundary_conditions`| object  | yes*     | `LEFT` and `RIGHT` sub-objects. |
| `diff_constant`      | number  | yes      | Diffusivity / viscosity. |
| `time_step`          | number  | yes      | dt > 0. |
| `total_time`         | number  | yes      | Total simulation time T > 0. |
| `num_points`         | integer | yes      | Number of globs / particles / grid points. |

---

## Validation errors

If a config file has missing or invalid fields, the solver prints a clear error
before running.  Examples:

```
ValueError: Config is missing required field: 'diff_constant'.
ValueError: 'equation_type' must be one of ['burgers', 'fitzhugh-nagumo', 'heat'], got 'Heat_eq'.
ValueError: boundary_conditions.LEFT.type must be one of ['dirichlet', 'neumann'], got 'fixed'.
```
