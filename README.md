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
| Burgers  | Cole-Hopf GRW (main); direct GRW diagnostic; Lagrangian GRW (legacy) | Cole-Hopf: exact analytical solution; others: FD reference |
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

### Burgers — Cole-Hopf GRW (thesis-faithful main method)

Burgers' equation: u\_t + u \* u\_x = nu \* u\_xx.

The **Cole-Hopf transformation** (`config: burgers_mode = "cole_hopf_grw"`, default)
is the primary thesis-faithful approach.  It is the method described in Section 5 of
the thesis for reducing Burgers to a heat equation solvable by GRW.

#### Cole-Hopf GRW (main method, `cole_hopf_grw`)

The transformation  u = -2\*nu \* phi\_x / phi  maps Burgers into the heat equation:
  phi\_t = nu \* phi\_xx

Steps at t=0:
1. Compute Psi\_0(x) = integral\_0^x u\_0(s) ds  (trapezoidal rule).
2. phi\_0(x) = exp(-Psi\_0(x) / (2\*nu)), normalized so phi\_0\_max = 1.
3. phi\_0\_x(x) = -u\_0(x) \* phi\_0(x) / (2\*nu); discretize into N phi\_x globs.

GRW evolution (each time step):
- Brownian walk: `x_i += Normal(0, sqrt(2*nu*dt))`  (same step as heat GRW, alpha=nu)
- Neumann boundary reflection: phi\_x = 0 at walls (equivalent to u=0 at boundaries;
  use a large domain so the wave does not reach the walls during the simulation).

Reconstruction at time T:
1. Bin phi\_x glob weights onto uniform output grid.
2. phi(x\_j) = 1 + cumsum(bin\_weights) (cumulative integral of phi\_x).
3. u(x\_j) = -2\*nu \* phi\_x(x\_j) / phi(x\_j).

**Conditioning note**: phi\_0 varies as exp(-Psi\_0/(2\*nu)).  For small nu or large
domains, phi\_0 can span many orders of magnitude, causing numerical overflow.  The
benchmarks use nu=0.5 and domain [0, 4], where conditioning is manageable.  A warning
is printed if the log\_phi0 range exceeds 50.

**Primary benchmark** (`configs/burgers_stationary_shock.json`): stationary shock IC
  u\_0(x) = -A \* tanh(A \* (x - x\_c) / (2 \* nu))
with **exact stationary solution** u(x,t) = u\_0(x) for all t >= 0.

This is an exact solution to u\_t + u\*u\_x = nu\*u\_xx for any amplitude A and viscosity
nu.  The Cole-Hopf phi satisfies phi\_0(0) = phi\_0(L) (symmetric domain), so the
cumulative sum reconstruction of phi returns to its starting value — well-conditioned
for the GRW.  Expected relL2 approximately 0.30–0.35 at N=400 due to GRW particle
noise and Gaussian smoothing; this is the honest accuracy of the method.

**Additional benchmark** (`configs/burgers_traveling_wave.json`): traveling wave IC
  u\_0(x) = 1 - 2\*sqrt(nu) \* tanh((x - x\_0) / sqrt(nu))
with exact solution u(x,t) = 1 - 2\*sqrt(nu) \* tanh((x - x\_0 - t) / sqrt(nu)).
Note: this IC has phi\_0(0) != phi\_0(L) on a finite domain, causing systematic
reconstruction errors near the domain boundaries.  Reported for completeness but the
stationary shock benchmark is recommended for evaluating the Cole-Hopf GRW pipeline.

#### Direct Burgers GRW (diagnostic path, `direct_grw`)

The direct approach evolves globs representing v = u\_x directly.  The gradient equation
  v\_t = nu\*v\_xx - u\*v\_x - v^2
leads to the per-glob reaction statistic:
  R(u) = -(u \* u\_xx / u\_x + u\_x)

This requires computing u\_xx from the noisy particle field (two numerical differentiations),
which amplifies statistical noise.  The result is expected to be severely noisy and
impractical as a primary solver.  This path is included as a **diagnostic** to reproduce
the thesis discussion of why the direct GRW method for Burgers fails.

Use `configs/burgers_direct_grw_diagnostic.json` (nu=0.01, T=0.03) to reproduce
this experiment.  Verification is against a high-resolution FD reference.

#### Lagrangian GRW (experimental, `lagrangian_grw`)

An earlier experimental method using operator splitting (Lagrangian advection +
GRW diffusion).  Kept for comparison with the shock IC (`configs/burgers_shock.json`).

Verification: high-resolution FD reference (`simulate_burgers_fd`).

### FitzHugh-Nagumo — thesis scalar GRW (traveling-wave formulation)

This branch implements the FHN GRW method following Chapter 4 of the thesis.
The formulation is a **scalar** traveling-wave GRW, not a standard two-component
reaction-diffusion solver.

**Scalar equation and exact solution**

The thesis reduces the FHN system to a scalar PDE with a reaction term whose
derivative is:

```
R(u) = -3*u^2 + 2*(0.5 - a)*u - a
```

The exact traveling-wave solution is:

```
u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
theta    = sqrt(2) * (0.5 - a)
```

**GRW gradient-side algorithm (globs represent u\_x)**

Each glob carries a position `x_i` and a scalar weight `w_i` (contribution to
the cumulative-sum reconstruction of `u`).  The algorithm per time step:

1. **Brownian walk**: `x_i += Normal(0, sqrt(2 * D * dt))`
2. **Boundary reflection**: Dirichlet — symmetric (preserve weight);
   Neumann — anti-symmetric (negate weight).
3. **Sort globs** by position.
4. **Reconstruct** `u(x_i) = sum_{k <= i} w_k`  (cumulative sum, as in heat GRW).
5. **React**: `w_i += dt * R(u_i) * w_i`
   followed by renormalization `w_i /= sum(w)` to maintain total weight = 1.

The renormalization is required because `int_0^1 R(u) du = -1/2 - 2*a != 0`
for positive `a`, so the multiplicative weight update does not conserve total
mass.  This reflects a mathematical inconsistency in the specified reaction
statistic (discussed in the thesis and known from the analytical derivations).

**Initial condition modes** (configured via `fhn_initial_condition.type`):

| Type | Description |
|------|-------------|
| `steady_solution` | Globs at `x_i = x_center - 2*log(1/u_i - 1)`, `u_i` uniform on (0,1) |
| `nonsmooth` | Linear-ramp IC; globs uniformly placed in a transition zone |
| `discontinuous` | All globs at `x_center` (Heaviside / Dirac delta IC) |

**Verification**

The GRW output is compared against the exact traveling-wave solution at multiple
time snapshots (t = 0, T/3, 2T/3, T).  Error metrics (L2, max|err|, relL2, front
location difference) are saved to `metrics.json`.

**Legacy two-component solver** (retained for backward compatibility)

The older two-component GRW particle method (`simulate_fitzhugh_nagumo_two_component`)
is still available for configs using the legacy `stimulated_region` IC format.
It is not the primary FHN method on this branch.

---

## Verification philosophy

```
Heat             → exact analytical solution (erf), valid for step IC
Burgers (cole_hopf_grw + stationary_shock IC)  → exact analytical stationary solution
Burgers (cole_hopf_grw + traveling_wave IC)    → exact analytical traveling wave solution
Burgers (direct_grw, lagrangian_grw)           → high-resolution stable FD reference
FHN (scalar GRW) → exact traveling-wave solution (analytic, multi-time)
```

For heat, the GRW is expected to closely match the exact solution.  The diagnostics
(weight conservation, observed vs theoretical sigma) confirm the random walk itself
is behaving correctly.

For Burgers Cole-Hopf GRW with the stationary shock IC, comparison against the exact
solution measures the full pipeline: IC transformation, GRW evolution, and
reconstruction.  The expected relL2 is approximately 0.30–0.35, an honest quantification
of the GRW particle noise and Gaussian-smoothing reconstruction bias.

For direct GRW, large errors against the FD reference confirm the thesis finding
that the direct method is impractical.  The noise is intentional.

The FD reference solver (`simulate_burgers_fd`) uses adaptive sub-cycled time stepping
to satisfy both diffusion and advection CFL conditions, guaranteeing stability for any
nu and initial condition.

For FHN (scalar GRW, thesis formulation), comparison is against the exact
traveling-wave solution.  The reaction statistic R(u) as specified in the thesis
is mathematically inconsistent (does not conserve total weight for typical a > 0),
so a mass-correction renormalization is applied at each step.  The resulting errors
against the exact solution quantify the feasibility of this GRW formulation and
document the known mathematical limitations of the specified reaction statistic.

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
python main.py configs/burgers_stationary_shock.json # Cole-Hopf GRW (primary benchmark)
python main.py configs/burgers_traveling_wave.json   # Cole-Hopf GRW (additional benchmark)
python main.py configs/burgers_shock.json            # Lagrangian GRW (legacy)
python main.py configs/fhn_oscillatory.json
```

### 4. Run the verification suite

```bash
python verify_solver.py              # all three equations (uses default configs)
python verify_solver.py --equation heat
python verify_solver.py --equation burgers --config configs/burgers_stationary_shock.json   # Cole-Hopf vs exact
python verify_solver.py --equation burgers --config configs/burgers_traveling_wave.json     # Cole-Hopf vs exact
python verify_solver.py --equation burgers --config configs/burgers_direct_grw_diagnostic.json  # noise diagnostic
python verify_solver.py --equation burgers --config configs/burgers_shock.json
python verify_solver.py --equation fhn                                       # steady_solution IC (default)
python verify_solver.py --equation fhn --config configs/fhn_grw_nonsmooth.json
python verify_solver.py --equation fhn --config configs/fhn_grw_discontinuous.json
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
│                              Burgers: Cole-Hopf GRW main + direct GRW diagnostic +
│                              Lagrangian GRW legacy; FHN: thesis scalar GRW main +
│                              legacy two-component GRW)
│                              Also contains _fd reference variants for verification
├── config.py                  Config loading, validation, IC generators
├── utils.py                   Plotting
├── verify_solver.py           Verification: exact (heat; Burgers stationary_shock/traveling_wave;
│                              FHN scalar GRW); FD reference (Burgers direct/lagrangian; FHN legacy)
├── config_template.jsonc      Master config reference (annotated examples)
├── configs/                   Ready-to-run example configs
│   ├── heat_step_dirichlet.json
│   ├── heat_step_neumann.json
│   ├── burgers_stationary_shock.json    Cole-Hopf GRW, stationary shock IC (primary benchmark)
│   ├── burgers_traveling_wave.json      Cole-Hopf GRW, traveling wave IC (additional benchmark)
│   ├── burgers_direct_grw_diagnostic.json  Direct GRW diagnostic (nu=0.01, noisy by design)
│   ├── burgers_shock.json               Lagrangian GRW, shock IC (legacy)
│   ├── fhn_grw_steady.json              Thesis FHN GRW, steady_solution IC (default benchmark)
│   ├── fhn_grw_nonsmooth.json           Thesis FHN GRW, linear-ramp non-smooth IC
│   ├── fhn_grw_discontinuous.json       Thesis FHN GRW, Heaviside (discontinuous) IC
│   ├── fitzhugh_nagumo_pulse.json       Legacy two-component GRW
│   └── fhn_oscillatory.json             Legacy two-component GRW
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
