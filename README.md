# GRW Feasibility Study -- Heat, Burgers, FitzHugh-Nagumo

> **Branch:** `main`
>
> This branch implements and evaluates the **Gradient Random Walk (GRW)** method
> for three PDE classes.  The goal is not to produce the most accurate solver for
> each equation; it is to study whether GRW-based formulations are feasible,
> how closely they reproduce known solutions, and where they fail and why.
> Error metrics are feasibility
> diagnostics, not accuracy claims.

---

## Current status

| Equation | Primary method | Verification reference | Approximate status |
|----------|---------------|------------------------|--------------------|
| Heat     | Direct GRW | Exact analytical solution | Good agreement |
| Burgers  | Cole-Hopf GRW (main) | Exact stationary shock | Approximate; dominant error is finite-domain BC mismatch (~88% of total) |
| Burgers  | Direct derivative GRW (diagnostic only) | High-res FD reference | Severely noisy by design |
| FHN      | Scalar traveling-wave GRW | Exact traveling-wave solution | Good agreement; residual is MC noise |

---

## Research framing

The GRW method models the **gradient** of the solution field rather than the field
itself.  Computational elements called **globs** carry positions and signed weights
representing pieces of the gradient distribution.  The physical field is recovered
by numerically integrating the glob distribution.

Three PDE classes are studied:

- **Heat**: the natural setting for GRW.  The gradient formulation is exact and
  the method has a direct derivation from the heat equation.
- **Burgers**: solved via the Cole-Hopf transformation, which reduces Burgers to a
  heat equation solvable by GRW.  A direct derivative-based GRW path is retained
  only as a diagnostic showing why that approach is impractical.
- **FitzHugh-Nagumo**: a scalar traveling-wave GRW formulation.
  The reaction statistic is derived analytically from the exact traveling-wave
  solution and conserves total weight exactly.

Standard FD implementations of Burgers and FHN are kept internally as
`simulate_burgers_fd` and `simulate_fitzhugh_nagumo_fd` in `simulation.py`,
used only to generate reference solutions in `verify_solver.py`.  They are not
the primary solvers on this branch.

For the full mixed-method codebase (heat GRW + standard FD Burgers + FD FHN),
see the `mixed-solvers-validation` branch.

---

## Method descriptions

### Heat -- direct GRW

The heat equation  u_t = alpha * u_xx  is solved on the gradient variable
v = u_x.  Globs represent pieces of the gradient distribution.  Each time step:

1. **Brownian walk**: displacement drawn from Normal(0, sqrt(2 * alpha * dt)).
2. **Boundary reflection** (overshoot):
   - Dirichlet: symmetric position reflection, weight preserved.
   - Neumann: symmetric position reflection, weight negated.
3. **Reconstruction**: sort globs by position; cumulative sum of signed weights
   gives u(x, t).

Verification: exact error-function analytical solution for a step initial condition.

---

### Burgers -- Cole-Hopf GRW (primary method)

Burgers equation:  u_t + u * u_x = nu * u_xx.

#### Cole-Hopf transformation

The substitution  u = -2*nu * phi_x / phi  maps Burgers into the heat equation
for phi:  phi_t = nu * phi_xx.  GRW evolves phi_x globs with the same Brownian
step as the heat solver (alpha = nu).

**Initialisation:**
1. Psi_0(x) = integral_0^x u_0(s) ds  (trapezoidal).
2. phi_0(x) = exp(-Psi_0(x) / (2*nu)), normalised so max(phi_0) = 1.
3. phi_x globs at midpoints x_{i+1/2} with weight w_i = phi_0(x_{i+1}) - phi_0(x_i).

**Boundary treatment:**
Weight-preserving (symmetric position) reflection is used for phi_x globs.
This implicitly enforces Dirichlet BC: phi(0, t) = phi_0(0) and
phi(L, t) = phi_0(L) remain constant for all t, because the cumsum reconstruction
is always anchored to those values.

**Reconstruction at time T:**
1. Bin phi_x glob weights onto a uniform output grid.
2. Boundary-corrected Gaussian smoothing (sigma_bins = 12); the kernel is
   normalised by its effective support at each bin to remove truncation bias
   at x = 0 and x = L.
3. Enforce correct total integral of smoothed bins.
4. phi(x_j) = phi_0(0) + cumsum(smoothed_bins).
5. phi_x(x_j) = d(phi)/dx  (gradient of reconstructed phi).
6. u(x_j) = -2*nu * phi_x(x_j) / phi(x_j).

**Known limitation -- finite-domain BC mismatch:**
The stationary-shock benchmark is an exact solution on an infinite domain.
On a finite domain [0, L] the GRW Dirichlet BC (phi = 1 at both walls) is
inconsistent with that exact solution: under Dirichlet phi = 1, the heat
equation drives phi toward 1 (flat), flattening the shock amplitude over time.

A deterministic FD reference heat solve (same Dirichlet BC, zero particle noise)
is computed alongside the GRW and shown in the diagnostics figure.  Comparing
  (FD_ref - exact):  BC-mismatch error  -- systematic, ~88% of total rms error
  (GRW - FD_ref):    GRW shot noise     -- stochastic, ~32% of total rms error
confirms that the GRW is correctly implementing the heat equation under its BCs;
the dominant remaining error is the BC mismatch, not a GRW implementation defect.

**Primary benchmark** (`configs/burgers_stationary_shock.json`):
  u_0(x) = -A * tanh(A * (x - x_c) / (2 * nu)),  A = 1,  nu = 0.5,  domain [0, 4].
Exact: u(x, t) = u_0(x) for all t.  Typical 5-run average: relL2 ~ 0.28.

**Secondary benchmark** (`configs/burgers_shock.json`):
Same form with A = 0.5 (wider shock, better-conditioned phi_0 with min ~ 0.71).
Typical 5-run average: relL2 ~ 0.22.

**Conditioning note:** ICs with non-zero background velocity (mean u != 0) cause
phi_0 to span a large dynamic range, amplifying reconstruction noise.  The
stationary-shock ICs (zero mean, phi_0(0) = phi_0(L)) are recommended.  A warning
is printed if the log_phi_0 range exceeds 50.

#### Direct Burgers GRW (diagnostic only, `direct_grw`)

Evolves globs representing v = u_x directly under the reaction-diffusion equation
  v_t = nu * v_xx - u * v_x - v^2.
The reaction statistic requires computing u_xx from the noisy particle field
(two numerical differentiations), which amplifies shot noise severely.  This path
is included to demonstrate that direct GRW for Burgers is
impractical.  Large errors against an FD reference are expected and intentional.

#### Lagrangian GRW (legacy, not a primary solver)

An earlier operator-splitting approach is preserved in `simulate_burgers_lagrangian`
for reference only.  It is not reachable from the `simulate_burgers` dispatcher on
this branch.

---

### FitzHugh-Nagumo -- scalar traveling-wave GRW

The FHN system is reduced to a scalar PDE  u_t = D * u_xx + f(u)  in the
traveling-wave formulation.  Globs represent contributions to u_x; u is
reconstructed by cumulative summation exactly as in the heat GRW.

**Exact traveling-wave solution:**

```
u(x, t) = 1 / (1 + exp(-(x + theta*t - x_center) / 2))
theta    = sqrt(2) * (0.5 - a)
```

**Reaction statistic:**

The statistic R(u) is derived by substituting the exact sigmoid into the PDE,
which requires  f(u) = u*(1-u) * [theta/2 - D*(1 - 2*u)/4].  Differentiating:

```
R(u) = f'(u) = -(3*D/2)*u^2 + (3*D/2 - theta)*u + (theta/2 - D/4)
```

This satisfies  integral_0^1 R(u) du = f(1) - f(0) = 0,  so total glob weight is
exactly conserved.  No per-step renormalization is applied or needed.

**Per-step algorithm:**
1. Brownian walk: x_i += Normal(0, sqrt(2 * D * dt)).
2. Boundary reflection (Dirichlet: preserve weight; Neumann: negate weight).
3. Sort globs by position.
4. Reconstruct u(x_i) = cumulative sum of sorted weights.
5. React: w_i += dt * R(u_i) * w_i.

**Initial condition modes:**

| Type | Description |
|------|-------------|
| `steady_solution` | Globs placed at sigmoid quantile positions; exact at t=0 |
| `nonsmooth` | Linear-ramp IC; globs uniform in the transition zone |
| `discontinuous` | All globs at x_center (Heaviside / Dirac delta IC) |

**Verification:** multi-snapshot comparison against the exact traveling-wave solution.
Typical relL2 ~ 0.02--0.05; residual error is MC shot noise.

**Legacy two-component solver** (`simulate_fitzhugh_nagumo_two_component`) is
retained for configs using the legacy `stimulated_region` IC format and is not
the primary FHN method on this branch.

---

## Verification

```
Heat                                  -> exact erf analytical solution
Burgers (cole_hopf_grw, stationary)   -> exact stationary-shock solution
Burgers (cole_hopf_grw, diagnostics)  -> FD reference heat solve (same BCs as GRW)
Burgers (direct_grw)                  -> high-res FD reference (noise expected)
FHN scalar GRW                        -> exact traveling-wave solution (multi-time)
```

The Cole-Hopf diagnostics figure (`cole_hopf_diagnostics.png`) shows three curves
per panel: GRW (blue), FD reference with same BCs (orange), exact infinite-domain
shape (black).  The error-decomposition panel quantifies BC-mismatch vs GRW noise
separately.

The FD reference solver (`simulate_burgers_fd`) uses adaptive sub-cycled time
stepping to satisfy both diffusion and advection CFL conditions.

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
python main.py configs/burgers_stationary_shock.json   # Cole-Hopf GRW (primary)
python main.py configs/burgers_shock.json              # Cole-Hopf GRW (A=0.5)
python main.py configs/fhn_grw_steady.json
```

### 4. Run the verification suite

```bash
python verify_solver.py --equation heat
python verify_solver.py --equation burgers --config configs/burgers_stationary_shock.json
python verify_solver.py --equation burgers --config configs/burgers_shock.json
python verify_solver.py --equation fhn
python verify_solver.py --equation fhn --config configs/fhn_grw_nonsmooth.json
python verify_solver.py --equation fhn --config configs/fhn_grw_discontinuous.json
```

Outputs are written to `output/verify/<equation>/`:
- `comparison_plot.png`        -- numerical vs reference overlay + pointwise error
- `cole_hopf_diagnostics.png`  -- Burgers transformed-variable error decomposition
- `metrics.json`               -- L1, L2, max|err|, relL2, RMSE, equation-specific

---

## Repository layout

```
heat_burgers_fhn/
+-- main.py                    Entry point
+-- simulation.py              GRW solvers:
|                                heat: direct GRW
|                                Burgers: Cole-Hopf GRW (main), direct GRW (diagnostic),
|                                         Lagrangian GRW (legacy, not in main path)
|                                FHN: scalar traveling-wave GRW (main),
|                                     legacy two-component GRW
|                              Also: FD reference variants for verification
+-- config.py                  Config loading, validation, IC generators
+-- utils.py                   Plotting utilities
+-- verify_solver.py           Verification runner; exact and FD reference comparisons
+-- config_template.jsonc      Master config reference (annotated)
+-- configs/
|   +-- heat_step_dirichlet.json
|   +-- heat_step_neumann.json
|   +-- burgers_stationary_shock.json    Cole-Hopf GRW, A=1 (primary benchmark)
|   +-- burgers_shock.json               Cole-Hopf GRW, A=0.5 (secondary benchmark)
|   +-- burgers_direct_grw_diagnostic.json  Direct GRW diagnostic (noisy by design)
|   +-- fhn_grw_steady.json              FHN scalar GRW, steady_solution IC
|   +-- fhn_grw_nonsmooth.json           FHN scalar GRW, linear-ramp IC
|   +-- fhn_grw_discontinuous.json       FHN scalar GRW, Heaviside IC
|   +-- fitzhugh_nagumo_pulse.json       Legacy two-component GRW
|   +-- fhn_oscillatory.json             Legacy two-component GRW
+-- output/                    Generated figures and metrics (auto-created)
```

---

## Config reference

**`config_template.jsonc`** is the master reference with three self-contained
annotated example configs (one per equation).  Strip `//` comments before use;
plain `.json` is required.

### Global fields (all equations)

| Field                | Type    | Description |
|----------------------|---------|-------------|
| `equation_type`      | string  | `"heat"` / `"burgers"` / `"fitzhugh-nagumo"` |
| `domain_type`        | string  | `"Finite"` / `"Semi-Infinite"` / `"Infinite"` |
| `domain_size`        | number  | Right endpoint L; domain is [0, L] |
| `boundary_conditions`| object  | `LEFT` and `RIGHT` sub-objects with `type` and `value` |
| `diff_constant`      | number  | Diffusivity (heat/FHN) or viscosity (Burgers) |
| `time_step`          | number  | dt > 0 |
| `total_time`         | number  | Total simulation time T |
| `num_points`         | integer | Number of globs / particles |

---

## Validation errors

Config errors are caught before the simulation starts:

```
ValueError: Config is missing required field: 'diff_constant'.
ValueError: 'equation_type' must be one of ['burgers', 'fitzhugh-nagumo', 'heat'].
ValueError: boundary_conditions.LEFT.type must be one of ['dirichlet', 'neumann'].
```
