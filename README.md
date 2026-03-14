# GRW Feasibility Study — Heat, Burgers, FitzHugh-Nagumo

How to run the code and view the results. For method details, code organization, and verification philosophy, see the accompanying document.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install numpy matplotlib
```

---

## Run simulations

```bash
python main.py configs/heat_step_dirichlet.json
python main.py configs/heat_step_neumann.json
python main.py configs/burgers_stationary_shock.json
python main.py configs/burgers_shock.json
python main.py configs/burgers_traveling_wave.json
python main.py configs/fhn_grw_steady.json
```

With no config file, `main.py` prompts for input interactively.

**Outputs** (in `output/`):
- Heat: `heat_density.png` or `heat_field_dirichlet.png`
- Burgers: `burgers_u.png`
- FHN: `fhn_uv.png`

---

## Run verification

```bash
# All equations
python verify_solver.py --equation all

# Single equation
python verify_solver.py --equation heat
python verify_solver.py --equation burgers --config configs/burgers_stationary_shock.json
python verify_solver.py --equation burgers --config configs/burgers_shock.json
python verify_solver.py --equation burgers --config configs/burgers_traveling_wave.json
python verify_solver.py --equation fhn
python verify_solver.py --equation fhn --config configs/fhn_grw_nonsmooth.json
python verify_solver.py --equation fhn --config configs/fhn_grw_discontinuous.json

# Heat-only GRW checks (glob stats, weight conservation)
python verify_grw.py
python verify_grw.py configs/heat_step_dirichlet.json
```

**Outputs** (in `output/verify/<equation>/`):
- `comparison_plot.png` — numerical vs reference
- `cole_hopf_diagnostics.png` — Burgers only: phi, phi_x, u, error decomposition
- `metrics.json` — L1, L2, max error, relL2, RMSE

**Heat-only verification** (`verify_grw.py`): writes `output/verify_grw_vs_exact.png`.

---

## Config files

| Config | Equation |
|--------|----------|
| `heat_step_dirichlet.json` | Heat, step IC, Dirichlet BC |
| `heat_step_neumann.json` | Heat, step IC, Neumann BC |
| `burgers_stationary_shock.json` | Burgers, A=1 |
| `burgers_shock.json` | Burgers, A=0.5 |
| `burgers_traveling_wave.json` | Burgers, traveling wave |
| `fhn_grw_steady.json` | FHN, steady_solution IC |
| `fhn_grw_nonsmooth.json` | FHN, linear-ramp IC |
| `fhn_grw_discontinuous.json` | FHN, Heaviside IC |

`config_template.jsonc` has annotated examples. Remove `//` comments before use.

For the mixed-method branch (heat GRW + FD Burgers + FD FHN), see `mixed-solvers-validation`.

---

## Changing parameters

Edit the JSON config. Common fields: `time_step`, `total_time`, `num_points`, `diff_constant`, plus equation-specific IC options.
