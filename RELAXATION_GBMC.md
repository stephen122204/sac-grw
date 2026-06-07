# Relaxation GBMC — Burgers particle method

## Summary

`relaxation_gbmc.py` implements the validated BPC two-speed relaxation + Gradient
Brownian Monte Carlo (GBMC) method for viscous Burgers:

$$u_t + u\,u_x = \nu\,u_{xx}$$

**Domain mode**: WHOLE LINE only.  Particles move freely on the real line; no
boundary reflection is applied.

**Supported IC**: stationary viscous shock only.

---

## Particle representation

Gradient particles represent $w = u_x$ via the signed empirical measure

$$w^N(x,t) = \sum_i m_i\,\delta(x - X_i(t))$$

The field $u$ is recovered from the RAW cumulative sum of sorted particles:

$$u(x,t) = u_{-\infty} + \sum_{j:\,X_j \le x} m_j$$

No smoothing is applied to the primary output.

---

## Quantile initialisation (stationary shock)

$$u(x) = -A\tanh\!\left(\frac{A(x-x_c)}{2\nu}\right)$$

Particles are placed at the quantiles of the exact CDF:

$$r_i = \frac{i - \tfrac12}{N},\quad
X_i = x_c + \frac{2\nu}{A}\operatorname{arctanh}(2r_i-1),\quad
m_i = -\frac{2A}{N},\quad
u_{-\infty} = A$$

This recovers the exact IC in the $N\to\infty$ limit.

---

## Validated time-step algorithm

Each step applies a Lie splitting **A then B**.

### Step A — BPC instantaneous-equilibrium relaxation transport

1. **Transport** (no reflection): $X_i \leftarrow X_i + V_i\,\Delta t$, where
   $V_i \in \{-a, +a\}$ are the persistent labels from the previous switch.
2. **Sort** $(X, m, V)$ together by $X$.
3. **Reconstruct**: $u_i = u_{-\infty} + \operatorname{cumsum}(m)_i$
4. **Subcharacteristic check**: raise `RuntimeError` if $\max_i|u_i| \ge a$.
5. **Validate probabilities**: raise `RuntimeError` if any $p_i^+$ outside $(0,1)$.
6. **Stochastic instantaneous-equilibrium switching**:

$$p_i^+ = \frac{a + u_i}{2a},\quad
V_i = \begin{cases}+a & U < p_i^+\\ -a & U \ge p_i^+\end{cases},\quad U \sim \operatorname{Uniform}(0,1)$$

$\mathbb{E}[V_i \mid u_i] = u_i$: recovers Burgers' characteristic in expectation.

Labels are resampled **exactly once per step**, after reconstruction and before diffusion.

### Step B — Brownian diffusion

7. **Diffuse** (no reflection): $X_i \leftarrow X_i + \xi_i$, where $\xi_i \sim \mathcal{N}(0,\,2\nu\Delta t)$.

### Initialisation

Steps 2–6 are applied once before the loop (transport step 1 is skipped at $t=0$).

---

## Relaxation speed

$a$ is a **fixed** scalar for the entire run, set by `config.relaxation_speed_a`.

- Required: $a > A$ (strict subcharacteristic; equivalently $a > \max_i|u_i|$).
- `ValueError` is raised if `relaxation_speed_a` is `None` or $\le 0$.
- `ValueError` is raised if $a \le A$.
- The solver never adjusts $a$ dynamically.

---

## Outside-window tracking

Particles that exit the output window $[0, L]$ are **not removed**.  They:
- contribute correctly to the raw-cumsum reconstruction at output points;
- are counted in `n_outside` reported at the end of the run;
- their absolute mass is reported as `mass_outside_abs`.

---

## Conservation properties

- **Total particle mass**: $\sum_i m_i = -2A$ is preserved exactly throughout
  (masses are never modified; no reflection, no negation).
- **Output u bounds**: $u_{\text{out}} \in [-A, A]$ by construction (raw cumsum
  of all-negative masses anchored at $u_{-\infty} = A$).

---

## Configuration

```jsonc
{
  "equation_type": "burgers",
  "domain_type": "Finite",
  "domain_size": 4.0,
  "relaxation_domain_mode": "whole_line",  // required; only value supported
  "seed": 42,                               // optional; seeds private RNG
  "diff_constant": 0.5,
  "time_step": 0.005,
  "total_time": 0.5,
  "num_points": 400,
  "burgers_mode": "relaxation_gbmc",
  "relaxation_speed_a": 2.0,              // required; must be > amplitude
  "burgers_initial_condition": {
    "type": "stationary_shock",           // only supported IC
    "nu": 0.5,
    "x_center": 2.0,
    "amplitude": 1.0
  }
}
```

---

## Files

| File | Purpose |
|------|---------|
| `relaxation_gbmc.py` | Core solver (`simulate_burgers_relaxation_gbmc`) |
| `test_relaxation_gbmc.py` | Pytest suite (spec §10 tests A–J) |
| `configs/burgers_relaxation_gbmc.json` | Stationary-shock benchmark config |
| `run_n_refinement.py` | N-refinement convergence study |

---

## Usage

```bash
# Run tests
pytest test_relaxation_gbmc.py -v

# End-to-end pipeline
python main.py configs/burgers_relaxation_gbmc.json

# Verification vs exact solution
python verify_solver.py --equation burgers_rbmc

# N-refinement convergence study
python run_n_refinement.py
```

---

## Numerical results

For the stationary-shock benchmark ($A=1$, $\nu=0.5$, $T=0.5$, $N=400$, `seed=42`):

| Metric | Value |
|--------|-------|
| Relative L2 error | ~4–10% |
| $\|u\|_\infty$ range | $\le A = 1$ (exact) |
| Subcharacteristic | $\max|u_i| < a = 2$ at every step |
| Total particle mass | $-2A$ (exact, to machine precision) |

The empirical Monte Carlo convergence of the L2 error in $N$ is consistent with
$O(N^{-1/2})$ for the validated stationary-shock benchmark at fixed $\Delta t = 0.005$.

> **Note**: The established empirical rate for the spread metric is
> $E_{\mathrm{spread}} \sim N^{-0.507}$, 95% CI $[-0.534,\,-0.479]$,
> from the full convergence study at fixed $\Delta t = 0.0025$.
> This is an empirical Monte Carlo spread result, not a proved theorem for the
> full scheme.

---

## References

- Bouchut & Perthame (1993): kinetic relaxation approximation for conservation laws.
- Bossy & Talay (1997): Gradient Brownian Monte Carlo for Burgers via McKean–Vlasov.
- Méléard (1996): propagation of chaos and interacting particle systems.
