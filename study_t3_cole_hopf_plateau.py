"""Task 3: Diagnose the Cole-Hopf error plateau via four controlled experiments.

  A: domain sensitivity (fixed shock, varying domain size)
  B: deterministic transform control (exact phi, then differentiate/invert)
  C: particle-phi reconstruction control (N vs 2N particles)
  D: output/reference grid sensitivity
Output: output/final_prepublication_tests/cole_hopf_plateau/
"""
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SimulationConfig, generate_burgers_stationary_shock_ic
from verify_solver import exact_burgers_stationary_shock, compute_metrics

OUT_BASE = 'output/final_prepublication_tests/cole_hopf_plateau'


def _mk(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    return p


def _savefig(fig, path_noext):
    for ext in ('png', 'pdf'):
        p = path_noext + '.' + ext
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _run_cole_hopf(N, nu, T, dt, L, A, xc, seed):
    """Single Cole-Hopf GRW run; return (x_out, u_out, elapsed)."""
    np.random.seed(seed)
    from simulation import simulate_burgers_cole_hopf_grw

    ic = generate_burgers_stationary_shock_ic(L, N, nu, x_center=xc, amplitude=A)
    cfg = SimulationConfig(
        equation_type='burgers',
        domain_type='Finite',
        domain_size=L,
        boundary_conditions={'LEFT': {'type': 'Dirichlet', 'value': 0.0},
                             'RIGHT': {'type': 'Dirichlet', 'value': 0.0}},
        diff_constant=nu,
        time_step=dt,
        total_time=T,
        num_points=N,
        initial_conditions=ic,
        reaction_term=False,
        burgers_mode='cole_hopf_grw',
        burgers_ic_type='stationary_shock',
        burgers_ic_amplitude=A,
    )
    globs = [{'position': float(p), 'value': [float(v)]} for p, v in ic]
    t0 = time.perf_counter()
    result = simulate_burgers_cole_hopf_grw(globs, cfg)
    elapsed = time.perf_counter() - t0
    x_out = np.array([g['position'] for g in result])
    u_out = np.array([g['value'][0] for g in result])
    return x_out, u_out, elapsed


def study_A_domain_sensitivity(nu=0.5, A=1.0, T=0.5, dt=0.005,
                                N_per_unit=100, S=10, base_seed=42):
    """Study A: vary domain size with shock at center."""
    print(f"\n{'='*60}\n  Study A: Domain sensitivity\n{'='*60}")
    domains = [(4.0, 100), (8.0, 200), (10.0, 250), (16.0, 400)]
    # N proportional to domain keeps particle density constant; a fixed-N variant
    # isolates domain-size effects from resolution effects.
    results_prop = []
    results_fixed = []

    N_fixed = 400
    rows = []

    for L, N_prop in domains:
        xc = L / 2.0
        sep = L / 2.0  # separation of shock from boundaries

        for N, mode in [(N_prop, 'proportional'), (N_fixed, 'fixed_N')]:
            l2_list = []; linf_list = []
            for rep in range(S):
                seed = base_seed + rep
                try:
                    x_out, u_out, _ = _run_cole_hopf(N, nu, T, dt, L, A, xc, seed)
                    u_exact = exact_burgers_stationary_shock(x_out, nu, x_center=xc, amplitude=A)
                    dx = float(x_out[1] - x_out[0]) if len(x_out) > 1 else 1.0
                    l2 = float(np.sqrt(np.sum((u_out - u_exact)**2) * dx))
                    linf = float(np.max(np.abs(u_out - u_exact)))
                    l2_list.append(l2); linf_list.append(linf)
                except Exception as e:
                    print(f"    WARNING L={L} N={N} seed={seed}: {e}")

            if l2_list:
                row = {
                    'L': L, 'N': N, 'mode': mode, 'sep': sep,
                    'shock_to_boundary_ratio': sep / (2.0 * nu / A),
                    'l2_mean': float(np.mean(l2_list)),
                    'l2_std': float(np.std(l2_list)),
                    'linf_mean': float(np.mean(linf_list)),
                }
                rows.append(row)
                if mode == 'proportional':
                    results_prop.append(row)
                else:
                    results_fixed.append(row)
                print(f"  L={L:5.1f} N={N:5d} ({mode:15s}) "
                      f"sep={sep:.1f}  L2={row['l2_mean']:.4f}±{row['l2_std']:.4f}")

    with open(_mk(OUT_BASE, 'study_A_domain.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, recs, title in [
        (axes[0], results_prop, 'Proportional N (fixed density)'),
        (axes[1], results_fixed, f'Fixed N={N_fixed}'),
    ]:
        if recs:
            L_arr = np.array([r['L'] for r in recs])
            l2_arr = np.array([r['l2_mean'] for r in recs])
            l2_std = np.array([r['l2_std'] for r in recs])
            ax.errorbar(L_arr, l2_arr, yerr=l2_std, fmt='bo-', lw=1.5, ms=6,
                        capsize=4, label='L2 error')
            ax.set_xlabel('Domain size L')
            ax.set_ylabel('L2 error')
            ax.set_title(f'Cole-Hopf: error vs domain size\n{title}')
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'cole_hopf_error_vs_domain'))
    return rows


def study_B_deterministic_transform(nu=0.5, A=1.0, T=0.5, L=4.0, N_grid=400):
    """Study B: exact phi (no particles) + numerical differentiation/inversion,
    isolating the transform step from particle noise.
    """
    print(f"\n{'='*60}\n  Study B: Deterministic transform control\n{'='*60}")

    xc = L / 2.0
    x_grid = np.linspace(0.0, L, N_grid)
    dx = float(x_grid[1] - x_grid[0])

    u_exact = exact_burgers_stationary_shock(x_grid, nu, x_center=xc, amplitude=A)

    # Exact phi(x,T) for the stationary shock:
    # phi(x,T) = C(T) * cosh(A*(x-xc)/(2*nu)) (time-dependent prefactor cancels in ratio)
    # Normalised: phi_norm(x) = cosh(A*(x-xc)/(2*nu)) / cosh(A*xc/(2*nu))
    arg = A * (x_grid - xc) / (2.0 * nu)
    phi_exact = np.cosh(arg) / float(np.cosh(A * xc / (2.0 * nu)))
    phi_x_over_phi_exact = (A / (2.0 * nu)) * np.tanh(arg)
    u_exact_from_transform = -2.0 * nu * phi_x_over_phi_exact

    results = {}

    # Method 1: exact phi, exact ratio
    err_exact_ratio = float(np.sqrt(np.sum((u_exact_from_transform - u_exact)**2 * dx)))
    print(f"  Exact phi, exact ratio -> L2={err_exact_ratio:.6e}  (sanity check, should be ~0)")
    results['exact_phi_exact_ratio'] = err_exact_ratio

    # Method 2: exact phi, numerical gradient
    phi_x_grad = np.gradient(phi_exact, dx)
    phi_safe = np.where(np.abs(phi_exact) < 1e-12, 1e-12, phi_exact)
    u_num_grad = -2.0 * nu * phi_x_grad / phi_safe
    err_numgrad = float(np.sqrt(np.sum((u_num_grad - u_exact)**2 * dx)))
    print(f"  Exact phi, numerical gradient -> L2={err_numgrad:.6e}")
    results['exact_phi_num_grad'] = err_numgrad

    # Method 3: exact phi, finite-difference variations
    for order, label in [(2, 'central_2nd'), (4, 'central_4th')]:
        if order == 2:
            phi_x_fd = np.gradient(phi_exact, dx)
        else:
            # 4th order central differences
            phi_x_fd = np.zeros_like(phi_exact)
            phi_x_fd[2:-2] = (-phi_exact[4:] + 8*phi_exact[3:-1]
                              - 8*phi_exact[1:-3] + phi_exact[:-4]) / (12.0 * dx)
            phi_x_fd[:2] = np.gradient(phi_exact, dx)[:2]
            phi_x_fd[-2:] = np.gradient(phi_exact, dx)[-2:]
        u_fd = -2.0 * nu * phi_x_fd / phi_safe
        err = float(np.sqrt(np.sum((u_fd - u_exact)**2 * dx)))
        print(f"  Exact phi, {label} -> L2={err:.6e}")
        results[f'exact_phi_{label}'] = err

    # Method 4: slight perturbation to phi (add noise)
    rng = np.random.default_rng(42)
    for noise_level in [1e-3, 1e-2, 1e-1]:
        phi_noisy = phi_exact + noise_level * rng.standard_normal(N_grid)
        phi_noisy = np.maximum(phi_noisy, 1e-10)
        phi_x_noisy = np.gradient(phi_noisy, dx)
        u_noisy = -2.0 * nu * phi_x_noisy / phi_noisy
        err = float(np.sqrt(np.sum((u_noisy - u_exact)**2 * dx)))
        print(f"  Exact phi + noise({noise_level:.0e}) -> L2={err:.4f}")
        results[f'perturbed_phi_noise_{noise_level:.0e}'] = err

    with open(_mk(OUT_BASE, 'study_B_deterministic.json'), 'w') as f:
        json.dump(results, f, indent=2)

    noise_levels = [1e-3, 1e-2, 1e-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x_grid, phi_exact, 'k-', lw=2, label='Exact phi')
    axes[0].plot(x_grid, phi_x_grad / phi_safe * (2.0 * nu) * (-1), 'r--', lw=1.5,
                 label='u from num-grad')
    axes[0].plot(x_grid, u_exact, 'b:', lw=1.5, label='Exact u')
    axes[0].set_title('Exact phi → u reconstruction')
    axes[0].set_xlabel('x'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    nl_arr = np.array(noise_levels)
    err_arr = np.array([results[f'perturbed_phi_noise_{nl:.0e}'] for nl in noise_levels])
    axes[1].loglog(nl_arr, err_arr, 'rs-', lw=1.5, ms=6)
    axes[1].set_xlabel('Phi perturbation level')
    axes[1].set_ylabel('L2 error in u')
    axes[1].set_title('Transform error vs phi perturbation')
    axes[1].grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'cole_hopf_deterministic_transform_error'))
    return results


def study_C_particle_phi_control(nu=0.5, A=1.0, T=0.5, dt=0.005, L=4.0,
                                  N_seq=None, S=5, base_seed=42):
    """Study C: particle-phi reconstruction control — standard N-particle runs
    vs 2N-particle runs interpolated back to the N grid.
    """
    print(f"\n{'='*60}\n  Study C: Particle-phi reconstruction control\n{'='*60}")
    if N_seq is None:
        N_seq = [200, 400, 800, 1600]
    xc = L / 2.0

    results = []

    for N in N_seq:
        l2_std_recon = []  # standard reconstruction (N output pts)
        l2_2N_recon  = []  # 2N output pts interpolated back

        for rep in range(S):
            seed = base_seed + rep
            try:
                x_out, u_out, _ = _run_cole_hopf(N, nu, T, dt, L, A, xc, seed)
                u_exact = exact_burgers_stationary_shock(x_out, nu, x_center=xc, amplitude=A)
                dx = float(x_out[1] - x_out[0])
                l2_std_recon.append(float(np.sqrt(np.sum((u_out - u_exact)**2 * dx)))                )
            except Exception as e:
                print(f"  WARNING study C N={N} std seed={seed}: {e}")

            # 2N output grid: run with 2N particles on same IC, compare at N grid
            try:
                x_out2, u_out2, _ = _run_cole_hopf(2*N, nu, T, dt, L, A, xc, seed+500)
                x_ref = np.linspace(0.0, L, N)
                u_interp = np.interp(x_ref, x_out2, u_out2)
                u_exact_ref = exact_burgers_stationary_shock(x_ref, nu, x_center=xc, amplitude=A)
                dx = float(x_ref[1] - x_ref[0])
                l2_2N_recon.append(float(np.sqrt(np.sum((u_interp - u_exact_ref)**2 * dx))))
            except Exception as e:
                print(f"  WARNING study C N={N} 2N seed={seed+500}: {e}")

        row = {
            'N': N,
            'l2_std_mean': float(np.mean(l2_std_recon)) if l2_std_recon else float('nan'),
            'l2_std_std':  float(np.std(l2_std_recon))  if l2_std_recon else float('nan'),
            'l2_2N_mean':  float(np.mean(l2_2N_recon))  if l2_2N_recon else float('nan'),
            'l2_2N_std':   float(np.std(l2_2N_recon))   if l2_2N_recon else float('nan'),
        }
        results.append(row)
        print(f"  N={N:5d}  L2_std={row['l2_std_mean']:.4f}  "
              f"L2_2N={row['l2_2N_mean']:.4f}")

    with open(_mk(OUT_BASE, 'study_C_particle_phi.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    n_arr = np.array([r['N'] for r in results], dtype=float)
    l2_std = np.array([r['l2_std_mean'] for r in results])
    l2_2N  = np.array([r['l2_2N_mean']  for r in results])
    ax.semilogx(n_arr, l2_std, 'bs-', lw=1.5, ms=6, label='Standard (N particles, N output)')
    ax.semilogx(n_arr, l2_2N,  'ro-', lw=1.5, ms=6, label='2N particles, N output')
    ax.set_xlabel('N'); ax.set_ylabel('L2 error')
    ax.set_title('Cole-Hopf: particle reconstruction comparison')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'cole_hopf_particle_vs_deterministic_phi'))
    return results


def study_D_grid_sensitivity(nu=0.5, A=1.0, T=0.5, dt=0.005, L=4.0,
                              N_particles=800, S=5, base_seed=42):
    """Study D: output/reference-grid sensitivity.

    The solver couples the output grid to the particle count, so N_out cannot
    be varied independently: each point runs with N = N_out particles.
    """
    print(f"\n{'='*60}\n  Study D: Output-grid sensitivity\n{'='*60}")
    xc = L / 2.0
    N_out_seq = [50, 100, 200, 400, 800, 1600]
    results = []

    for N_out in N_out_seq:
        l2_list = []
        for rep in range(S):
            seed = base_seed + rep
            try:
                x_out, u_out, _ = _run_cole_hopf(N_out, nu, T, dt, L, A, xc, seed)
                u_exact = exact_burgers_stationary_shock(x_out, nu, x_center=xc, amplitude=A)
                dx = float(x_out[1] - x_out[0])
                l2_list.append(float(np.sqrt(np.sum((u_out - u_exact)**2 * dx))))
            except Exception as e:
                print(f"  WARNING study D N_out={N_out} seed={seed}: {e}")

        row = {
            'N_out': N_out,
            'l2_mean': float(np.mean(l2_list)) if l2_list else float('nan'),
            'l2_std':  float(np.std(l2_list))  if l2_list else float('nan'),
        }
        results.append(row)
        print(f"  N_out={N_out:5d}  L2={row['l2_mean']:.4f}±{row['l2_std']:.4f}")

    with open(_mk(OUT_BASE, 'study_D_grid.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    nout_arr = np.array([r['N_out'] for r in results], dtype=float)
    l2_arr = np.array([r['l2_mean'] for r in results])
    l2_std = np.array([r['l2_std'] for r in results])
    ax.semilogx(nout_arr, l2_arr, 'go-', lw=1.5, ms=6)
    ax.fill_between(nout_arr, l2_arr - l2_std, l2_arr + l2_std, alpha=0.2, color='green')
    ax.set_xlabel('N (output grid & particles)')
    ax.set_ylabel('L2 error')
    ax.set_title('Cole-Hopf: error vs output grid resolution')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'cole_hopf_error_vs_output_grid'))
    return results


def run_task3(S=10, base_seed=42):
    """Run all four Cole-Hopf plateau diagnosis studies."""
    os.makedirs(OUT_BASE, exist_ok=True)

    print(f"\n{'='*60}\n  Task 3: Cole-Hopf plateau diagnosis\n{'='*60}")

    A_results = study_A_domain_sensitivity(S=S, base_seed=base_seed)
    B_results = study_B_deterministic_transform()
    C_results = study_C_particle_phi_control(S=S, base_seed=base_seed)
    D_results = study_D_grid_sensitivity(S=S, base_seed=base_seed)

    # Observed plateau: L2 ~ 0.40 at N=200-1600, L=4
    plateau_l2 = 0.40

    # From Study B: what is the error floor from transform differentiation alone?
    err_numgrad = B_results.get('exact_phi_num_grad', float('nan'))
    err_noise_1pct = B_results.get('perturbed_phi_noise_1e-02', float('nan'))

    # From Study A: error at larger domains (domain mismatch contribution)
    domain_l2 = {r['L']: r['l2_mean'] for r in A_results if r['mode'] == 'fixed_N'}

    if err_numgrad < 0.01:
        diff_cause = "transform_differentiation_negligible"
    elif err_numgrad > 0.2:
        diff_cause = "transform_differentiation_dominant"
    else:
        diff_cause = "transform_differentiation_moderate"

    domain_cause = "boundary_mismatch"
    if domain_l2:
        L_vals = sorted(domain_l2.keys())
        if len(L_vals) >= 2:
            trend = domain_l2[L_vals[-1]] - domain_l2[L_vals[0]]
            if trend < -0.1:
                domain_cause = "boundary_mismatch_significant"
            elif abs(trend) < 0.05:
                domain_cause = "boundary_mismatch_not_dominant"

    decomposition = {
        'plateau_l2_observed': plateau_l2,
        'transform_differentiation': {
            'error_exact_phi_num_grad': err_numgrad,
            'conclusion': diff_cause,
        },
        'boundary_mismatch': {
            'error_by_domain': domain_l2,
            'conclusion': domain_cause,
        },
        'particle_reconstruction': {
            'data': C_results,
        },
        'grid_resolution': {
            'data': D_results,
        },
    }

    if err_numgrad > 0.3 * plateau_l2:
        primary_cause = "transform_differentiation_plus_particle_noise"
    elif domain_cause == "boundary_mismatch_significant":
        primary_cause = "boundary_mismatch"
    else:
        primary_cause = "combination_particle_reconstruction_and_differentiation"

    decomposition['primary_cause'] = primary_cause
    print(f"\n  PRIMARY CAUSE: {primary_cause}")
    print(f"  Differentiation error (exact phi): {err_numgrad:.4f}")
    print(f"  1% phi noise error: {err_noise_1pct:.4f}")

    with open(_mk(OUT_BASE, 'plateau_decomposition.json'), 'w') as f:
        json.dump(decomposition, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sources = ['Diff. only\n(exact phi)', '1% phi noise', 'Observed\nplateau']
    values = [max(err_numgrad, 0), max(err_noise_1pct, 0), plateau_l2]
    colors = ['steelblue', 'orange', 'red']
    bars = axes[0].bar(sources, values, color=colors, alpha=0.8)
    axes[0].set_ylabel('L2 error')
    axes[0].set_title('Cole-Hopf plateau: error source decomposition')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', fontsize=9)
    if domain_l2:
        L_vals = sorted(domain_l2.keys())
        axes[1].plot(L_vals, [domain_l2[L] for L in L_vals], 'gs-', lw=1.5, ms=6)
        axes[1].axhline(plateau_l2, color='r', ls='--', label=f'Plateau ({plateau_l2})')
        axes[1].set_xlabel('Domain size L')
        axes[1].set_ylabel('L2 error')
        axes[1].set_title('Error vs domain size (fixed N=400)')
        axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, _mk(OUT_BASE, 'cole_hopf_plateau_decomposition'))

    print(f"\n  [Task 3] Done. Primary cause: {primary_cause}")
    return decomposition


if __name__ == '__main__':
    run_task3()
