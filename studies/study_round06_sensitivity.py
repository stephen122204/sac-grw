"""Round-6 part 1d: fully archived single-particle sensitivity experiment.

Round 5 reported a 3.5% wrong-sign fraction for Burgers and used it to argue
that separable monotonicity fails. That number came from an unarchived scratch
run: no script, no states, no per-seed curves, no uncertainty, no multiplicity
control, no displacement-size sweep. It is rebuilt here from scratch with all
of that recorded, and it is reported as a magnitude, not only as a fraction of
grid points.

QUESTION. For the signed gradient-particle update, is the map

    configuration  ->  E[ U(x,T) ]                        (terminal field)

monotone in each particle's position with orientation set by that particle's
mass sign? Displacing particle i0 from p to p+d removes m_i0 from the cumulative
sum on [p, p+d), so the PREDICTED response sign is -sign(m_i0 * d) at every x.
A response of the opposite sign at any x contradicts that sufficient condition.

WHAT IS MEASURED. A FINITE-DISPLACEMENT response, not a derivative:

    R_d(x) = E[U_pert(x,T)] - E[U_base(x,T)],      X_i0 -> X_i0 + d.

R_d/d approaches the directional derivative as d -> 0 only if the map is
differentiable; the sweep over d is what separates a genuine sensitivity from a
finite-displacement artefact. Both runs share noise BY PARTICLE IDENTITY, which
is a coupling choice only: the noise is i.i.d. across particles, so each run
keeps its exact marginal law while the paired difference has small variance.

RESOLVED WRONG-SIGN POINT. x is a resolved wrong-sign point when

    sign(m_i0) * R_d(x) > z * se(x),      se(x) = sd of paired per-seed
                                          differences / sqrt(S),

with z the normal quantile at level 0.05 BONFERRONI-corrected over all grid
points tested, so the reported count is not a multiple-comparisons artefact.
Magnitudes are reported alongside: the peak wrong-sign response relative to the
peak response, and the integrated wrong-sign mass relative to the integrated
absolute response.

CONTROLS. Heat with an exact analytic response (sign must be exact), heat by
Monte Carlo (same estimator, no interaction), then Burgers. Every configuration
is repeated on an independent seed key for fresh confirmation.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field

OUT = ROOT / 'output/round06_sensitivity_2026_09_09'
NU, T, H, NPART = 0.1, 0.5, 0.005, 60
DISPLACEMENTS = (0.01, 0.025, 0.05, 0.1)
S_SEEDS = 3000
XG = np.linspace(-3.0, 3.0, 241)
ALPHA = 0.05


def base_state(seed=7):
    """Mixed-sign configuration; deterministic, archived in full."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.5, 1.5, NPART))
    m = np.where(np.arange(NPART) % 2 == 0, 1.0, -1.0) / NPART
    m = m - m.mean()                      # exact zero net mass
    return x, m, 0.0


def pick_particle(x, m):
    """Displaced particle: the one nearest the configuration centroid, so the
    perturbation sits where the interaction is strongest. Fixed before any run."""
    return int(np.argmin(np.abs(x - np.average(x, weights=np.abs(m)))))


def evolve(x0, m0, ul, drift, key, seeds, ids):
    """S paired realizations; noise indexed BY IDENTITY so base and perturbed
    runs share increments particle-by-particle. Returns per-seed fields."""
    K = round(T / H)
    sd = np.sqrt(2.0 * NU * H)
    F = np.empty((seeds, len(XG)))
    for s in range(seeds):
        r = np.random.default_rng(np.random.SeedSequence([6062, key, s]))
        x, m, idx = x0.copy(), m0.copy(), ids.copy()
        for _ in range(K):
            if drift == 'burgers':
                o = np.argsort(x, kind='stable')
                x, m, idx = x[o], m[o], idx[o]
                x = x + (ul + np.cumsum(m)) * H
            z = r.standard_normal(NPART)          # drawn in IDENTITY order
            x = x + sd * z[idx]
        F[s] = reconstruct_cumulative_field(x, m, ul, XG)
    return F


def analytic_heat(x0, m0, ul):
    s = np.sqrt(2.0 * NU * T)
    return ul + np.sum(m0[:, None] * norm.cdf((XG[None, :] - x0[:, None]) / s), axis=0)


def run(drift, d, key, seeds=S_SEEDS):
    x0, m0, ul = base_state()
    i0 = pick_particle(x0, m0)
    xp = x0.copy(); xp[i0] += d
    ids = np.arange(NPART)
    if drift == 'heat_analytic':
        R = analytic_heat(xp, m0, ul) - analytic_heat(x0, m0, ul)
        se = np.zeros_like(R)
        per_seed = None
    else:
        kind = 'heat' if drift == 'heat_mc' else 'burgers'
        Fb = evolve(x0, m0, ul, kind, key, seeds, ids)
        Fp = evolve(xp, m0, ul, kind, key, seeds, ids)
        D = Fp - Fb                                   # paired per-seed curves
        R = D.mean(axis=0)
        se = D.std(axis=0, ddof=1) / np.sqrt(seeds)
        per_seed = D
    sgn = np.sign(m0[i0])
    z = norm.ppf(1.0 - ALPHA / (2 * len(XG)))          # Bonferroni over the grid
    resolved = np.abs(R) > z * np.maximum(se, 0.0) if per_seed is not None \
        else np.abs(R) > 0.0
    wrong = (sgn * R > z * se) if per_seed is not None else (sgn * R > 0.0)
    dxg = float(XG[1] - XG[0])
    denom = float(np.sum(np.abs(R)) * dxg)
    return dict(
        drift=drift, displacement=d, seed_key=key, seeds=seeds,
        displaced_particle=i0, displaced_from=float(x0[i0]),
        displaced_mass=float(m0[i0]),
        n_grid=len(XG), n_resolved=int(resolved.sum()),
        n_wrong_sign=int(wrong.sum()),
        wrong_fraction_of_resolved=float(wrong.sum() / max(resolved.sum(), 1)),
        peak_abs_response=float(np.max(np.abs(R))),
        peak_wrong_sign_response=float(np.max(sgn * R)) if np.any(wrong) else 0.0,
        peak_wrong_relative=float(np.max(sgn * R) / np.max(np.abs(R)))
        if np.any(wrong) else 0.0,
        integrated_wrong_mass=float(np.sum(np.maximum(sgn * R, 0.0)) * dxg),
        integrated_wrong_relative=float(
            np.sum(np.maximum(sgn * R, 0.0)) * dxg / denom) if denom > 0 else 0.0,
        bonferroni_z=float(z),
        max_se=float(se.max()),
    ), R, se, per_seed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x0, m0, ul = base_state()
    i0 = pick_particle(x0, m0)
    print(f"configuration: N={NPART} mixed sign, T={T}, h={H}, nu={NU}, "
          f"S={S_SEEDS} paired seeds")
    print(f"displaced particle: index {i0} at x={x0[i0]:+.4f}, "
          f"m={m0[i0]:+.5f} (sign {int(np.sign(m0[i0])):+d})")
    print(f"predicted response sign everywhere: {-int(np.sign(m0[i0])):+d}")
    print(f"resolved wrong-sign test: Bonferroni over {len(XG)} grid points "
          f"at alpha={ALPHA}\n")
    rows, arch = [], {}
    print(f"{'drift':>15} {'d':>7} {'key':>4} {'resolved':>9} {'wrong':>6} "
          f"{'frac':>7} {'peak|R|':>10} {'peak wrong':>11} {'int wrong rel':>14}")
    t0 = time.perf_counter()
    for drift in ('heat_analytic', 'heat_mc', 'burgers'):
        for d in DISPLACEMENTS:
            for key in (1, 2):                        # 2 = fresh confirmation
                if drift == 'heat_analytic' and key == 2:
                    continue
                row, R, se, D = run(drift, d, key)
                rows.append(row)
                tag = f"{drift}_d{d:g}_k{key}"
                arch[f'{tag}_R'] = R
                arch[f'{tag}_se'] = se
                if D is not None:
                    arch[f'{tag}_per_seed_diff'] = D.astype(np.float32)
                print(f"{drift:>15} {d:7g} {key:>4} {row['n_resolved']:9d} "
                      f"{row['n_wrong_sign']:6d} "
                      f"{row['wrong_fraction_of_resolved']:7.4f} "
                      f"{row['peak_abs_response']:10.3e} "
                      f"{row['peak_wrong_sign_response']:11.3e} "
                      f"{row['integrated_wrong_relative']:14.3e}")
    arch['x_grid'] = XG
    arch['init_x'] = x0
    arch['init_m'] = m0
    arch['init_ids'] = np.arange(NPART)
    arch['init_velocity_burgers'] = ul + np.cumsum(m0)     # carried velocities
    arch['displaced_index'] = np.array([i0])
    for d in DISPLACEMENTS:
        xp = x0.copy(); xp[i0] += d
        arch[f'perturbed_x_d{d:g}'] = xp
        arch[f'perturbed_velocity_burgers_d{d:g}'] = ul + np.cumsum(m0)
    np.savez_compressed(OUT / 'sensitivity.npz', **arch)
    (OUT / 'sensitivity.json').write_text(json.dumps(dict(
        purpose='archived replacement for the unarchived round-5 sensitivity run',
        question='is E[U(x,T)] monotone in each particle position with '
                 'orientation -sign(m_i)?',
        measured='FINITE-DISPLACEMENT response R_d(x), not a derivative',
        coupling='common random numbers indexed by particle identity; a '
                 'coupling choice only, each run keeps its exact marginal law',
        resolved_definition='sign(m_i0)*R(x) > z*se(x), z at alpha=0.05 '
                            'Bonferroni-corrected over all grid points',
        nu=NU, T=T, h=H, N=NPART, seeds=S_SEEDS, alpha=ALPHA,
        displacements=list(DISPLACEMENTS),
        seed_scheme='np.random.SeedSequence([6062, key, seed_index]); '
                    'key 1 = primary, key 2 = fresh confirmation',
        displaced_particle=dict(index=i0, x=float(x0[i0]), m=float(m0[i0]),
                                rule='nearest the |m|-weighted centroid, fixed '
                                     'before any run'),
        rows=rows, runtime_s=time.perf_counter() - t0), indent=1))
    print(f"\nelapsed {time.perf_counter()-t0:.1f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
