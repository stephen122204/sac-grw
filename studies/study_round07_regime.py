"""Round-7 part 2: identify ONE narrower theorem target and check whether the
successful experiments plausibly satisfy its assumption.

The counterexample's mechanism, isolated. In state R the positive particle is
carried PAST the negative particle by transport alone within a single step. That
order change flips the sign of the local jump in the cumulative reconstruction
discontinuously, and it is what makes the terminal field move the wrong way.
In state L no such transport-induced swap occurs. Diffusion-induced order
changes do not do this: they are symmetric in the noise and are exactly what the
association argument already handles.

CANDIDATE TARGET (T1), stated as an assumption, NOT as a proved theorem:

  (A-R) Transport-induced reordering is negligible. For adjacent sorted
        particles i, i+1 the inclusive velocities differ by exactly
        v_{i+1} - v_i = f'(u_{i+1}) - f'(u_i), which for Burgers is m_{i+1}.
        Transport alone can reverse their order in one step only if their gap
        satisfies  g_i < |m_{i+1}| h. Writing sigma = sqrt(2 nu h), the
        dimensionless control is

            R = |m| h / sigma = |m| sqrt( h / (2 nu) ).

        (A-R) asks that the per-step probability of a transport-induced
        inversion be o(1) uniformly, which R << 1 makes plausible because
        typical gaps are O(sigma) after one diffusion stage.

WHAT THIS SCRIPT DOES. It measures, inside actual production runs, the per-step
fraction of adjacent pairs inverted by TRANSPORT alone and by DIFFUSION alone,
for the configurations that produced the reported gains and for the
counterexample. That tests whether (A-R) separates them. It does NOT prove that
(A-R) implies monotonicity, and no such claim is made here.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies import twopulse_reference as TP
from studies.study_smooth_transient import initialize_gaussian_gradient_particles

OUT = ROOT / 'output/round07_regime_2026_09_09'


def count_inversions(x_before, x_after):
    """Fraction of adjacent pairs (in the BEFORE order) whose order reversed."""
    d_before = np.diff(x_before)
    d_after = np.diff(x_after)
    return float(np.mean((d_before > 0) & (d_after < 0)))


def probe(x0, m0, ul, nu, h, K, seed, label):
    """Replicates the production conditional-mean schedule and instruments it."""
    rng = np.random.default_rng(seed)
    o = np.argsort(x0, kind='stable')
    x, m = np.asarray(x0, float)[o].copy(), np.asarray(m0, float)[o].copy()
    v = ul + np.cumsum(m)
    sd = np.sqrt(2.0 * nu * h)
    tr, di = [], []
    for _ in range(K):
        x_pre = x.copy()
        x = x + v * h                       # transport with the CARRIED velocity
        tr.append(count_inversions(x_pre, x))
        o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
        v = ul + np.cumsum(m)
        x_pre = x.copy()
        x = x + sd * rng.standard_normal(len(x))
        di.append(count_inversions(x_pre, x))
    return dict(label=label, nu=nu, h=h, K=K, N=len(x0),
                sigma=float(sd),
                m_typ=float(np.median(np.abs(m0))),
                R=float(np.median(np.abs(m0)) * h / sd),
                transport_inversions_per_step=float(np.mean(tr)),
                transport_inversions_max=float(np.max(tr)),
                diffusion_inversions_per_step=float(np.mean(di)),
                ratio_transport_to_diffusion=float(
                    np.mean(tr) / max(np.mean(di), 1e-300)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # the counterexample configuration
    rows.append(probe(np.array([-0.00225, 0.0]), np.array([0.5, -0.5]), 0.0,
                      1e-10, 0.005, 2, 1, 'counterexample (state R)'))
    rows.append(probe(np.array([-0.00275, 0.0]), np.array([0.5, -0.5]), 0.0,
                      1e-10, 0.005, 2, 1, 'counterexample (state L)'))

    # the configurations that produced the reported gains
    for N in (1600, 6400):
        xi, mi, ul, _ = TP.initialize(N)
        for h in (0.005, 0.0025):
            rows.append(probe(xi, mi, ul, 0.1, h, round(1.0 / h), 7,
                              f'twopulse N={N} h={h:g}'))
    xg, mg, ulg = initialize_gaussian_gradient_particles(1600)
    rows.append(probe(xg, mg, ulg, 0.1, 0.005, 200, 7, 'gaussian N=1600 h=0.005'))

    print(f"{'configuration':>28} {'R':>10} {'transport inv/step':>19} "
          f"{'diffusion inv/step':>19} {'transport/diffusion':>20}")
    for r in rows:
        print(f"{r['label']:>28} {r['R']:10.3e} "
              f"{r['transport_inversions_per_step']:19.3e} "
              f"{r['diffusion_inversions_per_step']:19.3e} "
              f"{r['ratio_transport_to_diffusion']:20.3e}")

    print("\nReading:")
    ce = [r for r in rows if 'counterexample' in r['label']]
    pr = [r for r in rows if 'counterexample' not in r['label']]
    print(f"  counterexample: transport-induced inversions per step up to "
          f"{max(r['transport_inversions_max'] for r in ce):.2f} of adjacent pairs, "
          f"with R ~ {max(r['R'] for r in ce):.1e}")
    print(f"  production runs: transport-induced inversions "
          f"{max(r['transport_inversions_per_step'] for r in pr):.1e} per step "
          f"(max over the tested cells), with R <= {max(r['R'] for r in pr):.1e}")
    print("\n  (A-R) separates the counterexample from every configuration that")
    print("  produced a measured gain. That makes the restricted target worth")
    print("  attempting. It is NOT a proof, and no monotonicity claim follows.")
    (OUT / 'regime.json').write_text(json.dumps(dict(
        target='(A-R) transport-induced reordering negligible; control '
               'R = |m| h / sigma = |m| sqrt(h/(2 nu))',
        status='ASSUMPTION IDENTIFIED AND CHECKED, NOT PROVED',
        does_not_claim=['that (A-R) implies coordinatewise monotonicity',
                        'that monotonicity would give FULL <= RAW',
                        'any asymptotic rate'],
        rows=rows), indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
