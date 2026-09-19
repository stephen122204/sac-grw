"""Numerical trace of the two-particle monotonicity counterexample (Section 3.6, Appendix B)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import advance_particles, reconstruct_cumulative_field

OUT = ROOT / 'output/monotonicity_counterexample'
UL, A_REL, H, NU, K = 0.0, 2.0, 0.005, 1e-10, 2
M = np.array([+0.5, -0.5])
PROBE = 0.00125
STATES = {'L': -0.00275, 'R': -0.00225}


class ZeroNoise:
    def normal(self, loc, scale, size):
        return np.zeros(size)


def field(out, probe=PROBE):
    return float(reconstruct_cumulative_field(out['x'], out['m'], UL,
                                              np.array([probe]))[0])


def trace_by_hand(xp):
    """Independent re-derivation of the two production steps, by hand."""
    x = np.array([xp, 0.0]); m = M.copy()
    o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
    v = UL + np.cumsum(m)                      # conditional-mean transport
    log = [('init', x.copy(), m.copy(), v.copy())]
    for s in range(K):
        x = x + v * H
        o = np.argsort(x, kind='stable'); x, m, v = x[o], m[o], v[o]
        v = UL + np.cumsum(m)
        log.append((f'step{s+1}', x.copy(), m.copy(), v.copy()))
    u = UL + np.sum(m[x <= PROBE])
    return x, m, float(u), log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print("CHECK 1 - zero-noise: hand trace vs the production solver\n")
    for tag, xp in STATES.items():
        hx, hm, hu, log = trace_by_hand(xp)
        out = advance_particles(np.array([xp, 0.0]), M, UL, NU, A_REL, H, K,
                                       np.random.default_rng(0),
                                       rng_brownian=ZeroNoise(),
                                       conditional_mean_transport=True)
        pu = field(out)
        print(f"  state {tag} (positive particle at {xp:+.5f}):")
        for name, x, m, v in log:
            print(f"      {name:>6}  x={np.array2string(x, precision=5, floatmode='fixed')}"
                  f"  m={np.array2string(m, precision=1, floatmode='fixed')}"
                  f"  v={np.array2string(v, precision=1, floatmode='fixed')}")
        print(f"      production terminal x={np.array2string(out['x'], precision=5, floatmode='fixed')}"
              f"  m={np.array2string(out['m'], precision=1, floatmode='fixed')}")
        print(f"      hand U(probe)={hu:+.3f}   production U(probe)={pu:+.3f}   "
              f"{'MATCH' if hu == pu else 'MISMATCH'}\n")
        assert hu == pu and np.max(np.abs(hx - out['x'])) < 1e-15
        res[tag] = dict(x_pos_init=xp, terminal_x=out['x'].tolist(),
                        terminal_m=out['m'].tolist(), U_probe=pu)

    dU = res['R']['U_probe'] - res['L']['U_probe']
    print(f"  moving the POSITIVE particle right by {STATES['R']-STATES['L']:+.5f} "
          f"changes U(probe) by {dU:+.3f}")
    print(f"  predicted by the monotonicity claim: NEGATIVE (orientation -sign(m)=-1)")
    print(f"  observed: {'POSITIVE -> claim violated' if dU > 0 else 'negative'}\n")

    print("CHECK 2 - production solver with sigma=1e-6, 512 seeds per state\n")
    sigma = float(np.sqrt(2 * NU * H))
    vals = {}
    for tag, xp in STATES.items():
        u = []
        for s in range(512):
            out = advance_particles(
                np.array([xp, 0.0]), M, UL, NU, A_REL, H, K,
                np.random.default_rng(0),
                rng_brownian=np.random.default_rng(np.random.SeedSequence([707, s])),
                conditional_mean_transport=True)
            u.append(field(out))
        vals[tag] = np.array(u)
        print(f"  state {tag}: U(probe) unique values = {np.unique(vals[tag])}, "
              f"mean = {vals[tag].mean():+.6f}")
    print(f"  sigma = {sigma:.1e};  E[U_R] - E[U_L] = "
          f"{vals['R'].mean()-vals['L'].mean():+.4f}\n")

    print("CHECK 3 - margins behind the Gaussian-tail bound\n")
    margins = {}
    for tag, xp in STATES.items():
        _, _, _, log = trace_by_hand(xp)
        gap1 = float(np.abs(np.diff(log[1][1])[0]))       # post-transport-1 gap
        term = log[2][1]
        dprobe = float(np.min(np.abs(term - PROBE)))      # probe classification
        margins[tag] = dict(order_gap_after_step1=gap1, probe_margin=dprobe)
        print(f"  state {tag}: ordering gap after step 1 = {gap1:.2e}; "
              f"nearest terminal particle to probe = {dprobe:.2e}")
    crit = min(min(m['order_gap_after_step1'] / 2.0, m['probe_margin'] / 2.0)
               for m in margins.values())
    print(f"\n  smallest half-margin = {crit:.2e}")
    delta = 1e-4
    print(f"  The displacement cap delta = {delta:.0e}  -> "
          f"{'SAFE (cap < every half-margin)' if delta < crit else 'NOT SAFE'}")
    n_disp = 4
    logeps = float(np.log(2 * n_disp) + norm.logcdf(-delta / sigma))
    print(f"  P(any of {n_disp} displacements exceeds {delta:.0e}) "
          f"<= 2*{n_disp}*Phi(-{delta/sigma:.0f}) = exp({logeps:.1f}) "
          f"(log10 = {logeps/np.log(10):.0f})")
    print(f"  hence E[U_R] - E[U_L] >= 0.5 - 1.5*eps > 0  ->  CLAIM REFUTED\n")

    # regime parameter separating this example from the production experiments
    print("CHECK 4 - where this configuration sits relative to the experiments\n")
    print("  Single-step reordering of two adjacent particles is transport-driven")
    print("  when the velocity gap |m| times h exceeds the diffusive smear sigma:")
    print("      R := |m| h / sigma = |m| sqrt(h / (2 nu)).")
    rows = [('counterexample', 0.5, H, NU),
            ('twopulse N=1600 h=0.005', 2.3 / 1600, 0.005, 0.1),
            ('twopulse N=1600 h=0.0025', 2.3 / 1600, 0.0025, 0.1),
            ('gaussian N=1600 h=0.005', 1.6 / 1600, 0.005, 0.1),
            ('twopulse N=6400 h=0.005', 2.3 / 6400, 0.005, 0.1)]
    print(f"  {'configuration':>26} {'|m|':>10} {'sigma':>10} {'R':>10}")
    reg = []
    for name, m, h, nu in rows:
        sg = np.sqrt(2 * nu * h)
        R = m * h / sg
        reg.append(dict(configuration=name, m=m, h=h, nu=nu, sigma=float(sg), R=float(R)))
        print(f"  {name:>26} {m:10.3e} {sg:10.3e} {R:10.3e}")
    print(f"\n  R spans {reg[0]['R']/reg[1]['R']:.1e} between the counterexample and "
          f"the production runs.")

    (OUT / 'counterexample.json').write_text(json.dumps(dict(
        claim='coordinatewise monotonicity of E[U(x,T)|state] with orientation -sign(m_i)',
        verdict='REFUTED without resolution restrictions, for this discrete solver',
        config=dict(u_left=UL, a=A_REL, h=H, nu=NU, steps=K, m=M.tolist(),
                    probe=PROBE, states=STATES, sigma=sigma),
        zero_noise=res, delta_U=dU,
        production_512_seeds={k: dict(unique=np.unique(v).tolist(),
                                      mean=float(v.mean())) for k, v in vals.items()},
        margins=margins, displacement_cap=delta,
        log_exceptional_probability=logeps,
        regime=reg,
        scope='two particles, strongly transport-dominated (R ~ 2.5e3). It does '
              'NOT show FULL worse than RAW, does not invalidate the measured '
              'gains at nu=0.1 with large N (R ~ 1e-4), and does not exclude a '
              'theorem carrying explicit resolution assumptions.'), indent=1))
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
