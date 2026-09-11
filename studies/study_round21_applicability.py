"""Round-21: lift the signed-structure applicability map to a production cell.

GAP CLOSED. The map of "where the coupling does nothing" came from the round-5
screen at N=400 -- a discovery instrument -- while every other result in the paper
is at N=2048-8192. A central claim should not rest on a coarser configuration
than the rest of the paper. Same four structures, same two dynamics, at N=2048
with 48 replicates and whole-replicate bootstrap intervals.

Structures (definitions unchanged from the round-5 screen, rescaled to N):
  clustered  alternating signs, tightly interleaved
  separated  + block far left, - block far right
  merging    blocks driven together by their own field
  one_sign   all gradient masses of one sign (the pathwise-identity control)
Dynamics: heat (no transport; isolates the coupling) and burgers.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import reconstruct_cumulative_field
from studies.study_round06_cubic import advance_pair_flux, XG
from studies.round06_spectral_reference import FPRIME

OUT = ROOT / 'output/round21_applicability_2026_09_10'
NU, T, H, N, REPS = 0.1, 1.0, 0.005, 2048, 48


def config(name, N):
    n = N // 2; M0 = 1.0 / N
    if name == 'clustered':
        x = np.linspace(-0.25, 0.25, N); m = np.where(np.arange(N) % 2 == 0, M0, -M0)
    elif name == 'separated':
        x = np.concatenate([np.linspace(-2.0, -1.4, n), np.linspace(1.4, 2.0, n)])
        m = np.concatenate([np.full(n, M0), np.full(n, -M0)])
    elif name == 'merging':
        x = np.concatenate([np.linspace(-0.9, -0.5, n), np.linspace(0.5, 0.9, n)])
        m = np.concatenate([np.full(n, M0), np.full(n, -M0)])
    elif name == 'one_sign':
        x = np.linspace(-1.0, 1.0, N); m = np.full(N, -2.0 / N)
    o = np.argsort(x)
    return x[o].copy(), m[o].copy(), 0.0


def run(cfg, drift, arm, key):
    x0, m0, ul = config(cfg, N); K = round(T / H)
    fp = FPRIME['burgers'] if drift == 'burgers' else (lambda u: np.zeros_like(u))
    P = []
    for r in range(REPS):
        s = np.random.default_rng(np.random.SeedSequence([key, r]))
        if drift == 'burgers':
            A, B, u = advance_pair_flux((x0, m0, ul), NU, H, K, s, arm, 'burgers')
        else:
            A, B, u = advance_pair_flux((x0, m0, ul), NU, H, K, s, arm, 'heat0')
        P.append(0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                        + reconstruct_cumulative_field(B[0], B[1], u, XG)))
    return np.array(P)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FPRIME['heat0'] = lambda u: np.zeros_like(u)
    dx = float(XG[1] - XG[0]); rng = np.random.default_rng(2101); t0 = time.perf_counter()
    print(f"Signed-structure applicability at a PRODUCTION cell "
          f"(N={N}, h={H}, {REPS} replicates)\n")
    print(f"   {'structure':>11} {'dynamics':>9} {'V_switch/V_reflect':>19} "
          f"{'95% CI (bootstrap)':>22} {'verdict':>12}")
    rows = []
    for cfg in ('clustered', 'merging', 'separated', 'one_sign'):
        for drift in ('heat', 'burgers'):
            R = run(cfg, drift, 'RAW', 2100)
            F = run(cfg, drift, 'FULL', 2110)
            vR = float(dx * np.sum(R.var(0, ddof=1)))
            vF = float(dx * np.sum(F.var(0, ddof=1)))
            bs = []
            for _ in range(2000):
                i = rng.integers(0, REPS, REPS); j = rng.integers(0, REPS, REPS)
                bs.append(dx * np.sum(F[i].var(0, ddof=1))
                          / (dx * np.sum(R[j].var(0, ddof=1))))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            pathwise = float(np.max(np.abs(R[0] - F[0])))
            v = ('pathwise identical' if pathwise == 0.0 else
                 'help' if hi < 1 else ('HARM' if lo > 1 else 'unresolved'))
            print(f"   {cfg:>11} {drift:>9} {vF/vR:19.4f} [{lo:.4f},{hi:.4f}]"
                  f" {v:>12}")
            rows.append(dict(structure=cfg, dynamics=drift, ratio=vF / vR,
                             ci=[float(lo), float(hi)], verdict=v,
                             pathwise_max_diff=pathwise))
    json.dump(dict(N=N, h=H, nu=NU, T=T, reps=REPS, rows=rows,
                   note='replaces the N=400 round-5 screen for the applicability '
                        'map; whole-replicate percentile bootstrap intervals'),
              open(OUT / 'applicability.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter()-t0:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
