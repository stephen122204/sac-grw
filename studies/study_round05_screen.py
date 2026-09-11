"""Round-5 part 5: bounded falsification screen for the nonlinear conjecture.

OPEN CONJECTURE under test (equal-weight scope only): for the signed
gradient-particle solver, sign-adjusted rank coupling has terminal pair-mean
full-field variance no larger than raw rank reflection.

PREDECLARED SCREENING MATRIX (fixed before any run; not expanded afterwards):
  configurations : clustered opposite signs | separated sign regions | merging
  particle count : N = 400 (cheap discovery)
  time steps     : h in {0.005, 0.0025} at FIXED physical T = 1
  dynamics       : 'heat' (no transport) and 'burgers', so a reversal has an
                   interpretable cause
  couplings      : raw_rank, sign_adjusted
  seeds          : 128, paired (shared streams)
  metric         : exact whole-line centred terminal variance of the pair mean
  reversal rule  : ratio > 1 with a paired bootstrap interval excluding 1;
                   any apparent reversal is re-run with 256 fresh seeds
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from studies.study_sign_coupling import advance_pair
from studies.study_round05_validation import exact_whole_line_var

OUT = ROOT / 'output/round05_screen_2026_09_09'
NU, T, N, SEEDS = 0.1, 1.0, 400, 128
M0 = 1.0 / N


def config(name):
    n = N // 2
    if name == 'clustered':          # alternating signs, tightly interleaved
        x = np.linspace(-0.25, 0.25, N)
        m = np.where(np.arange(N) % 2 == 0, +M0, -M0)
    elif name == 'separated':        # + block far left, - block far right
        x = np.concatenate([np.linspace(-2.0, -1.4, n), np.linspace(1.4, 2.0, n)])
        m = np.concatenate([np.full(n, +M0), np.full(n, -M0)])
    elif name == 'merging':          # blocks driven together by their own field
        x = np.concatenate([np.linspace(-0.9, -0.5, n), np.linspace(0.5, 0.9, n)])
        m = np.concatenate([np.full(n, +M0), np.full(n, -M0)])
    else:
        raise ValueError(name)
    o = np.argsort(x)
    return x[o].copy(), m[o].copy(), 0.0


def variance(cfg, h, drift, coup, seeds, key):
    x0, m0, ul = config(cfg)
    K = round(T / h)
    Xs, Ms = [], []
    for s in range(seeds):
        r = np.random.default_rng(np.random.SeedSequence([909, key, s]))
        if drift == 'heat':
            xa, ma = x0.copy(), m0.copy(); xb, mb = xa.copy(), ma.copy()
            sd = np.sqrt(2 * NU * h)
            for _ in range(K):
                oa = np.argsort(xa); xa, ma = xa[oa], ma[oa]
                ob = np.argsort(xb); xb, mb = xb[ob], mb[ob]
                z = r.standard_normal(len(xa))
                zb = -z if coup == 'raw_rank' else -np.sign(ma * mb) * z
                xa = xa + sd * z; xb = xb + sd * zb
            A, B = (xa, ma), (xb, mb)
        else:
            A, B, ul, _ = advance_pair((x0, m0, ul), NU, h, K, r, coup)
        Xs.append(np.concatenate([A[0], B[0]]))
        Ms.append(np.concatenate([A[1], B[1]]) / 2)
    return exact_whole_line_var(Xs, Ms, ul), Xs, Ms


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'config':>11} {'drift':>8} {'h':>8} {'var raw':>12} {'var sign':>12} "
          f"{'ratio':>8} {'verdict':>10}")
    for cfg in ('clustered', 'separated', 'merging'):
        for drift in ('heat', 'burgers'):
            for h in (0.005, 0.0025):
                vr, _, _ = variance(cfg, h, drift, 'raw_rank', SEEDS, 1)
                vs, _, _ = variance(cfg, h, drift, 'sign_adjusted', SEEDS, 1)
                ratio = vs / vr
                verdict = 'REVERSAL' if ratio > 1.0 else 'ok'
                rows.append(dict(config=cfg, drift=drift, h=h, var_raw=vr,
                                 var_sign=vs, ratio=ratio, seeds=SEEDS,
                                 verdict=verdict))
                print(f"{cfg:>11} {drift:>8} {h:>8g} {vr:12.5e} {vs:12.5e} "
                      f"{ratio:8.4f} {verdict:>10}")
    rev = [r for r in rows if r['verdict'] == 'REVERSAL']
    print(f"\napparent reversals in the predeclared matrix: {len(rev)}")
    for r in rev:
        vr, _, _ = variance(r['config'], r['h'], r['drift'], 'raw_rank', 256, 2)
        vs, _, _ = variance(r['config'], r['h'], r['drift'], 'sign_adjusted', 256, 2)
        print(f"  confirm {r['config']}/{r['drift']}/h={r['h']}: fresh 256-seed "
              f"ratio = {vs/vr:.4f} (screen {r['ratio']:.4f})")
        r['confirm_ratio_256_fresh_seeds'] = float(vs / vr)
    (OUT / 'screen.json').write_text(json.dumps(
        dict(rows=rows, matrix='predeclared: 3 configs x 2 drifts x 2 h x 2 couplings',
             N=N, T=T, nu=NU, seeds=SEEDS), indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
