"""Compares a coupled replica with the single-simulation update on the same draws (Section 3.3)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import advance_particles, reconstruct_cumulative_field
from studies.work_target_validation import advance_single_flux
from studies.cubic_flux_transfer import XG
from studies import twopulse_reference as TP

OUT = ROOT / 'output/single_replica_law_check'
VAL = ROOT / 'output/work_target_validation/validation.json'


def stat_repair(v):
    """Approximate pointwise inference from the 32 scalar block errors."""
    tol = v['tol']
    E = {a: np.array(d['per_block']) for a, d in v['realised'].items()}
    n = len(E['FULL'])
    print("1. STATISTICAL REPAIR  (32 disjoint blocks per arm; blocks of different")
    print("   arms are unrelated, so differences are UNPAIRED)\n")
    print(f"   {'arm':>7} {'mean':>12} {'95% t interval':>26} {'vs tol':>28}")
    rows = {}
    for a in ('SINGLE', 'RAW', 'FULL'):
        m = E[a].mean(); se = E[a].std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        lo, hi = m - t * se, m + t * se
        verdict = ('interval below tol' if hi < tol else
                   'interval straddles tol' if lo < tol else 'interval above tol')
        rows[a] = dict(mean=float(m), se=float(se), ci=[float(lo), float(hi)],
                       verdict=verdict)
        print(f"   {a:>7} {m:12.4e} [{lo:.3e}, {hi:.3e}] {verdict:>28}")
    print(f"\n   target tol = {tol:.1e}\n")
    print(f"   {'comparison':>18} {'difference':>13} {'Welch 95% interval':>26} {'verdict':>12}")
    diffs = {}
    for a, b in (('FULL', 'SINGLE'), ('FULL', 'RAW'), ('SINGLE', 'RAW')):
        d = E[a].mean() - E[b].mean()
        se = np.sqrt(E[a].var(ddof=1) / n + E[b].var(ddof=1) / n)
        df = (E[a].var(ddof=1) / n + E[b].var(ddof=1) / n) ** 2 / (
            (E[a].var(ddof=1) / n) ** 2 / (n - 1) + (E[b].var(ddof=1) / n) ** 2 / (n - 1))
        t = stats.t.ppf(0.975, df)
        lo, hi = d - t * se, d + t * se
        res = 'resolved' if (hi < 0 or lo > 0) else 'UNRESOLVED'
        diffs[f'{a}-{b}'] = dict(diff=float(d), ci=[float(lo), float(hi)], verdict=res)
        print(f"   {a+' - '+b:>18} {d:13.4e} [{lo:+.3e}, {hi:+.3e}] {res:>12}")
    print("\n   Reading: FULL's mean interval lies below the target, narrowly.")
    print("   FULL vs SINGLE in expected error is UNRESOLVED. FULL vs the executed")
    print("   RAW plan is resolved. These are pointwise approximate intervals, not a")
    print("   simultaneous or high-probability guarantee, and the per-block spread")
    print("   (blocks below tol: " + ", ".join(
        f"{a} {v['realised'][a]['blocks_below_tol']}/32" for a in ('SINGLE','RAW','FULL'))
        + ") supports no probability-of-success claim.")
    print("\n   Cost ratios describe THESE executed plans. They are not bounds on the")
    print("   minimum work each method needs at a common accuracy, which was never measured.")
    return rows, diffs


def repro_audit():
    print("\n2. REPRODUCIBILITY AUDIT\n")
    v = json.load(open(VAL))
    print(f"   saved validation keys: {sorted(v.keys())}")
    print(f"   PYTHONHASHSEED recorded in the archive: "
          f"{'yes' if 'PYTHONHASHSEED' in json.dumps(v) else 'NO'}")
    print(f"   PYTHONHASHSEED in the current environment: "
          f"{os.environ.get('PYTHONHASHSEED', '(unset)')}")
    outs = []
    for _ in range(3):
        r = subprocess.run([sys.executable, '-c', "print(hash('SINGLE'))"],
                           capture_output=True, text=True)
        outs.append(r.stdout.strip())
    print(f"   hash('SINGLE') across three fresh interpreters: {outs}")
    stable = len(set(outs)) == 1
    print(f"   -> hash() is {'stable' if stable else 'PROCESS-DEPENDENT'}; the driver used")
    print(f"      SeedSequence([EVAL_KEY, hash(arm) % 9973, b, r]), so the saved key")
    print(f"      alone {'can' if stable else 'CANNOT'} reproduce those streams.")
    print("   The archive is NOT overwritten and no seeds are invented. The round-13")
    print("   result stands as an executed held-out evaluation whose exact streams")
    print("   cannot be replayed; the experiment can be repeated with stable IDs.")
    return dict(hash_stable=bool(stable), hash_samples=outs,
                pythonhashseed_recorded=False,
                pythonhashseed_env=os.environ.get('PYTHONHASHSEED'))


ARM_ID = {'SINGLE': 1, 'RAW': 2, 'FULL': 3}       # stable integer identifiers


def stable_seed(eval_key, arm, block, rep):
    return np.random.SeedSequence([eval_key, ARM_ID[arm], block, rep])


def fixtures():
    print("\n3. DETERMINISTIC FIXTURES (tiny; no solver batch)\n")
    a = [stable_seed(1313, 'FULL', 2, 5).generate_state(2).tolist() for _ in range(3)]
    r = subprocess.run(
        [sys.executable, '-c',
         "import numpy as np;print(np.random.SeedSequence([1313,3,2,5])"
         ".generate_state(2).tolist())"], capture_output=True, text=True)
    print(f"   stable_seed same process, 3 calls: {a}")
    print(f"   fresh interpreter:                 {r.stdout.strip()}")
    ok = len(set(map(str, a))) == 1 and str(a[0]) == r.stdout.strip()
    print(f"   -> deterministic across processes: {ok}")

    # single-arm path vs production, SAME prescribed noise, FULL state compared
    N, K, nu, h, a_rel = 64, 25, 0.1, 0.005, 2.0
    x0, m0, ul, _ = TP.initialize(N)
    Z = np.random.default_rng(4242).standard_normal((K, N))
    sd = np.sqrt(2 * nu * h)

    class Supply:
        def __init__(s): s.k = 0
        def normal(s, loc, scale, size):
            z = Z[s.k][:size]; s.k += 1; return scale * z

    class SupplyStd:
        def __init__(s): s.k = 0
        def standard_normal(s, n):
            z = Z[s.k][:n]; s.k += 1; return z

    prod = advance_particles(x0, m0, ul, nu, a_rel, h, K,
                                    np.random.default_rng(0), rng_brownian=Supply(),
                                    conditional_mean_transport=True)
    xs, ms, uls = advance_single_flux((x0, m0, ul), nu, h, K, SupplyStd(), 'burgers')
    dx = float(np.max(np.abs(prod['x'] - xs)))
    dm = float(np.max(np.abs(prod['m'] - ms)))
    fp = reconstruct_cumulative_field(prod['x'], prod['m'], ul, XG)
    fs = reconstruct_cumulative_field(xs, ms, uls, XG)
    df = float(np.max(np.abs(fp - fs)))
    print(f"\n   single-arm path vs production under the SAME prescribed noise:")
    print(f"     max|dx| = {dx:.1e}   max|dm| = {dm:.1e}   max|d field| = {df:.1e}")
    print(f"     (elementwise positions AND masses AND the signed field, not a")
    print(f"      distributional comparison)   {'PASS' if dx==0 and dm==0 and df==0 else 'FAIL'}")
    assert dx == 0.0 and dm == 0.0 and df == 0.0
    return dict(seed_deterministic=bool(ok), single_vs_production=dict(
        max_abs_dx=dx, max_abs_dm=dm, max_abs_dfield=df))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v = json.load(open(VAL))
    rows, diffs = stat_repair(v)
    rep = repro_audit()
    fix = fixtures()
    json.dump(dict(inference=rows, differences=diffs,
                   inference_note='pointwise approximate t intervals on 32 scalar '
                                  'block errors; unpaired across arms; not '
                                  'simultaneous and not a probability guarantee',
                   reproducibility=rep, fixtures=fix, arm_ids=ARM_ID),
              open(OUT / 'audit.json', 'w'), indent=1)
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
