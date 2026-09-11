"""Round-7 part 3: accuracy-versus-work under ONE shared bias, with calibrated
squared-bias inference, plus the two refinement cells from archived data.

THREE ROUND-6 DEFECTS, CORRECTED HERE.

(E1) The matched-work model used RAW's own noisy squared-bias estimate in the
     denominator and the sign-adjusted one in the numerator, although the
     marginal-law argument says the bias is COMMON. That imports sampling noise
     into a quantity the model treats as shared. Corrected: one explicitly
     chosen shared estimate per (problem, N, h) configuration -- the FULL
     (sign-adjusted) arm, chosen because it has the smallest V_pair and hence
     the sharpest estimate -- resampled JOINTLY with every V_pair so their
     correlation is preserved. Per-method estimates are kept as diagnostics and
     used to test the shared-bias premise.

(E2) The percentile interval for b2_hat = ||mean_S - ref||^2 - V_hat/S is
     miscalibrated. In the bootstrap world the parameter is
     theta* = ||mean_S - ref||^2, so E*[T*] ~ theta* = T + V_hat/S: the
     bootstrap law sits a distance V_hat/S ABOVE the statistic. A naive
     percentile interval inherits that shift. Corrected with the basic
     (pivotal) bootstrap, CI = [T + theta* - q*_{97.5}, T + theta* - q*_{2.5}],
     and the size of the shift is reported so the reader can see whether it
     mattered. Both intervals are printed.

(E3) The round-6 report said the low-variance coupling resolves the common bias
     "roughly 30x more tightly". The printed widths do not support that
     uniformly. The measured width ratios are reported here instead.

Fractional ensemble sizes B are MODEL INTERPOLATION of b2 + V/B at equal wall
clock, not executed runs. No claim is made that bias is zero.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'output/round07_common_bias_2026_09_09'
BOOT = 4000
BS = (1, 2, 4, 8)
SHARED_ARM = 'FULL'          # explicitly chosen: smallest V_pair -> sharpest b2


def load_cells():
    """Every cell is (label, {policy: PM}, ref, dx, {policy: cost}, N, h)."""
    cells = {}
    z5 = np.load(ROOT / 'output/round05_validation_2026_09_09/archive.npz')
    zr = np.load(ROOT / 'output/round06_replay_2026_09_09/archive_v2.npz')
    name = {'RAW': 'raw_rank', 'FULL': 'sign_adjusted', 'WITHIN': 'within_sign',
            'INDEP': 'independent'}
    for prob in ('gaussian', 'twopulse'):
        x = zr[f'{prob}_1600_x']
        cells[f'{prob} N=1600 h=0.005'] = dict(
            PM={p: zr[f'{prob}_1600_{c}_pairmean'] for p, c in name.items()},
            ref=zr[f'{prob}_1600_ref'], dx=float(x[1] - x[0]),
            cost={p: float(np.mean(z5[f'{prob}_1600_{c}_secs']))
                  for p, c in name.items()}, N=1600, h=0.005, problem=prob)
    # refinement cell A: N 1600 -> 6400 at h=0.005 (round-4 archive, key 51)
    zs = np.load(ROOT / 'output/sign_coupling_2026_09_09/fields.npz')
    import csv as _csv
    cost4 = {}
    for r in _csv.DictReader(open(ROOT / 'output/sign_coupling_2026_09_09/sign_coupling.csv')):
        cost4[(r['problem'], int(r['N']), r['coupling'])] = float(r['cost_s'])
    for prob in ('gaussian', 'twopulse'):
        x = zs[f'{prob}_6400_x']
        cells[f'{prob} N=6400 h=0.005'] = dict(
            PM={p: zs[f'{prob}_6400_{c}'] for p, c in name.items()},
            ref=zs[f'{prob}_6400_ref'], dx=float(x[1] - x[0]),
            cost={p: cost4[(prob, 6400, c)] for p, c in name.items()},
            N=6400, h=0.005, problem=prob)
        cells[f'{prob} N=1600 h=0.005 (key51)'] = dict(
            PM={p: zs[f'{prob}_1600_{c}'] for p, c in name.items()},
            ref=zs[f'{prob}_1600_ref'], dx=float(x[1] - x[0]),
            cost={p: cost4[(prob, 1600, c)] for p, c in name.items()},
            N=1600, h=0.005, problem=prob)
    # refinement cell B: h 0.005 -> 0.0025 at N=1600 (round-6 cubic, burgers arm)
    zc = np.load(ROOT / 'output/round06_cubic_2026_09_09/cubic.npz')
    costc = {}
    for r in _csv.DictReader(open(ROOT / 'output/round06_cubic_2026_09_09/rows.csv')):
        costc[(r['flux'], float(r['h']), r['seed_key'], r['policy'])] = float(r['cost_s'])
    xg = zc['x_grid']
    for h in (0.005, 0.0025):
        cells[f'twopulse N=1600 h={h:g} (spectral ref)'] = dict(
            PM={p: zc[f'burgers_{h:g}_1_{p}_pairmean'] for p in
                ('RAW', 'FULL', 'WITHIN')},
            ref=zc['reference_burgers'], dx=float(xg[1] - xg[0]),
            cost={p: costc[('burgers', h, '1', p)] for p in
                  ('RAW', 'FULL', 'WITHIN')}, N=1600, h=h, problem='twopulse')
    return cells


def stats(PM, ref, dx, idx=None):
    """(T, theta_plugin, V) where T is the unbiased squared-bias statistic."""
    P = PM if idx is None else PM[idx]
    S = P.shape[0]
    V = float(dx * np.sum(P.var(axis=0, ddof=1)))
    theta = float(dx * np.sum((P.mean(0) - ref) ** 2))
    return theta - V / S, theta, V


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    rng = np.random.default_rng(7071)
    rows, summary = [], {}
    for label, c in cells.items():
        pols = list(c['PM'])
        S = c['PM'][pols[0]].shape[0]
        idx = rng.integers(0, S, (BOOT, S))
        pt, bs = {}, {}
        for p in pols:
            T, th, V = stats(c['PM'][p], c['ref'], c['dx'])
            pt[p] = dict(T=T, theta=th, V=V)
            bs[p] = np.array([stats(c['PM'][p], c['ref'], c['dx'], i)[:3:2]
                              for i in idx])          # columns: T*, V*
        # ---- E2: calibrated interval for the SHARED bias
        Tsh, thsh = pt[SHARED_ARM]['T'], pt[SHARED_ARM]['theta']
        Tstar = bs[SHARED_ARM][:, 0]
        naive = np.percentile(Tstar, [2.5, 97.5])
        pivotal = [Tsh + thsh - np.percentile(Tstar, 97.5),
                   Tsh + thsh - np.percentile(Tstar, 2.5)]
        shift = pt[SHARED_ARM]['V'] / S
        print(f"\n{label}   (S={S})")
        print("-" * 100)
        print(f"  shared bias arm = {SHARED_ARM}:  b2_hat = {Tsh:.4e}")
        print(f"     naive percentile CI  [{naive[0]:.4e}, {naive[1]:.4e}]"
              f"   (bootstrap law sits +V/S = {shift:.3e} above the statistic)")
        print(f"     pivotal (basic) CI   [{pivotal[0]:.4e}, {pivotal[1]:.4e}]"
              f"   <- used below")
        # ---- shared-bias premise check, per-method diagnostics
        print(f"  per-method diagnostics (premise: all estimate the SAME b2)")
        widths = {}
        for p in pols:
            ts = bs[p][:, 0]
            lo = pt[p]['T'] + pt[p]['theta'] - np.percentile(ts, 97.5)
            hi = pt[p]['T'] + pt[p]['theta'] - np.percentile(ts, 2.5)
            widths[p] = hi - lo
            ok = not (hi < pivotal[0] or lo > pivotal[1])
            print(f"     {p:>6}: b2_hat={pt[p]['T']:+.4e}  CI[{lo:+.4e},{hi:+.4e}]"
                  f"  V={pt[p]['V']:.4e}  {'consistent' if ok else 'INCONSISTENT'}")
        wr = widths['RAW'] / widths[SHARED_ARM]
        print(f"     interval-width ratio RAW / {SHARED_ARM} = {wr:.1f}x"
              f"   (round 6 said ~30x; that was not supported)")
        # ---- E1: matched-work model with ONE shared bias, jointly resampled
        theta_draws = 2 * Tsh - Tstar          # pivotal draws for the shared b2
        neg = float(np.mean(theta_draws < 0))
        print(f"  matched-work model  (shared b2; fractional B is INTERPOLATION "
              f"of b2 + V/B, not executed runs)")
        for B in BS:
            w = B * c['cost'][ 'RAW']
            eR = Tsh + pt['RAW']['V'] / B
            Bf = w / c['cost'][SHARED_ARM]
            eF = Tsh + pt[SHARED_ARM]['V'] / Bf
            rr = ((theta_draws + bs[SHARED_ARM][:, 1] / Bf) /
                  (theta_draws + bs['RAW'][:, 1] / B))
            lo, hi = np.percentile(rr, [2.5, 97.5])
            print(f"     work {w:6.3f}s (RAW B={B}, {SHARED_ARM} B={Bf:4.1f}): "
                  f"{eF:.3e} / {eR:.3e} = ratio {eF/eR:.3f} [{lo:.3f},{hi:.3f}]")
            rows.append(dict(cell=label, problem=c['problem'], N=c['N'], h=c['h'],
                             B_raw=B, work_s=w, B_full_interp=Bf,
                             err_raw=eR, err_full=eF, ratio=eF / eR,
                             ratio_ci=[float(lo), float(hi)],
                             shared_b2=Tsh, shared_b2_ci=pivotal))
        Bstar = {p: pt[p]['V'] / Tsh for p in pols}
        print(f"  B* = V/b2 under the shared bias: " +
              "  ".join(f"{p}={Bstar[p]:.1f}" for p in pols) +
              "   (method-specific, not a universal ceiling)")
        summary[label] = dict(N=c['N'], h=c['h'], problem=c['problem'], S=S,
                              shared_b2=Tsh, shared_b2_ci_pivotal=pivotal,
                              shared_b2_ci_naive=naive.tolist(),
                              calibration_shift=shift,
                              V={p: pt[p]['V'] for p in pols},
                              cost={p: c['cost'][p] for p in pols},
                              Bstar={p: float(Bstar[p]) for p in pols},
                              width_ratio_RAW_over_shared=float(wr),
                              negative_b2_draw_fraction=neg,
                              per_method_b2={p: pt[p]['T'] for p in pols})

    # ---- refinement: how does the shared bias move?
    print("\n\nREFINEMENT (archived cells; no new runs)")
    print("=" * 100)
    pairs = [('twopulse N=1600 h=0.005 (key51)', 'twopulse N=6400 h=0.005',
              'N: 1600 -> 6400 at h=0.005'),
             ('gaussian N=1600 h=0.005 (key51)', 'gaussian N=6400 h=0.005',
              'N: 1600 -> 6400 at h=0.005'),
             ('twopulse N=1600 h=0.005 (spectral ref)',
              'twopulse N=1600 h=0.0025 (spectral ref)',
              'h: 0.005 -> 0.0025 at N=1600')]
    ref_rows = []
    for a, b, what in pairs:
        A, B_ = summary[a], summary[b]
        print(f"\n  {A['problem']}  {what}")
        print(f"    shared b2 : {A['shared_b2']:.4e} {A['shared_b2_ci_pivotal']}")
        print(f"             -> {B_['shared_b2']:.4e} {B_['shared_b2_ci_pivotal']}")
        print(f"    V_FULL    : {A['V'][SHARED_ARM]:.4e} -> {B_['V'][SHARED_ARM]:.4e}")
        print(f"    B*_FULL   : {A['Bstar'][SHARED_ARM]:.1f} -> {B_['Bstar'][SHARED_ARM]:.1f}")
        sep = (B_['shared_b2_ci_pivotal'][1] < A['shared_b2_ci_pivotal'][0]) or \
              (A['shared_b2_ci_pivotal'][1] < B_['shared_b2_ci_pivotal'][0])
        print(f"    change in b2 resolved by the intervals: "
              f"{'YES' if sep else 'NO - overlapping, direction only'}")
        ref_rows.append(dict(comparison=what, problem=A['problem'], coarse=a,
                             fine=b, b2_coarse=A['shared_b2'],
                             b2_fine=B_['shared_b2'],
                             ci_coarse=A['shared_b2_ci_pivotal'],
                             ci_fine=B_['shared_b2_ci_pivotal'],
                             Bstar_coarse=A['Bstar'][SHARED_ARM],
                             Bstar_fine=B_['Bstar'][SHARED_ARM],
                             resolved=bool(sep)))
    with (OUT / 'matched_work.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'common_bias.json').write_text(json.dumps(dict(
        corrections=['E1 one shared bias, jointly resampled with every V_pair',
                     'E2 pivotal bootstrap for squared bias; shift reported',
                     'E3 measured interval-width ratios replace the ~30x claim'],
        shared_arm=SHARED_ARM,
        shared_arm_rationale='smallest V_pair, hence sharpest estimate of the '
                             'common bias; the choice is stated, not fitted',
        fractional_B='model interpolation of b2 + V/B at equal wall clock, '
                     'not executed ensembles',
        bootstrap=BOOT, cells=summary, matched_work=rows,
        refinement=ref_rows,
        caveat='two refinement cells indicate a direction; they do not '
               'establish an asymptotic rate or separate N/h interactions'),
        indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
