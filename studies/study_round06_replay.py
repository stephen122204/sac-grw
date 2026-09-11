"""Round-6 part 1: evidence-gap repairs for the round-5 archive.

Three defects reported by the orchestrator, all confirmed by inspection of
`studies/study_round05_validation.py`:

  (D1) The archive stores XA/XB per seed per coupling but only ONE mass vector
       per problem (`MA[0]`, from whichever coupling ran last) and no MB at all.
       Signed rank order is path dependent, so that vector cannot reconstruct
       the archived fields. Separate partner fields fA, fB are also absent.
  (D2) The two-sided pathwise witness REIMPLEMENTS the paired update and then
       compares SORTED UNWEIGHTED positions. It never calls the production
       stepper and never checks the mass-position association or the signed
       reconstructed field, so it cannot certify that the paired driver is the
       production update.
  (D3) `warnings.filterwarnings('ignore')`; recorded per-seed wall time also
       contains the reconstruction estimator and the audit-only disagreement
       scan; timings run grouped by coupling.

This driver repairs all three WITHOUT touching the production solver, the
round-4/round-5 drivers, or the old archive. It writes a NEW versioned archive.

Repair of D1 -- deterministic replay. `advance_pair` is a pure function of
(initial state, nu, h, K, generator seed, coupling), so replaying the recorded
SeedSequence keys reproduces every terminal state exactly. The replay is
VERIFIED bit-for-bit against the archived XA/XB/pairmean before its recovered
MA/MB are trusted.

Repair of D2 -- production-driven two-sided witness. Under sign-adjusted
coupling arm A's increment is sd*Z, which does not depend on arm B. So the pair
can be reproduced by two SEQUENTIAL production runs:

    pass A: advance_rbgbmc_particles(conditional_mean_transport=True) with a
            noise adapter that draws Z and records (Z_k, sign of the arm's own
            post-sort masses m_k) read from the live production frame;
    pass B: the same production entry point with an adapter returning
            -sd * sign(mA_k) * sign(m_k^B) * Z_k, again reading m^B from the
            live production frame.

The adapters are read-only: they inspect production's local state and supply
normals, and never write back. Both partners are then compared with the paired
driver ELEMENTWISE in position AND mass (so the association is tested, not just
the point set) and through the signed reconstructed field.

Repair of D3 -- warnings are recorded rather than suppressed; replay timing
excludes the estimator and the disagreement scan and interleaves couplings.
Old timings are copied through unchanged under an explicit round-5 key and are
NOT comparable with the new ones.
"""
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relaxation_gbmc import advance_rbgbmc_particles, reconstruct_cumulative_field
from studies.study_sign_coupling import advance_pair, setup, PAR, PK, COUPLINGS
from studies import twopulse_reference as TP

OLD = ROOT / 'output/round05_validation_2026_09_09/archive.npz'
OUT = ROOT / 'output/round06_replay_2026_09_09'
SEEDS, KEY = 64, 505          # exactly the round-5 configuration, replayed


# --------------------------------------------------------------------------
# D2: production-driven two-sided witness
# --------------------------------------------------------------------------
def _live_masses():
    """Post-sort particle masses of the production stepper that is calling us.

    Walks the stack to the `advance_rbgbmc_particles` frame. Read-only: nothing
    is written back into production state.
    """
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_name == 'advance_rbgbmc_particles' and 'm_p' in f.f_locals:
            return np.asarray(f.f_locals['m_p'], float)
        f = f.f_back
    raise RuntimeError('production frame not found; adapter used out of context')


class RecordA:
    """Arm-A noise adapter: draws sd*Z and records Z with the live mass signs."""

    def __init__(self, seed, expect_sd):
        self.g = np.random.default_rng(seed)
        self.expect_sd = expect_sd
        self.Z, self.SA = [], []

    def normal(self, loc, scale, size):
        assert loc == 0.0 and abs(scale - self.expect_sd) < 1e-15, (loc, scale)
        z = self.g.standard_normal(size)
        self.Z.append(z.copy())
        self.SA.append(np.sign(_live_masses()))
        return scale * z


class ReplayB:
    """Arm-B noise adapter: -sd * s_A * s_B * Z with s_B read live from arm B."""

    def __init__(self, rec, expect_sd):
        self.rec, self.k = rec, 0
        self.expect_sd = expect_sd
        self.SB = []

    def normal(self, loc, scale, size):
        assert loc == 0.0 and abs(scale - self.expect_sd) < 1e-15, (loc, scale)
        sB = np.sign(_live_masses())
        self.SB.append(sB)
        z = self.rec.Z[self.k]
        sA = self.rec.SA[self.k]
        self.k += 1
        return -scale * (sA * sB) * z


def witness_production_two_sided(N=200, K=40, nu=0.1, a=2.0, h=0.005, seed=11):
    """Both partners of the sign-adjusted pair, reproduced by PRODUCTION code."""
    x0, m0, ul, _ = TP.initialize(N)
    sd = float(np.sqrt(2.0 * nu * h))

    ref_A, ref_B, ul_ref, dis = advance_pair((x0, m0, ul), nu, h, K,
                                             np.random.default_rng(seed),
                                             'sign_adjusted', record_disagree=True)

    dummy = np.random.default_rng(0)          # unused: cond-mean draws no uniforms
    recA = RecordA(seed, sd)
    outA = advance_rbgbmc_particles(x0, m0, ul, nu, a, h, K, dummy,
                                    rng_brownian=recA,
                                    conditional_mean_transport=True)
    repB = ReplayB(recA, sd)
    outB = advance_rbgbmc_particles(x0, m0, ul, nu, a, h, K,
                                    np.random.default_rng(0), rng_brownian=repB,
                                    conditional_mean_transport=True)

    xg = np.linspace(-5.0, 5.0, 400)
    res = {}
    for tag, out, ref in (('A', outA, ref_A), ('B', outB, ref_B)):
        dx = float(np.max(np.abs(out['x'] - ref[0])))          # ELEMENTWISE
        dm = float(np.max(np.abs(out['m'] - ref[1])))          # association
        fp = reconstruct_cumulative_field(out['x'], out['m'], ul, xg)
        fr = reconstruct_cumulative_field(ref[0], ref[1], ul_ref, xg)
        df = float(np.max(np.abs(fp - fr)))                    # SIGNED field
        res[tag] = dict(max_abs_dx=dx, max_abs_dm=dm, max_abs_dfield=df)
    res['sign_disagreement_mean'] = float(dis.mean())
    res['sign_disagreement_max'] = float(dis.max())
    res['steps_with_disagreement'] = int(np.sum(dis > 0))
    res['partner_sign_differences'] = int(sum(
        int(np.sum(sa != sb)) for sa, sb in zip(recA.SA, repB.SB)))
    res['config'] = dict(N=N, K=K, nu=nu, a=a, h=h, seed=seed,
                         profile='twopulse (mixed sign)',
                         driver='advance_rbgbmc_particles + read-only noise adapter')
    return res


# --------------------------------------------------------------------------
# D1: deterministic replay recovering per-seed, per-coupling MA and MB
# --------------------------------------------------------------------------
def main(seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    caught = []
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')

        wit = witness_production_two_sided()
        print('production-driven two-sided witness (twopulse, mixed sign, '
              f"disagreement mean {wit['sign_disagreement_mean']:.3f} "
              f"max {wit['sign_disagreement_max']:.3f}, "
              f"{wit['partner_sign_differences']} partner sign differences)")
        for tag in ('A', 'B'):
            r = wit[tag]
            print(f"  partner {tag}: max|dx|={r['max_abs_dx']:.1e}  "
                  f"max|dm|={r['max_abs_dm']:.1e}  "
                  f"max|d field|={r['max_abs_dfield']:.1e}")
        print()

        old = np.load(OLD)
        arch, rows, checks = {}, [], []
        for problem, N in (('gaussian', 1600), ('twopulse', 1600)):
            p = PAR[problem]
            x_out, ref, init, _ = setup(problem, N)
            dx = float(x_out[1] - x_out[0])
            K = round(p['T'] / p['h'])
            print(f'{problem} N={N}: deterministic replay of key {KEY}, {seeds} seeds')
            print('-' * 92)

            store = {c: dict(XA=[], MA=[], XB=[], MB=[], FA=[], FB=[], PM=[])
                     for c in COUPLINGS}
            # interleave couplings within each seed so timing order is not
            # grouped; the replay clock excludes estimator and audit work
            secs = {c: [] for c in COUPLINGS}
            for s in range(seeds):
                for coup in COUPLINGS:
                    r = np.random.default_rng(
                        np.random.SeedSequence([PK[problem], N, KEY, s]))
                    t0 = time.perf_counter()
                    A, B, ul, _ = advance_pair(init, p['nu'], p['h'], K, r, coup,
                                               record_disagree=False)
                    secs[coup].append(time.perf_counter() - t0)   # solver only
                    fa = reconstruct_cumulative_field(A[0], A[1], ul, x_out)
                    fb = reconstruct_cumulative_field(B[0], B[1], ul, x_out)
                    d = store[coup]
                    d['XA'].append(A[0]); d['MA'].append(A[1])
                    d['XB'].append(B[0]); d['MB'].append(B[1])
                    d['FA'].append(fa); d['FB'].append(fb)
                    d['PM'].append((fa + fb) / 2)

            for coup in COUPLINGS:
                d = {k: np.asarray(v) for k, v in store[coup].items()}
                # verification against the OLD archive, bit-for-bit
                chk = {}
                for k in ('XA', 'XB'):
                    o = old[f'{problem}_{N}_{coup}_{k}']
                    chk[k] = float(np.max(np.abs(d[k] - o)))
                chk['pairmean'] = float(np.max(np.abs(
                    d['PM'] - old[f'{problem}_{N}_{coup}_pairmean'])))
                ok = max(chk.values()) == 0.0
                checks.append(dict(problem=problem, N=N, coupling=coup,
                                   bitwise_identical=ok, **chk))
                print(f"  {coup:>14} replay vs round-5 archive: "
                      f"max|dXA|={chk['XA']:.1e} max|dXB|={chk['XB']:.1e} "
                      f"max|d pairmean|={chk['pairmean']:.1e}  "
                      f"{'IDENTICAL' if ok else 'MISMATCH'}")
                if not ok:
                    raise RuntimeError(f'replay mismatch for {problem}/{coup}: {chk}')

                base = f'{problem}_{N}_{coup}'
                for k in ('XA', 'MA', 'XB', 'MB', 'FA', 'FB'):
                    arch[f'{base}_{k}'] = d[k]
                arch[f'{base}_pairmean'] = d['PM']
                arch[f'{base}_secs_round05'] = old[f'{base}_secs']       # preserved
                arch[f'{base}_disagree_round05'] = old[f'{base}_disagree']
                arch[f'{base}_secs_replay_solver_only'] = np.asarray(secs[coup])
                rows.append(dict(
                    problem=problem, N=N, coupling=coup, seeds=seeds,
                    grid_var=float(dx * np.sum(d['PM'].var(axis=0, ddof=1))),
                    mse=float(np.mean(dx * np.sum((d['PM'] - ref) ** 2, axis=1))),
                    secs_round05_mean=float(np.mean(old[f'{base}_secs'])),
                    secs_replay_solver_only_mean=float(np.mean(secs[coup])),
                    timing_note='round-5 and replay timings are NOT comparable: '
                                'round-5 included the estimator and the audit scan'))
            arch[f'{problem}_{N}_x'] = x_out
            arch[f'{problem}_{N}_ref'] = ref
            arch[f'{problem}_{N}_init_x'] = np.asarray(init[0])
            arch[f'{problem}_{N}_init_m'] = np.asarray(init[1])
            arch[f'{problem}_{N}_u_left'] = np.asarray([float(init[2])])
            print()
        caught = [f'{w.category.__name__}: {w.message}' for w in wlog]

    np.savez_compressed(OUT / 'archive_v2.npz', **arch)
    with (OUT / 'rows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]
    (OUT / 'replay.json').write_text(json.dumps(dict(
        purpose='round-6 repair of round-5 archive defects D1/D2/D3',
        seeds=seeds, seed_key=KEY,
        seed_scheme='np.random.SeedSequence([PK[problem], N, 505, seed_index])',
        PK=PK, params=PAR, couplings=COUPLINGS,
        replay_verification=checks,
        production_witness=wit,
        warnings_recorded=caught,
        old_archive=str(OLD.relative_to(ROOT)),
        new_archive='output/round06_replay_2026_09_09/archive_v2.npz',
        recovered_fields=['MA', 'MB', 'FA', 'FB'],
        preserved_fields=['secs_round05', 'disagree_round05'],
        hashes=dict(solver=sha('relaxation_gbmc.py'),
                    coupler=sha('studies/study_sign_coupling.py'),
                    twopulse=sha('studies/twopulse_reference.py'),
                    round05_driver=sha('studies/study_round05_validation.py'),
                    this_driver=hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest()[:16]),
        rows=rows), indent=1))
    print(f'warnings recorded during replay: {len(caught)}')
    for c in caught[:10]:
        print(f'  {c}')
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
