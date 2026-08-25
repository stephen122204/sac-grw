"""reproduce.py — entry points for reproducing the Paper 2 (RB-GBMC) studies.

Thin wrapper: each target invokes the corresponding checked-in study script
with the paper configuration. `verify` reruns nothing; it checks the
checked-in summaries against expected_values.json.

Usage:
    python reproduce.py <target>

Targets: t6 ta adt ablation t1 t2 multinu multinu-scaled pilot transient
         all studies studies-all figures verify   (see --help)
  all      legacy: the original four studies (T6, TA, T1, T2)
  studies  every manuscript study (t6 ta adt ablation t1 t2 multinu
           multinu-scaled pilot transient); studies-all is an alias
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Fixed date for matplotlib PDF metadata so regenerated figures are byte-identical
# across runs (matplotlib honors SOURCE_DATE_EPOCH). Subprocesses inherit this.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1704067200')  # 2024-01-01 UTC

TARGETS = {
    't1': ('studies/study_t1_gbmc_dt_bias.py',
           'T1  RB-GBMC dt-bias at N=6400 (5 dt values, S=40)  '
           '-> time-step refinement table'),
    't2': ('studies/run_t2_S30.py',
           'T2  traveling-shock validation at S=30 (PAPER config; the S=10 '
           'default inside run_task2() is an exploration setting)  '
           '-> traveling-shock table + figures'),
    't6': ('studies/study_gbmc_production_n_refinement.py',
           'T6  production RB-GBMC N-refinement (N=100..6400, S=50)  '
           '-> stationary refinement + recovered-viscosity tables + figures'),
    'ta': ('studies/study_relaxation_speed_sensitivity.py',
           'TA  relaxation-speed sensitivity (a=1.5,2,3,4; three N; S=50)  '
           '-> parameter-sensitivity table'),
    'adt': ('studies/study_a_dt_interaction.py',
            'ADT joint relaxation-speed x time-step interaction '
            '(a=1.5,2,3,4 x dt=0.01..0.000625 at N=6400, M=400, S=50; '
            'a=2,dt=0.0025 cross-checks T6 at N=6400)  -> interaction table'),
    'ablation': ('studies/study_conditional_mean_ablation.py',
                 'internal conditional-mean transport control (attribution '
                 'ablation): two-speed a=2/a=4 vs V=u control, split streams, '
                 'dt in {0.01,0.0025,0.000625}, N=6400, M=400, S=50'),
    'multinu': ('studies/study_multiviscosity_sweep.py',
                'multi-viscosity sweep (nu in {0.5,0.25,0.1,0.05,0.025} x '
                '{two-speed a=2,a=4, cond-mean control}, N=6400, dt=0.0025, S=50): '
                'paired label excess and control residual as nu falls at FIXED dt'),
    'multinu-scaled': ('studies/study_multiviscosity_scaled_dt.py',
                       'scaled-time-step companion (dt = 0.005*nu, so dt/nu is '
                       'fixed while nu falls): separates viscosity dependence '
                       'from temporal resolution; nu=0.5 row cross-checks the '
                       'fixed-step sweep bit-for-bit'),
    'pilot': ('studies/study_ordering_pilot.py',
              'ordering pilot: carried-label vs redraw-after-diffusion schedule '
              'at nu in {0.1,0.05,0.025}, dt in {0.0025,0.00125}; attributes '
              'the low-nu control residual to carried-velocity timing plus an '
              'O(dt) remainder (both refinement-vanishing)'),
    'transient': ('studies/study_smooth_transient.py',
                  'smooth nonstationary transient (Gaussian hump, nu=0.1, T=1, '
                  'Cole-Hopf quadrature reference with documented tolerance): '
                  'paired L2 excess over the control across dt and a'),
    'all': (None,
            'legacy target: run the original four studies T6, TA, T1, T2.'),
    'studies': (None,
                'run every manuscript study: T6, TA, ADT, ablation, T1, T2, '
                'multinu, multinu-scaled, pilot, transient.'),
    'studies-all': (None,
                    'alias of `studies` (kept for compatibility).'),
    'figures': ('figure_scripts/regenerate_paper_figures.py',
                'regenerate the nine title-less Paper 2 figures from checked-in '
                'data (no rerun) -> output/final_prepublication_tests/paper2_figures/'),
}

EPILOG = """\
targets:
""" + "\n".join(f"  {k:<7} {v[1]}" for k, v in TARGETS.items()) + """
  verify  NO rerun: check the checked-in summaries under
          output/final_prepublication_tests/ against
          expected_values.json for every archived study
          (T6, TA, ADT, ablation, T1, T2, multinu, multinu-scaled,
          pilot, transient).

notes:
  * Full reruns require scipy (tanh fits) and take roughly 10-20 minutes
    per principal study on a laptop. Without scipy the studies fall back to a cruder
    zero-crossing fit for the recovered shock parameters (xc, nu); the
    E_bias/E_spread/E_total error columns do not depend on the fit.
  * Function defaults inside the study modules are exploration settings.
    The paper configurations are exactly what these targets invoke
    (in particular, the T2 paper run is S=30 via studies/run_t2_S30.py).
  * Rerunning t6, ta, t1, or t2 overwrites that study's subdirectory under
    output/final_prepublication_tests/ (t2 also emits a nu=0.2 sharp-layer
    exploration figure there that is not used in the paper). Reruns are
    seed-deterministic: on the archived environment (requirements-lock.txt),
    every non-runtime field is reproduced bit-for-bit in our checks.
  * Seeds: base seed 42, consecutive per ensemble member. The same seed
    identifiers are reused at each N for reproducibility; this is not a strict
    common-random-number coupling across N.
  * multinu and ablation are resumable by cell and validate a configuration
    fingerprint stored in their manifest.json: an incompatible S, N, dt, T,
    reconstruction count, viscosity list, arm list, window design, fit-bound
    rule, or seed scheme refuses to resume instead of mixing designs. multinu
    seeds are keyed by the viscosity VALUE (SeedSequence([base, round(nu*1e6),
    seed_idx])), so `--nu` subsets and reorderings reproduce identical cells,
    and a subset invocation never shrinks a complete summary. A viscosity
    outside the canonical list is refused (edit NU_SEQ = a new study).
"""


def run_target(name):
    script, desc = TARGETS[name]
    print(f"[reproduce] {name}: {desc}")
    print(f"[reproduce] invoking {script}")
    return subprocess.run([sys.executable, os.path.join(ROOT, script)],
                          cwd=ROOT).returncode


# ---------------------------------------------------------------- verify -- #

def _json_path(obj, path):
    """Resolve a dotted path ('fit.bias_slope', 'per_dt.0.E_bias') in nested JSON."""
    cur = obj
    for part in path.split('.'):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def _close(a, b, rtol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b))
    return a == b


def _check_json_paths(chk):
    with open(os.path.join(ROOT, chk['file'])) as f:
        obj = json.load(f)
    fails = []
    for path, expect in chk['values'].items():
        got = _json_path(obj, path)
        if not _close(got, expect, chk['rtol']):
            fails.append(f"{path}: expected {expect}, got {got}")
    return fails


def _check_json_select(chk):
    with open(os.path.join(ROOT, chk['file'])) as f:
        obj = json.load(f)
    lst = _json_path(obj, chk['list_path']) if chk['list_path'] else obj
    rows = [r for r in lst
            if all(_close(r.get(k), v, 1e-12) for k, v in chk['where'].items())]
    if len(rows) != 1:
        return [f"where={chk['where']}: matched {len(rows)} rows (want 1)"]
    fails = []
    for col, expect in chk['expect'].items():
        got = rows[0].get(col)
        if not _close(got, expect, chk['rtol']):
            fails.append(f"{col}: expected {expect}, got {got}")
    return fails


def _check_csv_rows(chk):
    with open(os.path.join(ROOT, chk['file'])) as f:
        data = {r[chk['key']]: r for r in csv.DictReader(f)}
    fails = []
    for key, cols in chk['rows'].items():
        if key not in data:
            fails.append(f"{chk['key']}={key}: row missing")
            continue
        for col, expect in cols.items():
            got = float(data[key][col])
            if not _close(got, expect, chk['rtol']):
                fails.append(f"{chk['key']}={key} {col}: expected {expect}, got {got}")
    return fails


def _check_csv_columns_identical(chk):
    def load(p):
        with open(os.path.join(ROOT, p)) as f:
            return list(csv.DictReader(f))
    a, b = load(chk['file']), load(chk['file_b'])
    if len(a) != len(b):
        return [f"row count {len(a)} != {len(b)}"]
    fails = []
    for col in chk['columns']:
        for i, (ra, rb) in enumerate(zip(a, b)):
            if ra[col] != rb[col]:
                fails.append(f"row {i} col {col}: {ra[col]!r} != {rb[col]!r}")
    return fails


CHECKERS = {
    'json_paths': _check_json_paths,
    'json_select': _check_json_select,
    'csv_rows': _check_csv_rows,
    'csv_columns_identical': _check_csv_columns_identical,
}


def verify():
    with open(os.path.join(ROOT, 'expected_values.json')) as f:
        spec = json.load(f)
    n_pass = n_fail = 0
    for chk in spec['checks']:
        try:
            fails = CHECKERS[chk['type']](chk)
        except Exception as e:
            fails = [f"{type(e).__name__}: {e}"]
        if fails:
            n_fail += 1
            print(f"FAIL  [{chk.get('study', '?'):5s}] {chk['id']}")
            for msg in fails:
                print(f"      - {msg}")
        else:
            n_pass += 1
            print(f"PASS  [{chk.get('study', '?'):5s}] {chk['id']}")
    print(f"\nverify (Paper 2): {n_pass} passed, {n_fail} failed, "
          f"{n_pass + n_fail} total")
    return 1 if n_fail else 0


def main():
    parser = argparse.ArgumentParser(
        prog='reproduce.py',
        description='Reproduce the Paper 2 (RB-GBMC) studies, or verify the '
                    'checked-in results without rerunning anything.',
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', choices=list(TARGETS) + ['verify'])
    args = parser.parse_args()
    if args.target == 'verify':
        sys.exit(verify())
    if args.target == 'all':
        rc = 0
        for t in ('t6', 'ta', 't1', 't2'):
            rc = run_target(t) or rc
        sys.exit(rc)
    if args.target in ('studies', 'studies-all'):
        rc = 0
        for t in ('t6', 'ta', 'adt', 'ablation', 't1', 't2', 'multinu',
                  'multinu-scaled', 'pilot', 'transient'):
            rc = run_target(t) or rc
        sys.exit(rc)
    sys.exit(run_target(args.target))


if __name__ == '__main__':
    main()
