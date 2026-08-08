"""reproduce.py — entry points for reproducing the Paper 2 (RB-GBMC) studies.

Thin wrapper: each target invokes the corresponding checked-in study script
with the paper configuration. `verify` reruns nothing; it checks the
checked-in summaries against expected_values.json.

Usage:
    python reproduce.py <target>

Targets: t1 t2 t3 t4 t5 t6 all verify   (see --help)
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
           'T1  GBMC dt-bias at N=6400 (5 dt values, S=40)  -> Table 5'),
    't2': ('studies/run_t2_S30.py',
           'T2  traveling-shock validation at S=30 (PAPER config; the S=10 '
           'default inside run_task2() is an exploration setting)  -> Table 6 + figures'),
    't3': ('studies/study_t3_cole_hopf_plateau.py',
           'T3  Cole-Hopf plateau decomposition (deterministic diagnostic, S=10)  '
           '-> Section 6 numbers + decomposition figure'),
    't4': ('studies/study_t4_heat_extended.py',
           'T4  extended heat GRW N-refinement (7 N values, S=30)  -> Table 1'),
    't5': ('studies/study_t5_fhn_extended.py',
           'T5  extended FHN N-refinement + dt quartet (S=30)  -> Table 2'),
    't6': ('studies/study_gbmc_production_n_refinement.py',
           'T6  production GBMC N-refinement (N=100..6400, S=50)  -> Tables 3-4 + figures'),
    'all': (None,
            'T1-T6 in sequence: all six paper studies at their paper configs.'),
    'figures': ('figure_scripts/regenerate_paper_figures.py',
                'regenerate the 11 title-less paper figures from the checked-in '
                'study data (no rerun) -> output/final_prepublication_tests/paper_figures/'),
}

EPILOG = """\
targets:
""" + "\n".join(f"  {k:<7} {v[1]}" for k, v in TARGETS.items()) + """
  verify  NO rerun: check the checked-in summaries under
          output/final_prepublication_tests/ against
          expected_values.json; prints PASS/FAIL per item.

notes:
  * Full reruns require scipy (tanh fits) and take roughly 10-20 minutes
    per study on a laptop. Without scipy the studies fall back to a cruder
    zero-crossing fit for the recovered shock parameters (xc, nu); the
    E_bias/E_spread/E_total error columns do not depend on the fit.
  * Function defaults inside the study modules are exploration settings.
    The paper configurations are exactly what these targets invoke
    (in particular, the T2 paper run is S=30 via studies/run_t2_S30.py).
  * Rerunning t1..t6 overwrites that study's subdirectory under
    output/final_prepublication_tests/ (t2 also emits a nu=0.2 sharp-layer
    exploration figure there that is not used in the paper).
  * Seeds: base seed 42, consecutive per ensemble member. The same seed
    identifiers are reused at each N for reproducibility; this is not a strict
    common-random-number coupling across N.
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
    print(f"\nverify: {n_pass} passed, {n_fail} failed, {n_pass + n_fail} total")
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
        for t in ('t1', 't2', 't3', 't4', 't5', 't6'):
            rc = run_target(t) or rc
        sys.exit(rc)
    sys.exit(run_target(args.target))


if __name__ == '__main__':
    main()
