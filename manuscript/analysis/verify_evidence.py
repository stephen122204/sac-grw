"""Independent arithmetic checks of central manuscript evidence (no new timing).

This supplements the printed-number scan; it recomputes from partner arrays and
held-out block errors and checks the scope/location of headline table entries.
"""

from pathlib import Path
import json
import re
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N = json.loads((HERE / "numbers.json").read_text())
TEX = (HERE.parent / "main-2-coupling.tex").read_text()
checks = 0


def close(actual, expected):
    global checks
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-15)
    checks += 1


z = np.load(HERE.parent / "evidence/replayed_fields.npz")
dx = z["x"][1] - z["x"][0]
fields = {a: (z[a + "_A"] + z[a + "_B"]) / 2 for a in ("RAW", "FULL", "WITHIN")}
var = {a: dx * np.var(f, axis=0, ddof=1).sum() for a, f in fields.items()}
mse = {
    a: np.mean(dx * ((f - z["reference"]) ** 2).sum(axis=1)) for a, f in fields.items()
}
for a in fields:
    close(var[a], N["field"]["integrated_variance"][a])
    close(mse[a], N["field"]["integrated_mse"][a])
    # Independent sample decomposition with correct finite-sample variance factor.
    bias_plugin = dx * ((fields[a].mean(0) - z["reference"]) ** 2).sum()
    close(mse[a], bias_plugin + (len(fields[a]) - 1) / len(fields[a]) * var[a])
headline = TEX[TEX.index(r"\label{tab:headline}") :]
headline = headline[: headline.index(r"\end{table}")]
for k, row in N["field"]["ratios"].items():
    a, b = k.split("/")
    close(row["variance"], var[a] / var[b])
    close(row["mse"], mse[a] / mse[b])
    # Values must occur together on the appropriate table row, not elsewhere.
    rows = headline.splitlines()
    assert any(
        f"{row['variance']:.4f}" in line
        and f"{row['mse']:.4f}" in line
        and f"{row['variance_ci'][0]:.4f}" in line
        and f"{row['variance_ci'][1]:.4f}" in line
        for line in rows
    ), k
    checks += 1

sources = {
    "A": "round17_compare_2026_09_10/compare.json",
    "B": "round23_mechanism_2026_09_10/worktarget.json",
}
for name, source in sources.items():
    raw = json.loads((ROOT / "output" / source).read_text())
    summary = N["heldout_" + name]

    # Locate saved per-block errors independently of build_numbers' summary.
    def find_blocks(obj):
        found = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    isinstance(value, list)
                    and len(value) == 32
                    and all(isinstance(v, (float, int)) for v in value)
                ):
                    found.append(np.array(value))
                else:
                    found.extend(find_blocks(value))
        elif isinstance(obj, list):
            for value in obj:
                found.extend(find_blocks(value))
        return found

    arrays = find_blocks(raw)
    assert arrays, source
    count = len(summary["arms"])
    crit = t.ppf(1 - 0.05 / (2 * count), 31)
    for arm, r in summary["arms"].items():
        candidates = [
            e
            for e in arrays
            if np.isclose(e.mean(), r["attained"], rtol=1e-10, atol=1e-15)
        ]
        assert len(candidates) == 1, (name, arm)
        e = candidates[0]
        se = e.std(ddof=1) / np.sqrt(len(e))
        close(e.mean(), r["attained"])
        close(se, r["se"])
        close(e.mean() + crit * se, r["upper_95_bonferroni"])
        assert r["upper_95_bonferroni"] < summary["target"]
        expected = r["B"] * (1 if arm in ("SINGLE", "RQMC") else 2)
        assert expected == r["trajectories"]
        checks += 2
    assert summary["pilot_key"] != summary["eval_key"]
    checks += 1

state = np.load(HERE / "fig1_state.npz")
for a in ("A", "B"):
    assert np.all(
        np.diff(state["x" + a]) >= 0
    ), "Figure 1 must show pre-diffusion ranks"
    assert np.all(state["m" + a] != 0)
    checks += 2
assert np.any(np.sign(state["mA"]) != np.sign(state["mB"]))
checks += 1
labels = re.findall(r"\\label\{([^}]+)\}", TEX)
assert len(labels) == len(set(labels)), "duplicate LaTeX labels"
refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", TEX)
assert set(refs) <= set(labels), set(refs) - set(labels)
checks += 2
report = {
    "checks": checks,
    "status": "passed",
    "scope": "partner-array arithmetic, headline table rows, both held-out block means/SEs/nominal bounds/counts, sorted construction state, LaTeX labels",
    "does_not_establish": "statistical coverage, optimal plans, universal variance ordering, novelty, or new timing",
}
(HERE.parent / "evidence/verification.json").write_text(json.dumps(report, indent=2))
print(f"{checks} independent evidence checks passed")
