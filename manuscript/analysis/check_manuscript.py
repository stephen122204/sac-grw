"""Cross-check the numbers printed in main-2-coupling.tex against numbers.json.

Each entry names a value in the manuscript, the path to the same value in
numbers.json, and the number of decimals the manuscript prints it to. A mismatch
is an error, not a warning.
"""

import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.load(open(HERE / "numbers.json"))
TEX = (HERE.parent / "main-2-coupling.tex").read_text()


def g(*path):
    o = N
    for p in path:
        o = o[p]
    return o


def find(v, dec, label):
    s = f"{v:.{dec}f}"
    return (s in TEX), s, label


checks = []
fr = g("field", "ratios")
for k, dec in (("WITHIN/RAW", 4), ("FULL/RAW", 4), ("FULL/WITHIN", 4)):
    checks.append(find(fr[k]["variance"], dec, f"field var {k}"))
    checks.append(find(fr[k]["variance_ci"][0], dec, f"field var lo {k}"))
    checks.append(find(fr[k]["variance_ci"][1], dec, f"field var hi {k}"))
    checks.append(find(fr[k]["mse"], dec, f"field mse {k}"))
    checks.append(find(fr[k]["mse_ci"][0], dec, f"field mse lo {k}"))
    checks.append(find(fr[k]["mse_ci"][1], dec, f"field mse hi {k}"))

c = g("cost")
checks.append(find(c["ratios_8192"]["FULL/RAW"], 3, "cost FULL/RAW 8192"))
checks.append(find(c["ratios_8192"]["FULL/WITHIN"], 3, "cost FULL/WITHIN 8192"))
checks.append(find(c["ratios_2048"]["FULL/RAW"], 3, "cost FULL/RAW 2048"))
checks.append(find(c["ratios_2048"]["FULL/WITHIN"], 3, "cost FULL/WITHIN 2048"))
checks.append(find(c["pair_over_single_8192"], 2, "pair/single"))
checks.append(find(c["variance_times_cost_FULL_over_WITHIN"], 3, "v x c"))

for r in g("controls", "rows"):
    if r["arm"] in ("FULL", "WITHIN", "RAW+CVF(c*)", "FULL+RB"):
        checks.append(find(r["ratio_vs_RAW"], 4, f"control {r['flux']} {r['arm']}"))
    if "ratio_vs_WITHIN" in r and r["arm"] == "FULL":
        checks.append(find(r["ratio_vs_WITHIN"], 4, f"FULL/WITHIN {r['flux']}"))

for r in g("mechanism", "rows"):
    checks.append(find(r["full_over_within"], 3, f"ladder {r['profile']} T={r['T']}"))

for r in g("separation", "rows"):
    if r["kind"] == "matched_state":
        checks.append(find(r["ratio"], 3, f"sep {r['profile']} T={r['T']}"))

for a, r in g("heldout_A", "arms").items():
    checks.append(find(r["work_s"], 3, f"A work {a}"))
for a, r in g("heldout_B", "arms").items():
    checks.append(find(r["work_interleaved_s"], 3, f"B work {a}"))
checks.append(
    find(g("heldout_B", "work_ratios_interleaved")["FULL/WITHIN"], 3, "B work ratio")
)

for r in g("nonlinear", "ratios"):
    if r["comparison"] in ("FULL/RAW", "FULL/WITHIN"):
        checks.append(
            find(
                r["empirical_mse_ratio"],
                3,
                f"nl {r['observable']} {r['estimator']} {r['comparison']}",
            )
        )

for r in g("applicability", "rows"):
    checks.append(find(r["ratio"], 3, f"applic {r['structure']} {r['dynamics']}"))

for r in g("final_stage_check", "share"):
    checks.append(
        find(r["guaranteed_share"] * 100, 2, f"share {r['problem']} h={r['h']}")
    )

bad = [(s, l) for ok, s, l in checks if not ok]
print(f"{len(checks) - len(bad)}/{len(checks)} values found verbatim in the manuscript")
for s, l in bad:
    print(f"  MISSING  {l:45s} expected {s}")
sys.exit(1 if bad else 0)
