# Rebuilding the coupling manuscript

Run numerical commands from the repository root after installing
`requirements-coupling.txt`. The manuscript uses relative repository paths.

```bash
python manuscript/analysis/build_numbers.py
python manuscript/analysis/verify_evidence.py
python manuscript/analysis/check_manuscript.py
python manuscript/analysis/make_figures.py
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main-2-coupling.tex
```

The compiled output is `manuscript/build/main-2-coupling.pdf`; the reviewed PDF
is also distributed at `manuscript/main-2-coupling.pdf`. A TeX installation with
`algorithm2e`, `natbib`, `tabularx`, `longtable`, and `latexmk` is needed.
The official Springer `svjour3.cls`, `svglov3.clo`, and `spmpsci.bst` are
included. The document uses JSC's `smallextended` layout; `nospthms` allows
the existing theorem environments and numbering to remain unchanged.
The old `arxiv.sty` is retained for historical versions and is no longer loaded.

See [JSC preparation notes](JSC-PREPARATION.md) for the template source,
validation, and remaining author-supplied submission details.

## Levels of reproduction

1. **Stored evidence checks:** recompute errors and confidence intervals from the
   archived arrays and block errors. No new simulation or timing is implied.
2. **Algorithm checks:** `python -m pytest -q` compares both replicas with the
   original complete-state solver using prescribed draws, verifies one-sign
   identity, and checks the final-stage identity by independent indicator
   quadrature. It also retains the legacy regression suite.
3. **Full experiments:** the round-named scripts under `studies/` reproduce
   individual designs. They write to their historical output directories;
   run a separate checkout if preserving the committed archives. Run times vary
   from seconds to tens of minutes, and floating-point/bootstrap results can
   vary across environments. Wall-clock observations are machine-specific.

Principal drivers: `study_round22_corrections.py` (field baseline),
`study_round09_costs_control.py` (controls and flux),
`study_round23_mechanism.py` (variance profiles only; old separation diagnostic
superseded), `study_round24_separation.py` (correct geometry),
`study_round17_compare.py` and `study_round23_worktarget.py` (held-out targets),
and the round-25 divergence, final-stage, and calibration drivers.
The independent replay in `evidence/replayed_fields.npz` stores both partners
in float64. Its original replay metadata is retained; manuscript bootstrap
intervals use a different recorded resampling seed, so they need not equal
those in the earlier replay report.

The archive and statistical checks establish reproducibility within their
scope. They do not certify publication readiness, prior-art exclusivity, or
confidence-interval coverage beyond the stated assumptions.
